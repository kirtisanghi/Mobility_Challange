#!/usr/bin/env python

import logging
import os
from datetime import datetime as dt

import yaml
import cv2
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from google.cloud.storage import Client, Blob, Bucket

from .constants import (
    RED_RGB,
    GREEN_RGB,
    YELLOW_RGB,
    SEQUENCE_TO_TIME_MAP,
    TEAM_GCP_PROJECT_ID,
    TEAM_GCS_BUCKET,
    OD_RAW_RESULTS_PATH,
    COLAB_RELEASE_TAG,
    FRAME_COL,
    TIMESTAMP_COL,
    CAMERA_NAME_COL,
    TURNING_PATTERN_COL,
    VERBOSE_LOGGING,
    RUNNING_IN_LOCAL
)

# setup the logger
logger = logging.getLogger(__name__)


def is_colab_env():
    """ verifies if the runtime environment is associated to collab """
    return COLAB_RELEASE_TAG in os.environ


def environmental_variable_is_present(env_var: str):
    """
    verifies if the provided variable is present in system
    environmental variable
    """
    return env_var in os.environ


def extract_file_name(full_file_path: str) -> str:
    """
    extracts the file name from the provided fully
    qualified file path
    """
    return os.path.basename(full_file_path)


def load_annotations_config(config_path: str):
    """
    reads the annotation config file and loads
    yaml config into python dictionary
    """
    try:
        with open(file=config_path, mode="r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError or yaml.YAMLError as e:
        raise e


def extract_camera_name(file_name: str) -> str:
    """
    extracts the camera name identifier from the video
    file name
    """
    file_name = file_name.removesuffix(".mp4")
    if "_time_" in file_name:
        splitted_items = file_name.split("_time_")
        return splitted_items[0]
    return file_name


def extract_file_name_minus_extension(file_name: str) -> str:
    splitted_items = file_name.split(".")
    return splitted_items[0]


def construct_timestamp_from_file_name(file_name: str):
    if environmental_variable_is_present(RUNNING_IN_LOCAL):
        file_name = file_name.replace('&',':')

    _, timestamp_part = file_name.split("_time_")
    timestamp_part = timestamp_part.removesuffix(".mp4")
    timestamp_str, index = timestamp_part.split("_")
    timestamp_dt = dt.fromisoformat(timestamp_str)
    constructed_timestamp_str = f"{timestamp_dt.date()}T{SEQUENCE_TO_TIME_MAP.get(index)}"
    constructed_timestamp = dt.fromisoformat(constructed_timestamp_str)
    return constructed_timestamp


def draw_grid_with_coordinates(frame, step=50, color=(255, 255, 255), thickness=1, font_scale=0.4):
    """
    annotates the entire frame into x-y grids with each cell of 50 pixels
    """
    height, width, _ = frame.shape
    # Draw vertical lines
    for x in range(0, width, step):
        cv2.line(frame, (x, 0), (x, height), color, thickness)
        cv2.putText(frame, str(x), (x, 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    # Draw horizontal lines
    for y in range(0, height, step):
        cv2.line(frame, (0, y), (width, y), color, thickness)
        cv2.putText(frame, str(y), (0, y + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def calculate_center_of_bounding_box(bounding_box):
    """
    calculate the center for the provided bounding box
    coordinates
    """
    x_min, y_min, x_max, y_max = bounding_box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    x_center = int(x_min + x_max) // 2
    y_center = int(y_min + y_max) // 2
    return x_center, y_center


def annotate_object_center(frame, msg, center_coordinates):
    """
    annotates the center of a bounding box within the frame
    with red dot
    """
    cv2.circle(frame, center_coordinates, 4, RED_RGB, -1)
    cv2.putText(frame, msg, center_coordinates, cv2.FONT_HERSHEY_COMPLEX, 0.8, YELLOW_RGB, 2)


def annotate_object_bounding_box(frame, bounding_box):
    """
    annotates the bounding box within the frame with
    green mark
    """
    x_min, y_min, x_max, y_max = bounding_box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), GREEN_RGB, 2)



def annotate_detection_result(frame, detected_vehicles, offset, origin_coordinates, direction):
    """
    annotates the frame with detection result
    """
    # offset = 30
    # x_axis = 100
    x_axis, y_axis = origin_coordinates
    cv2.putText(frame, direction, origin_coordinates, cv2.FONT_HERSHEY_DUPLEX, 1, YELLOW_RGB, 2, cv2.LINE_AA)
    for key, value in detected_vehicles[direction].items():
        y_axis += offset
        msg = f"{key}: {value}"
        cv2.putText(frame, msg, (x_axis, y_axis), cv2.FONT_HERSHEY_DUPLEX, 1, YELLOW_RGB, 2, cv2.LINE_AA)


def is_object_in_polygon_area(center_coordinates, polygon_points):
    """
    determines if the provided center point of the bounding box
    is within the mentioned polygonal area
    """
    result = cv2.pointPolygonTest(polygon_points, center_coordinates, measureDist=True)
    return result >= 0


def sample_and_aggregate_data(time_series_data: list, camera_name: str, grouping_key: list):
    # load the inference data as pandas dataframe
    data_frame = pd.DataFrame(time_series_data)

    # when in debug mode, dump entire timeseries dataframe
    # into a csv file
    if environmental_variable_is_present(VERBOSE_LOGGING):
        debug_file_name = f"{camera_name}-debug-dataframe.csv"
        save_output(data_frame, debug_file_name, to_gcs=False)

    # drop frame column
    data_frame.drop(columns=[FRAME_COL], inplace=True)

    process_dataframe(data_frame)

    # convert timestamp column into datetime type
    data_frame[TIMESTAMP_COL] = pd.to_datetime(data_frame[TIMESTAMP_COL])

    # set timestamp column as the index
    data_frame.set_index(TIMESTAMP_COL, inplace=True)

    # group the dataframe by direction and resample the dataframe
    # to aggregate all the numerical columns over 1 min interval
    grouped_and_sampled_df = data_frame.groupby(grouping_key).resample("1min").sum(numeric_only=True)

    # reset the index
    grouped_and_sampled_df.reset_index(inplace=True)

    # add a column having camera name
    grouped_and_sampled_df[CAMERA_NAME_COL] = camera_name

    return grouped_and_sampled_df


def include_leaderboard_submission_guidelines(data_frame: pd.DataFrame):
    # drop the camera name and timestamp column
    data_frame.drop(columns=[CAMERA_NAME_COL, TIMESTAMP_COL], inplace=True)

    # group by turning points and aggregate numerical column
    return data_frame.groupby(TURNING_PATTERN_COL).sum(numeric_only=True)


class AmazonService:
    def __init__(self, s3_bucket: str = "ieee-dataport"):
        self.bucket = s3_bucket
        self.session = boto3.session.Session()
        self.client = self.session.client("s3")
        logger.debug(f"AmazonService object initialization completed")

    def list_objects_with_prefix(self, prefix: str) -> list[str]:
        try:
            response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = response.get("Contents")
            if contents is None:
                logger.warning(f"no objects found in bucket \"{self.bucket}\" with prefix \"{prefix}\"")
                return []
            video_keys = [content.get("Key") for content in contents if content.get("Key").endswith(".mp4")]
            logger.info(f"found {len(contents)} objects with prefix {prefix}, of which {len(video_keys)} are video files")
            return video_keys
        except ClientError as err:
            raise Exception(f"an error occurred while listing objects in {self.bucket} with prefix \"{prefix}\"\n{err}")

    def generate_signed_url(self, object_key: str) -> str:
        try:
            signed_url = self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": object_key
                }
            )
            logger.info(f"signed url successfully generated for {object_key}")
            return signed_url
        except ClientError as err:
            raise Exception(f"failed to created signed url for {object_key}\n{err}")


def save_output(data_frame: pd.DataFrame, output_path: str, to_gcs: bool, index: str = None):
    """
    writes the provided dataframe in the given output
    either in local disk or to gcs. It uses pre-defined
    gcs bucket mentioned in the constant.py module. user
    must authenticate to google cloud before running the
    script with gcs push enabled
    """
    if to_gcs:
        blob_name = f"{OD_RAW_RESULTS_PATH}/{index}/{output_path}"
        client = Client(project=TEAM_GCP_PROJECT_ID)
        bucket = Bucket(client=client, name=TEAM_GCS_BUCKET)
        blob = Blob(name=blob_name, bucket=bucket)
        blob.upload_from_string(data_frame.to_csv(), 'text/csv')
    else:
        data_frame.to_csv(output_path)


def process_dataframe(time_series_df:pd.DataFrame):
    grouped_data = time_series_df.groupby(TURNING_PATTERN_COL)    

    #Get the count of car, oCar, Two-Wheeler,Motorcycle and bus,oBus for the turning patterns
    #car is the class from customized yolo while oCar is original yolo class. Same for two-wheeler and bus also
    turningpattern_car_count = grouped_data["Cars"].sum()
    turningpattern_ocar_count = grouped_data["oCar"].sum()
    turningpattern_TwoWheeler_count = grouped_data["Two-Wheeler"].sum()
    turningpattern_Motorcycle_count = grouped_data["Motorcycle"].sum()
    turningpattern_bus_count = grouped_data["Bus"].sum()
    turningpattern_obus_count = grouped_data["oBus"].sum()

    carcount = turningpattern_car_count.to_frame()
    ocarcount = turningpattern_ocar_count.to_frame()
    carcount["car_diff"] = carcount["Cars"] - ocarcount["oCar"]

    twoWheeler_count = turningpattern_TwoWheeler_count.to_frame()
    motorCycle_count = turningpattern_Motorcycle_count.to_frame()
    twoWheeler_count["twoWheeler_diff"] = twoWheeler_count["Two-Wheeler"] - motorCycle_count["Motorcycle"]

    bus_count = turningpattern_bus_count.to_frame()
    obus_count = turningpattern_obus_count.to_frame()
    bus_count["bus_diff"] = bus_count["Bus"] - obus_count["oBus"]

    #If, for a specific turning pattern, count is more for original Yolo class than customized yolo class, it means original 
    #Yolo class is working better for the particular direction. So use that data instead
    for key,value in carcount.iterrows():
        turning_pattern = key
        row_values = value
        print(row_values["car_diff"])
        if (row_values["car_diff"]<0):
            time_series_df["Cars"] = np.where((time_series_df["Turning Pattern"] == turning_pattern), time_series_df["oCar"], time_series_df["Cars"])

    for key,value in twoWheeler_count.iterrows():
        turning_pattern = key
        row_values = value
        print(row_values["twoWheeler_diff"])
        if (row_values["twoWheeler_diff"]<0):
            time_series_df["Two-Wheeler"] = np.where((time_series_df["Turning Pattern"] == turning_pattern), time_series_df["Motorcycle"], time_series_df["Two-Wheeler"])

    for key,value in bus_count.iterrows():
        turning_pattern = key
        row_values = value
        print(row_values["bus_diff"])
        if (row_values["bus_diff"]<0):
            time_series_df["Bus"] = np.where((time_series_df["Turning Pattern"] == turning_pattern), time_series_df["oBus"], time_series_df["Bus"])

    #Finally remove the original YOLO data from dataframe
    time_series_df.drop(columns=["oCar","Motorcycle","oBus"],axis=1,inplace=True)

