#!/usr/bin/env python

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
    ANNOTATIONS_CONFIG,
    RED_RGB,
    GREEN_RGB,
    YELLOW_RGB,
    SEQUENCE_TO_TIME_MAP,
    TEAM_GCP_PROJECT_ID,
    TEAM_GCS_BUCKET,
    OD_RAW_RESULTS_PATH
)


def is_colab_env():
    """ verifies if the runtime environment is associated to collab """
    return "COLAB_RELEASE_TAG" in os.environ


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


def load_annotations_config(config_path: str = ANNOTATIONS_CONFIG):
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
    splitted_items = file_name.split("_time_")
    return splitted_items[0]


def extract_file_name_minus_extension(file_name: str) -> str:
    splitted_items = file_name.split(".")
    return splitted_items[0]


def construct_timestamp_from_file_name(file_name: str):
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


def calculate_slope_and_intercept(starting_point, ending_point):
    """
    calculate the slope for the provided coordinates
    of starting and ending points of the line
    """
    x1, y1 = starting_point
    x2, y2 = ending_point
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - (slope * x1)
    return slope, intercept


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


def annotate_crossing_line(frame, start_coordinates, end_coordinates, origin_coordinates, msg="Crossing Line"):
    """
    annotates the frame with a crossing line
    """
    cv2.line(frame, start_coordinates, end_coordinates, RED_RGB, 4)
    cv2.putText(frame, msg, origin_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW_RGB, 2, cv2.LINE_AA)


def annotate_detection_result(frame, detected_vehicles, offset, origin_coordinates, direction):
    """
    annotates the frame with detection result
    """
    # offset = 30
    # x_axis = 100
    x_axis, y_axis = origin_coordinates
    cv2.putText(frame, direction, origin_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_RGB, 2, cv2.LINE_AA)
    for key, value in detected_vehicles[direction].items():
        y_axis += offset
        msg = f"{key}: {value}"
        cv2.putText(frame, msg, (x_axis, y_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_RGB, 2, cv2.LINE_AA)


def get_object_location_signs(start_coordinates, end_coordinates, cur_center_coordinates, prev_center_coordinates):
    """
    determines the sign of the object with respect to the provided
    line coordinates from current frame and previous frame
    """
    slope, intercept = calculate_slope_and_intercept(start_coordinates, end_coordinates)
    cur_x_center, cur_y_center = cur_center_coordinates
    prev_x_center, prev_y_center = prev_center_coordinates
    current_sign = np.sign(cur_y_center - (slope * cur_x_center + intercept))
    previous_sign = np.sign(prev_y_center - (slope * prev_x_center + intercept))
    return current_sign, previous_sign


def is_object_going_up(current_sign, previous_sign):
    """
    checks if the signature of object has changed in the recent
    consecutive frames or if it has remained negative
    """
    return (current_sign != previous_sign) or (current_sign < 0 and previous_sign < 0)


def is_object_going_down(current_sign, previous_sign):
    """
    checks if the signature of object has changed in the recent
    consecutive frames or if it has remained positive
    """
    return (current_sign != previous_sign) or (current_sign > 0 and previous_sign > 0)


def is_object_in_polygon_area(center_coordinates, polygon_points):
    """
    determines if the provided center point of the bounding box
    is within the mentioned polygonal area
    """
    result = cv2.pointPolygonTest(polygon_points, center_coordinates, measureDist=True)
    return result >= 0


def sample_and_aggregate_data(time_series_data: list, camera_name: str):
    # load the inference data as pandas dataframe
    data_frame = pd.DataFrame(time_series_data)

    # drop frame column
    data_frame.drop(columns=["frame"], inplace=True)

    # convert timestamp column into datetime type
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])

    # set timestamp column as the index
    data_frame.set_index("timestamp", inplace=True)

    # group the dataframe by direction and resample the dataframe
    # to aggregate all the numerical columns over 1 min interval
    grouped_and_sampled_df = data_frame.groupby("direction").resample("1min").sum(numeric_only=True)

    # reset the index
    grouped_and_sampled_df.reset_index(inplace=True)

    # add a column having camera name
    grouped_and_sampled_df["camera_name"] = camera_name

    return grouped_and_sampled_df


class AmazonService:
    def __init__(self, profile_name: str, s3_bucket: str = "ieee-dataport"):
        self.bucket = s3_bucket
        self.profile = profile_name

        self.session = boto3.session.Session(profile_name=self.profile)
        self.client = self.session.client("s3")

    def list_objects_with_prefix(self, prefix: str) -> list[str]:
        try:
            response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = response.get("Contents")
            if contents is None:
                print(f"no objects found in bucket \"{self.bucket}\" with prefix \"{prefix}\"")
                return []
            video_keys = [content.get("Key") for content in contents if content.get("Key").endswith(".mp4")]
            print(f"found {len(contents)} objects with prefix {prefix}, of which {len(video_keys)} are video files")
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
            print(f"signed url successfully generated for {object_key}")
            return signed_url
        except ClientError as err:
            raise Exception(f"failed to created signed url for {object_key}\n{err}")


def save_output(data_frame: pd.DataFrame, output_path: str, to_gcs: bool):
    """
    writes the provided dataframe in the given output
    either in local disk or to gcs. It uses pre-defined
    gcs bucket mentioned in the constant.py module. user
    must authenticate to google cloud before running the
    script with gcs push enabled
    """
    if to_gcs:
        date = dt.date(dt.now())
        blob_name = f"{OD_RAW_RESULTS_PATH}/{date}/{output_path}"
        client = Client(project=TEAM_GCP_PROJECT_ID)
        bucket = Bucket(client=client, name=TEAM_GCS_BUCKET)
        blob = Blob(name=blob_name, bucket=bucket)
        blob.upload_from_string(data_frame.to_csv(), 'text/csv')
    else:
        data_frame.to_csv(output_path)
