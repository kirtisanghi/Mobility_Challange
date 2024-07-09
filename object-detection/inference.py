#!/usr/bin/env python
import os
from collections import defaultdict, deque

import yaml
import cv2
from ultralytics import YOLO
import numpy as np

vehicle_class_map = {
    "bicycle": "Bicycle",
    "bus": "Bus",
    "car": "Cars",
    "motorcycle": "Two-Wheeler",
    "truck": "Truck",
}
target_class_list = [
    "bicycle",
    "car",
    "bus",
    "motorcycle",
    "truck"
]

RED_RGB = (0, 0, 255)
YELLOW_RGB = (0, 255, 255)
GREEN_RGB = (0, 255, 0)


# extracts the file name from the provided fully
# qualified file path
def extract_file_name(full_file_path: str) -> str:
    return os.path.basename(full_file_path)


# reads the annotation config file and loads
# yaml config into python dictionary
def load_annotations_config(config_path: str = "config/annotations.yaml"):
    try:
        with open(file=config_path, mode="r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError or yaml.YAMLError as e:
        raise e


# extracts the camera name identifier from the video
# file name
def extract_camera_name(file_name: str) -> str:
    splitted_items = file_name.split("_time_")
    return splitted_items[0]


def draw_grid_with_coordinates(frame, step=50, color=(255, 255, 255), thickness=1, font_scale=0.4):
    """ annotates the entire frame into x-y grids with each cell of 50 pixels """
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
    """ calculate the center for the provided bounding box coordinates """
    x_min, y_min, x_max, y_max = bounding_box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    x_center = int(x_min + x_max) // 2
    y_center = int(y_min + y_max) // 2
    return x_center, y_center


def has_object_crossed_line(start_coordinates, end_coordinates, cur_center_coordinates, prev_center_coordinates, direction):
    """ determines if the object has passed the crossing line """
    slope, intercept = calculate_slope_and_intercept(start_coordinates, end_coordinates)
    cur_x_center, cur_y_center = cur_center_coordinates
    prev_x_center, prev_y_center = prev_center_coordinates
    current_sign = np.sign(cur_y_center - (slope * cur_x_center + intercept))
    previous_sign = np.sign(prev_y_center - (slope * prev_x_center + intercept))
    # if direction == "north":
    #     return (current_sign != previous_sign) or (current_sign < 0 and previous_sign < 0)
    # elif direction == "south":
    #     return (current_sign != previous_sign) or (current_sign > 0 and previous_sign > 0)
    return current_sign != previous_sign


def annotate_object_center(frame, msg, center_coordinates):
    cv2.circle(frame, center_coordinates, 4, RED_RGB, -1)
    cv2.putText(frame, msg, center_coordinates, cv2.FONT_HERSHEY_COMPLEX, 0.8, YELLOW_RGB, 2)


def annotate_object_bounding_box(frame, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), GREEN_RGB, 2)


def annotate_crossing_line(frame, start_coordinates, end_coordinates, origin_coordinates, msg):
    # msg = "Crossing Line"
    cv2.line(frame, start_coordinates, end_coordinates, RED_RGB, 4)
    cv2.putText(frame, msg, origin_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW_RGB, 2, cv2.LINE_AA)


def annotate_detection_result(frame, detected_vehicles, offset, origin_coordinates, direction):
    # offset = 30
    # x_axis = 100
    x_axis, y_axis = origin_coordinates
    cv2.putText(frame, direction, origin_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_RGB, 2, cv2.LINE_AA)
    for key, value in detected_vehicles[direction].items():
        y_axis += offset
        msg = f"{key}: {value}"
        cv2.putText(frame, msg, (x_axis, y_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_RGB, 2, cv2.LINE_AA)


def main():
    # Load the YOLOv8 model
    model = YOLO("models/yolov8n.pt", verbose=True)

    config = load_annotations_config()
    locations = config.get("locations")

    # Open the video file
    video_path = "videos/18th_Crs_BsStp_JN_FIX_2_time_2024-05-14T07:30:02_000.mp4"
    video_name = extract_file_name(video_path)
    camera_name = extract_camera_name(video_name)

    camera_configurations = locations.get(camera_name)
    directions = camera_configurations.get("directions")

    track_history = defaultdict(lambda: deque())
    detected_vehicles = dict()
    crossed_vehicles = list()
    annotations = dict()

    for direction, annotation in directions.items():
        detected_vehicles[direction] = {vehicle_class_map[label]: 0 for label in target_class_list}
        line = annotation.get("line")
        text = annotation.get("text")
        result = annotation.get("result")
        annotations[direction] = {
            "starting_point": (line.get("x1"), line.get("y1")),
            "ending_point": (line.get("x2"), line.get("y2")),
            "text_origin": (text.get("x1"), text.get("y1")),
            "result_origin": (result.get("x1"), result.get("y1")),
            "result_offset": result.get("offset")
        }

    capture = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while capture.isOpened():

        # Read a frame from the video
        success, frame = capture.read()

        if success:
            frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)

            # # add grid to the image frame
            # frame = draw_grid_with_coordinates(frame)

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, conf=0.7, iou=0.5, persist=True)

            # Get the bounding boxes, scores, labels and tracking ids
            for result in results:
                if result.boxes.is_track is False:
                    print(f"no objects detected in {frame_number} frame")
                    continue
                trained_object_map = result.names
                bounding_boxes = result.boxes.xyxy.cpu()
                scores = result.boxes.conf.cpu()
                labels = result.boxes.cls.int().cpu().tolist()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                # annotated_frame = result.plot()

                # Plot the tracks
                for bounding_box, track_id, score, label_id in zip(bounding_boxes, track_ids, scores, labels):
                    label = trained_object_map[label_id]

                    if label in target_class_list:
                        label = vehicle_class_map.get(label)
                        obj_identifier = f"{track_id}: {label}"
                        center_coordinates = calculate_center_of_bounding_box(bounding_box)
                        track = track_history[track_id]
                        track.append(center_coordinates)

                        annotate_object_center(frame, obj_identifier, center_coordinates)

                        if len(track) > 30:  # retain 30 tracks for 90 frames
                            track.popleft()

                        if track_id in crossed_vehicles:
                            annotate_object_bounding_box(frame, bounding_box)
                        else:
                            if len(track) >= 2:
                                prev_center_coordinates = track[-2]
                                for direction, annotation in annotations.items():
                                    if has_object_crossed_line(
                                            annotation.get("starting_point"),
                                            annotation.get("ending_point"),
                                            center_coordinates,
                                            prev_center_coordinates,
                                            direction) is np.True_:
                                        crossed_vehicles.append(track_id)
                                        detected_vehicles[direction][label] += 1
                                        annotate_object_bounding_box(frame, bounding_box)
                                        break

                for direction, annotation in annotations.items():
                    annotate_crossing_line(
                        frame,
                        annotation.get("starting_point"),
                        annotation.get("ending_point"),
                        annotation.get("text_origin"),
                        direction
                    )

                    annotate_detection_result(
                        frame,
                        detected_vehicles,
                        annotation.get("result_offset"),
                        annotation.get("result_origin"),
                        direction
                    )

                # Display the annotated frame
                cv2.imshow("Realtime Object Detection & Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
