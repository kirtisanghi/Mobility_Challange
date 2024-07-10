#!/usr/bin/env python

from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

from utility import (
    extract_file_name,
    extract_camera_name,
    calculate_center_of_bounding_box,
    annotate_object_center,
    annotate_object_bounding_box, draw_grid_with_coordinates
)
from constants import (
    TARGET_CLASS_LIST,
    VEHICLE_CLASS_MAP
)
from detection import (
    detection_class_map,
    Camera4936
)


def detect_and_track(video_path: str):
    """
    initiate object detection and tracking using yolo
    model for the provided video path
    """
    # extract the camera name (unique identifier) from
    # the video file name
    video_name = extract_file_name(video_path)
    camera_name = extract_camera_name(video_name)
    detection_class = detection_class_map.get(camera_name)

    detection_class_obj = detection_class()

    # track_history = defaultdict(lambda: deque())
    # detected_vehicles = dict()
    # crossed_vehicles = list()

    # Load the YOLOv8 model
    model = YOLO("models/yolov8n.pt", verbose=True)

    # start capturing the video frames using opencv
    capture = cv2.VideoCapture(video_path)

    # Loop through all the frames within video
    while capture.isOpened():

        # Read a frame from the video
        success, frame = capture.read()

        if success:
            frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)
            print(f"Frame Number: {frame_number}")

            # Run object tracking using yolo model on the frame,
            # persisting tracks between frames
            results = model.track(frame, conf=0.6, iou=0.5, persist=True)

            # add grid to the image frame
            # frame = draw_grid_with_coordinates(frame)

            detection_class_obj.add_detection_annotation(frame)

            # every frame gives a single result, however we use a
            # for loop here to make the code readable
            # alternatively we could also use result = results[0]
            for result in results:

                # there could be a situation when no object is detected
                # within the frame, in that case `is_track` variable
                # will be set to false
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

                # combine the bounding boxes, confidence score, label and
                # track id of all the detected objects in the current frame
                # and loop through it for further processing
                for bounding_box, track_id, score, label_id in zip(bounding_boxes, track_ids, scores, labels):
                    label = trained_object_map[label_id]

                    if label in TARGET_CLASS_LIST:
                        label = VEHICLE_CLASS_MAP.get(label)
                        obj_identifier = f"{track_id}: {label}"
                        center_coordinates = calculate_center_of_bounding_box(bounding_box)
                        track = detection_class_obj.track_history[track_id]
                        track.append(center_coordinates)

                        annotate_object_center(frame, obj_identifier, center_coordinates)

                        if len(track) > 30:
                            track.popleft()

                        if track_id in detection_class_obj.crossed_vehicles:
                            annotate_object_bounding_box(frame, bounding_box)
                        else:
                            if len(track) >= 2:
                                prev_center_coordinates = track[-2]
                                detection_class_obj.track_object(
                                    frame,
                                    bounding_box,
                                    label,
                                    track_id,
                                    center_coordinates,
                                    prev_center_coordinates
                                )

                # annotate the frame with results
                detection_class_obj.add_result_annotation(frame)

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
