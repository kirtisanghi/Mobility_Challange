#!/usr/bin/env python
from collections import defaultdict, deque

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


# Function to draw grid and coordinates
def draw_grid_with_coordinates(frame, step=50, color=(255, 255, 255), thickness=1, font_scale=0.4):
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


def has_object_crossed_line(start_coordinates, end_coordinates, x_center_cur, y_center_cur, x_center_prev,
                            y_center_prev):
    """ determines if the object has passed the crossing line """
    slope, intercept = calculate_slope_and_intercept(start_coordinates, end_coordinates)
    current_sign = np.sign(y_center_cur - (slope * x_center_cur + intercept))
    previous_sign = np.sign(y_center_prev - (slope * x_center_prev + intercept))
    return (current_sign != previous_sign) or (current_sign < 0 and previous_sign < 0)


def main():
    # Load the YOLOv8 model
    model = YOLO("models/yolov8n.pt", verbose=True)

    # Open the video file
    video_path = "videos/18th_Crs_BsStp_JN_FIX_1_time_2024-05-14T07:30:02_000.mp4"
    capture = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: deque())
    frame_number = 0
    detected_vehicles = {vehicle_class_map[label]: 0 for label in target_class_list}
    crossed_vehicles = list()
    start_point = (500, 390)  # (450, 400)
    end_point = (1600, 280)  # (1750, 400)
    crossing_line_text_origin = (450, 430)

    # Loop through the video frames
    while capture.isOpened():

        # Read a frame from the video
        success, frame = capture.read()

        if success:
            frame_number += 1

            # add grid to the image frame
            # frame = draw_grid_with_coordinates(frame)

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Get the bounding boxes, scores, labels and tracking ids
            for result in results:
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
                        x_min, y_min, x_max, y_max = bounding_box
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                        x_center = int(x_min + x_max) // 2
                        y_center = int(y_min + y_max) // 2
                        track = track_history[track_id]
                        track.append((x_center, y_center))  # x, y center point

                        cv2.circle(frame, (x_center, y_center), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"{track_id}: {label}", (x_center, y_center), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                    (0, 255, 255), 2)

                        if len(track) > 30:  # retain 30 tracks for 90 frames
                            track.popleft()

                        if track_id in crossed_vehicles:
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        else:
                            if len(track) >= 2 and (
                                    has_object_crossed_line(start_point, end_point, x_center, y_center, track[-2][0],
                                                            track[-2][1]) is np.True_):
                                # (track[-2][1] > 400 > y_center) or (track[-2][1] < 400 and y_center < 400)):
                                crossed_vehicles.append(track_id)
                                detected_vehicles[label] += 1
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                red_color = (0, 0, 255)
                yellow_color = (0, 255, 255)

                cv2.line(frame, start_point, end_point, red_color, 5)
                cv2.putText(frame, 'Crossing Line', crossing_line_text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1,
                            yellow_color, 4,
                            cv2.LINE_AA)
                offset = 30
                x_axis = 100
                y_axis = 100
                for key, value in detected_vehicles.items():
                    cv2.putText(frame, f"{key}: {value}", (x_axis, y_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow_color,
                                2,
                                cv2.LINE_AA)
                    y_axis += offset

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
