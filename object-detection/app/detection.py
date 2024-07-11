#!/usr/bin/env python

from abc import ABC, abstractmethod
from collections import defaultdict, deque

import cv2
import numpy as np

from utility import (
    annotate_crossing_line,
    get_object_location_signs,
    is_object_in_polygon_area,
    is_object_going_up,
    is_object_going_down, annotate_object_bounding_box, annotate_detection_result
)
from constants import (
    TARGET_CLASS_LIST,
    VEHICLE_CLASS_MAP,
    GOING_DOWN,
    GOING_UP
)


class ObjectDetectionAndTracking(ABC):
    """
    base class for object detect and tracking
    """
    camera_name: str
    camera_number: int
    site_id: int

    track_history = defaultdict(lambda: deque())
    crossed_vehicles = list()
    detected_vehicles = dict()
    detected_vehicles_in_frame = dict()
    detected_vehicles_time_series = list()

    @abstractmethod
    def add_detection_annotation(self, frame):
        pass

    @abstractmethod
    def add_result_annotation(self, frame):
        pass

    @abstractmethod
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        pass

    @abstractmethod
    def construct_tracker_dict(self):
        """ construct the dict to store tracker results with required defaults """
        pass

    def reset_frame_tracker(self):
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def append_to_time_series(self, timestamp, frame_number):
        for key, value in self.detected_vehicles_in_frame.items():
            row_data = {
                "direction": key,
                "timestamp": timestamp,
                "frame": frame_number,
                **value
            }
            self.detected_vehicles_time_series.append(row_data)

    def _on_successful_tracking(self, frame, bounding_box, track_id, label, direction):
        self.crossed_vehicles.append(track_id)
        self.detected_vehicles[direction][label] += 1
        self.detected_vehicles_in_frame[direction][label] += 1
        annotate_object_bounding_box(frame, bounding_box)


class Camera4935(ObjectDetectionAndTracking):

    camera_name = "18th_Crs_BsStp_JN_FIX_1"
    camera_number = 4935
    site_id = 952

    def __init__(self):
        self.line_start = (500, 390)
        self.line_end = (1600, 280)
        self.line_text = (450, 430)
        self.result_origin = (100, 100)
        self.result_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def construct_tracker_dict(self):
        return {
            GOING_UP: {VEHICLE_CLASS_MAP[label]: 0 for label in TARGET_CLASS_LIST}
        }

    def add_detection_annotation(self, frame):
        annotate_crossing_line(frame, self.line_start, self.line_end, self.line_text)

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.result_offset, self.result_origin, GOING_UP)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        cur_sign, prev_sign = get_object_location_signs(
            self.line_start,
            self.line_end,
            cur_center_coord,
            prev_center_coord
        )
        if is_object_going_up(cur_sign, prev_sign) is np.True_:
            self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)


class Camera4936(ObjectDetectionAndTracking):
    camera_name = "18th_Crs_BsStp_JN_FIX_2"
    camera_number = 4936
    site_id = 952

    def __init__(self):
        self.left_polygon = np.array([[0, 500], [1150, 0], [1150, 175], [0, 840]], np.int32)
        self.left_polygon = self.left_polygon.reshape((-1, 1, 2))
        self.right_polygon = np.array([[50, 850], [1250, 100], [1490, 130], [900, 1050]], np.int32)
        self.right_polygon = self.right_polygon.reshape((-1, 1, 2))
        self.l1_start = (575, 225)
        self.l1_end = (1350, 450)
        self.l1_text = (700, 200)
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def construct_tracker_dict(self):
        return {
            GOING_DOWN: {VEHICLE_CLASS_MAP[label]: 0 for label in TARGET_CLASS_LIST},
            GOING_UP: {VEHICLE_CLASS_MAP[label]: 0 for label in TARGET_CLASS_LIST}
        }

    def add_detection_annotation(self, frame):

        # draw the polygons
        cv2.polylines(frame, [self.left_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.right_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        annotate_crossing_line(frame, self.l1_start, self.l1_end, self.l1_text)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        cur_sign, prev_sign = get_object_location_signs(self.l1_start, self.l1_end, cur_center_coord, prev_center_coord)

        if is_object_in_polygon_area(cur_center_coord, self.left_polygon):
            if is_object_going_up(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)
        elif is_object_in_polygon_area(cur_center_coord, self.right_polygon):
            if is_object_going_down(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_DOWN)
        else:
            pass

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.left_offset, self.left_result_origin, GOING_UP)
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin, GOING_DOWN)


detection_class_map = {
    "18th_Crs_BsStp_JN_FIX_1": Camera4935,
    "18th_Crs_BsStp_JN_FIX_2": Camera4936
}
