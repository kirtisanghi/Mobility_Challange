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


class SingleLane(ObjectDetectionAndTracking):
    """
    object detection class implementation for camera having
    visibility to a single lane
    """
    direction: str

    def __init__(self):
        self.line_start = ()
        self.line_end = ()
        self.line_text = ()
        self.result_origin = ()
        self.result_offset = 0
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        annotate_crossing_line(frame, self.line_start, self.line_end, self.line_text)

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.result_offset, self.result_origin, self.direction)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        cur_sign, prev_sign = get_object_location_signs(
            self.line_start,
            self.line_end,
            cur_center_coord,
            prev_center_coord
        )
        if self.direction == GOING_UP and is_object_going_up(cur_sign, prev_sign) is np.True_:
            self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)

        if self.direction == GOING_DOWN and is_object_going_down(cur_sign, prev_sign) is np.True_:
            self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_DOWN)

    def construct_tracker_dict(self):
        return {
            self.direction: {VEHICLE_CLASS_MAP[label]: 0 for label in TARGET_CLASS_LIST}
        }


class DoubleLane(ObjectDetectionAndTracking):
    """
    object detection class implementation for camera having
    visibility to a double lane, with vehicles going in
    the opposite direction with respect to both lanes
    """
    directions: list[str]

    def __init__(self):
        self.left_polygon = np.array([], np.int32)
        self.right_polygon = np.array([], np.int32)
        self.l1_start = ()
        self.l1_end = ()
        self.l1_text = ()
        self.left_result_origin = ()
        self.left_offset = 0
        self.right_result_origin = ()
        self.right_offset = 0
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        cv2.polylines(frame, [self.left_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.right_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        annotate_crossing_line(frame, self.l1_start, self.l1_end, self.l1_text)

    @abstractmethod
    def add_result_annotation(self, frame):
        pass

    @abstractmethod
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        pass

    def construct_tracker_dict(self):
        tracker = dict()
        for direction in self.directions:
            tracker[direction] = {VEHICLE_CLASS_MAP[label]: 0 for label in TARGET_CLASS_LIST}
        return tracker


class MultiLane(ObjectDetectionAndTracking):
    """
    object detection class implementation for camera having
    visibility to a multiple lanes, with vehicles going in
    all the direction.
    """

    def add_detection_annotation(self, frame):
        pass

    def add_result_annotation(self, frame):
        pass

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        pass

    def construct_tracker_dict(self):
        pass


class Camera4935(SingleLane):
    camera_name = "18th_Crs_BsStp_JN_FIX_1"
    camera_number = 4935
    site_id = 952
    direction = GOING_UP

    def __init__(self):
        super().__init__()
        self.line_start = (500, 390)
        self.line_end = (1600, 280)
        self.line_text = (450, 430)
        self.result_origin = (100, 100)
        self.result_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()


class Camera4936(DoubleLane):
    camera_name = "18th_Crs_BsStp_JN_FIX_2"
    camera_number = 4936
    site_id = 952
    directions = [GOING_UP, GOING_DOWN]

    def __init__(self):
        super().__init__()
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

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        cur_sign, prev_sign = get_object_location_signs(self.l1_start, self.l1_end, cur_center_coord, prev_center_coord)

        if is_object_in_polygon_area(cur_center_coord, self.left_polygon):
            if is_object_going_up(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)
        elif is_object_in_polygon_area(cur_center_coord, self.right_polygon):
            if is_object_going_down(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_DOWN)

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.left_offset, self.left_result_origin, GOING_UP)
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin, GOING_DOWN)


class Camera4895(SingleLane):
    camera_name = "18th_Crs_Bus_Stop_FIX_1"
    camera_number = 4895
    site_id = 954
    direction = GOING_DOWN

    def __init__(self):
        super().__init__()
        self.line_start = (50, 600)
        self.line_end = (1300, 650)
        self.line_text = (1300, 650)
        self.result_origin = (100, 100)
        self.result_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()


class Camera6645(DoubleLane):
    camera_name = "Ayyappa_Temple_FIX_1"
    camera_number = 6645
    site_id = 477
    directions = [GOING_UP, GOING_DOWN]

    def __init__(self):
        super().__init__()
        self.left_polygon = np.array([[380, 25], [920, 25], [1775, 1050], [425, 1050]], np.int32)
        self.left_polygon = self.left_polygon.reshape((-1, 1, 2))
        self.right_polygon = np.array([[950, 25], [1500, 25], [1900, 300], [1900, 1050]], np.int32)
        self.right_polygon = self.right_polygon.reshape((-1, 1, 2))
        self.l1_start = (400, 550)
        self.l1_end = (1850, 300)
        self.l1_text = (300, 600)
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.left_offset, self.left_result_origin, GOING_UP)
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin,
                                  GOING_DOWN)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        cur_sign, prev_sign = get_object_location_signs(self.l1_start, self.l1_end, cur_center_coord, prev_center_coord)

        if is_object_in_polygon_area(cur_center_coord, self.left_polygon):
            if is_object_going_up(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)
        elif is_object_in_polygon_area(cur_center_coord, self.right_polygon):
            if is_object_going_down(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_DOWN)


class Camera6170(DoubleLane):
    camera_name = "Devasandra_Sgnl_JN_FIX_1"
    camera_number = 6170
    site_id = 1132
    directions = [GOING_UP, GOING_DOWN]

    def __init__(self):
        super().__init__()
        self.left_polygon = np.array([[5, 390], [640, 5], [930, 5], [5, 850]], np.int32)
        self.left_polygon = self.left_polygon.reshape((-1, 1, 2))
        self.right_polygon = np.array([[5, 900], [950, 5], [1350, 5], [1350, 1050]], np.int32)
        self.right_polygon = self.right_polygon.reshape((-1, 1, 2))
        self.l1_start = (325, 200)
        self.l1_end = (1350, 450)
        self.l1_text = (1300, 500)
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.left_offset, self.left_result_origin, GOING_UP)
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin, GOING_DOWN)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        cur_sign, prev_sign = get_object_location_signs(self.l1_start, self.l1_end, cur_center_coord, prev_center_coord)

        if is_object_in_polygon_area(cur_center_coord, self.left_polygon):
            if is_object_going_up(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)
        elif is_object_in_polygon_area(cur_center_coord, self.right_polygon):
            if is_object_going_down(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_DOWN)


class Camera6164(DoubleLane):
    camera_name = "HP_Ptrl_Bnk_BEL_Rd_FIX_2"
    camera_number = 6164
    site_id = 1127
    directions = [GOING_UP, GOING_DOWN]

    def __init__(self):
        super().__init__()
        self.left_polygon = np.array([[670, 25], [900, 25], [1900, 1000], [850, 1050]], np.int32)
        self.left_polygon = self.left_polygon.reshape((-1, 1, 2))
        self.right_polygon = np.array([[950, 25], [1300, 25], [1900, 375], [1900, 850]], np.int32)
        self.right_polygon = self.right_polygon.reshape((-1, 1, 2))
        self.l1_start = (700, 400)
        self.l1_end = (1700, 250)
        self.l1_text = (600, 450)
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.left_offset, self.left_result_origin, GOING_UP)
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin, GOING_DOWN)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord, prev_center_coord):
        cur_sign, prev_sign = get_object_location_signs(self.l1_start, self.l1_end, cur_center_coord, prev_center_coord)

        if is_object_in_polygon_area(cur_center_coord, self.left_polygon):
            if is_object_going_up(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)
        elif is_object_in_polygon_area(cur_center_coord, self.right_polygon):
            if is_object_going_down(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_DOWN)


detection_class_map = {
    "18th_Crs_BsStp_JN_FIX_1": Camera4935,
    "18th_Crs_BsStp_JN_FIX_2": Camera4936,
    "18th_Crs_Bus_Stop_FIX_1": Camera4895,
    "Ayyappa_Temple_FIX_1": Camera6645,
    "Devasandra_Sgnl_JN_FIX_1": Camera6170,
    "HP_Ptrl_Bnk_BEL_Rd_FIX_2": Camera6164
}
