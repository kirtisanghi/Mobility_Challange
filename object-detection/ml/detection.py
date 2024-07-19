#!/usr/bin/env python

from abc import ABC, abstractmethod
from collections import defaultdict, deque

import cv2
import numpy as np

from .utility import (
    annotate_crossing_line,
    get_object_location_signs,
    is_object_in_polygon_area,
    is_object_going_up,
    is_object_going_down, annotate_object_bounding_box, annotate_detection_result, is_u_turn
)
from .constants import (
    TARGET_CLASS_LIST,
    VEHICLE_CLASS_MAP,
    GOING_DOWN,
    GOING_UP,
    INCOMING_UP,
    INCOMING_LEFT,
    INCOMING_RIGHT,
    OUTGOING_LEFT,
    OUTGOING_UP,
    OUTGOING_RIGHT,
    POLYGON_1,
    POLYGON_2,
    POLYGON_3,
)


class ObjectDetectionAndTracking(ABC):
    """
    base class for object detect and tracking
    """
    camera_name: str
    camera_number: int
    site_id: int

    # this keeps track of the path covered by
    # each detected object by storing the center
    # point of its bounding box.
    track_history = defaultdict(lambda: deque())

    # stores the track id of the vehicles which
    # are marked as tracked
    crossed_vehicles = list()
    # keeps the map of counter of detected vehicles
    # in each track by the vehicle type
    detected_vehicles = dict()
    # keeps the map of detected vehicles count within
    # each frame by the vehicle type
    detected_vehicles_in_frame = dict()
    # keeps the time series data of the detected vehicles
    # by each frame and vehicle type throughout the video
    detected_vehicles_time_series = list()

    @abstractmethod
    def add_detection_annotation(self, frame):
        pass

    @abstractmethod
    def add_result_annotation(self, frame):
        pass

    @abstractmethod
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        pass

    @abstractmethod
    def construct_tracker_dict(self):
        """ construct the dict to store tracker results with required defaults """
        pass

    def update_track_history(self, track_id, center_coordinates):
        track = self.track_history[track_id]
        track.append(center_coordinates)
        if len(track) > 30:
            track.popleft()

    def can_run_detection(self, track_id):
        """ detection to be run only when object is tracked at least in 2 frames """
        track = self.track_history[track_id]
        return len(track) >= 2

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

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        cur_sign, prev_sign = get_object_location_signs(
            self.line_start,
            self.line_end,
            cur_center_coord,
            self.track_history[track_id][-2]
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
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
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
    directions: list[tuple]
    # keeps a track history of all the detected vehicles
    # by its track id and the direction it has crossed
    # direction_track_history = {
    #     "1": [INCOMING_UP]
    # }
    direction_track_history = defaultdict(lambda: deque())
    # keeps a track history of the polygon area the object
    # is detected in
    polygon_track_history = defaultdict(lambda: set())

    def __init__(self):
        self.l1_start = ()
        self.l1_end = ()
        self.l1_text = ()

        self.l2_start = ()
        self.l2_end = ()
        self.l2_text = ()

        self.l3_start = ()
        self.l3_end = ()
        self.l3_text = ()

        self.l4_start = ()
        self.l4_end = ()
        self.l4_text = ()

        self.polygon_1 = np.array([], np.int32)
        self.polygon_3 = np.array([], np.int32)
        self.polygon_2 = np.array([], np.int32)
        self.polygon_4 = np.array([], np.int32)

        self.dir1_result_origin = ()
        self.dir1_offset = 0

        self.dir2_result_origin = ()
        self.dir2_offset = 0

        self.dir3_result_origin = ()
        self.dir3_offset = 0

        self.dir4_result_origin = ()
        self.dir4_offset = 0

        self.dir5_result_origin = ()
        self.dir5_offset = 0

        self.dir6_result_origin = ()
        self.dir6_offset = 0

    @abstractmethod
    def add_detection_annotation(self, frame):
        pass

    @abstractmethod
    def add_result_annotation(self, frame):
        pass

    @abstractmethod
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        pass

    def construct_tracker_dict(self):
        tracker = dict()
        for direction in self.directions:
            tracker[direction] = {VEHICLE_CLASS_MAP[label]: 0 for label in TARGET_CLASS_LIST}
        return tracker

    @abstractmethod
    def update_polygon_track_history(self, track_id, center_coordinates):
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

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        cur_sign, prev_sign = get_object_location_signs(
            self.l1_start,
            self.l1_end,
            cur_center_coord,
            self.track_history[track_id][-2]
        )
        if is_object_in_polygon_area(cur_center_coord, self.left_polygon):
            if is_object_going_up(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)
        elif is_object_in_polygon_area(cur_center_coord, self.right_polygon):
            if is_object_going_down(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_DOWN)

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.left_offset, self.left_result_origin, GOING_UP)
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin,
                                  GOING_DOWN)


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

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        cur_sign, prev_sign = get_object_location_signs(
            self.l1_start,
            self.l1_end,
            cur_center_coord,
            self.track_history[track_id][-2]
        )
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
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin,
                                  GOING_DOWN)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        cur_sign, prev_sign = get_object_location_signs(
            self.l1_start,
            self.l1_end,
            cur_center_coord,
            self.track_history[track_id][-2]
        )
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
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin,
                                  GOING_DOWN)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        cur_sign, prev_sign = get_object_location_signs(
            self.l1_start,
            self.l1_end,
            cur_center_coord,
            self.track_history[track_id][-2]
        )
        if is_object_in_polygon_area(cur_center_coord, self.left_polygon):
            if is_object_going_up(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_UP)
        elif is_object_in_polygon_area(cur_center_coord, self.right_polygon):
            if is_object_going_down(cur_sign, prev_sign) is np.True_:
                self._on_successful_tracking(frame, bounding_box, track_id, label, GOING_DOWN)


class Camera5816(MultiLane):
    """
    Detection and evaluation rules
    1. every vehicle has to cross 2 lines
    2. incoming vehicles are considered as detected only when
        it crosses the line and is part of its area polygon
    3. outgoing vehicles are considered as detected sa soon
        as it crosses the line
    4. there are 2 additional trackers introduced for multi-lane
        detection.
        a. to keep track of polygon area crossed by each vehicle
            throughout all the frames. this is tracked by default
            for all the detected objects
        b. to keep track of directions the vehicle is incoming from
            or outgoing to. this is tracked only when the detected
            object crosses the line
    5. For all incoming vehicles, we simply add the direction to
        the tracker as soon as it is detected and crossed
    6. For all outgoing vehicles, the action are taken based on the
        length of direction tracker for the specific detected object.
        a. length of list is equals to 2 - Here we verify the directions
            and if it is not a U-turn then mark it as detected and
            add it to the counter. If its a U-turn then find out
            that the object was first seen in which polygon area,
            and accordingly it decides the direction and updates
            counter
        b. length of list is non equals to 2 ( 1 or 3 or greater)
            Here also then it finds out that the object was first
            seen in which polygon area, and accordingly it decides
            the direction and updates counter
    """
    camera_name = "Stn_HD_1"
    camera_number = 5816
    site_id = 478
    BC = f"{INCOMING_LEFT}->{OUTGOING_UP}"
    BE = f"{INCOMING_LEFT}->{OUTGOING_RIGHT}"
    DE = f"{INCOMING_UP}->{OUTGOING_RIGHT}"
    DA = f"{INCOMING_UP}->{OUTGOING_LEFT}"
    FC = f"{INCOMING_RIGHT}->{OUTGOING_UP}"
    FA = f"{INCOMING_RIGHT}->{OUTGOING_LEFT}"
    directions = [BC, BE, DE, DA, FC, FA]

    def __init__(self):
        super().__init__()
        self.l1_start = (300, 100)
        self.l1_end = (300, 1080)
        self.l1_text = (50, 150)
        self.l1_result_origin = ()
        self.l1_offset = 0

        self.l2_start = (350, 100)
        self.l2_end = (1580, 125)
        self.l2_text = (350, 140)
        self.l2_result_origin = ()
        self.l2_offset = 0

        self.l3_start = (1600, 180)
        self.l3_end = (1600, 1080)
        self.l3_text = (1610, 1050)
        self.l3_result_origin = ()
        self.l3_offset = 0

        self.polygon_1 = np.array([[0, 75], [600, 75], [600, 400], [0, 400]], np.int32)
        self.polygon_1 = self.polygon_1.reshape((-1, 1, 2))

        self.polygon_2 = np.array([[1000, 0], [1580, 0], [1580, 300], [700, 300]], np.int32)
        self.polygon_2 = self.polygon_2.reshape((-1, 1, 2))

        self.polygon_3 = np.array([[800, 470], [1900, 70], [1920, 1080], [800, 1080]], np.int32)
        self.polygon_3 = self.polygon_3.reshape((-1, 1, 2))

        # Direction DA
        self.dir1_result_origin = (25, 800)
        self.dir1_offset = 20

        # Direction FA
        self.dir2_result_origin = (325, 800)
        self.dir2_offset = 20

        # Direction FC
        self.dir3_result_origin = (25, 50)
        self.dir3_offset = 20

        # Direction BC
        self.dir4_result_origin = (325, 50)
        self.dir4_offset = 20

        # Direction DE
        self.dir5_result_origin = (1400, 50)
        self.dir5_offset = 20

        # Direction BE
        self.dir6_result_origin = (1675, 50)
        self.dir6_offset = 20

        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        annotate_crossing_line(frame, self.l1_start, self.l1_end, self.l1_text, msg="")
        annotate_crossing_line(frame, self.l2_start, self.l2_end, self.l2_text, msg="")
        annotate_crossing_line(frame, self.l3_start, self.l3_end, self.l3_text, msg="")
        cv2.polylines(frame, [self.polygon_1], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.polygon_2], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.polygon_3], isClosed=True, color=(255, 0, 0), thickness=3)

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, self.DA)
        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, self.FA)
        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, self.FC)
        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, self.BC)
        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, self.DE)
        annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, self.BE)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the sign of object in current frame and previous frame
        # with respect to all the lines
        prev_center_coord = self.track_history[track_id][-2]
        directions = self.direction_track_history[track_id]
        polygons = list(self.polygon_track_history[track_id])
        if self._is_incoming_up(cur_center_coord, prev_center_coord) and len(directions) == 0:
            if INCOMING_UP not in directions:
                print(f"{track_id} _is_incoming_up")
                directions.append(INCOMING_UP)

        elif self._is_incoming_left(cur_center_coord, prev_center_coord) and len(directions) == 0:
            if INCOMING_LEFT not in directions:
                print(f"{track_id} _is_incoming_left")
                directions.append(INCOMING_LEFT)

        elif self._is_incoming_right(cur_center_coord, prev_center_coord) and len(directions) == 0:
            if INCOMING_RIGHT not in directions:
                print(f"{track_id} _is_incoming_right")
                directions.append(INCOMING_RIGHT)

        elif self._is_outgoing_up(cur_center_coord, prev_center_coord) and len(directions) >= 1:
            if OUTGOING_UP not in directions:
                print(f"{track_id} _is_outgoing_up")
                directions.append(OUTGOING_UP)

                # if length of directions is 2, then we have detected
                # the vehicle and got its direction
                if len(directions) == 2:
                    direction = f"{directions[0]}->{directions[1]}"
                    print(f"{track_id} directions {direction}")

                    # if it is detected as U-turn, check the first polygon
                    # that it was part of
                    if is_u_turn(directions):
                        print(f"u-turn {track_id} {polygons}")
                        self._inspect_polygon_for_outgoing_up(frame, bounding_box, track_id, label, polygons)
                    else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                else:
                    print(f"multi-directions {track_id} {polygons}")
                    self._inspect_polygon_for_outgoing_up(frame, bounding_box, track_id, label, polygons)

        elif self._is_outgoing_left(cur_center_coord, prev_center_coord) and len(directions) >= 1:
            # get the direction
            if OUTGOING_LEFT not in directions:
                print(f"{track_id} _is_outgoing_left")
                directions.append(OUTGOING_LEFT)

                # if length of directions is 2, then we have detected
                # the vehicle and got its direction
                if len(directions) == 2:
                    direction = f"{directions[0]}->{directions[1]}"
                    print(f"{track_id} directions {direction}")

                    # if it is detected as U-turn, check the first polygon
                    # that it was part of
                    if is_u_turn(directions):
                        print(f"u-turn {track_id} {polygons}")
                        self._inspect_polygon_for_outgoing_left(frame, bounding_box, track_id, label, polygons)
                    else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                else:
                    print(f"multi-directions {track_id} {polygons}")
                    self._inspect_polygon_for_outgoing_left(frame, bounding_box, track_id, label, polygons)

        elif self._is_outgoing_right(cur_center_coord, prev_center_coord) and len(directions) >= 1:
            if OUTGOING_RIGHT not in directions:
                print(f"{track_id} _is_outgoing_right")
                directions.append(OUTGOING_RIGHT)

                # if length of directions is 2, then we have detected
                # the vehicle and got its direction
                if len(directions) == 2:
                    direction = f"{directions[0]}->{directions[1]}"
                    print(f"{track_id} directions {direction}")

                    # if it is detected as U-turn, check the first polygon
                    # that it was part of
                    if is_u_turn(directions):
                        print(f"u-turn {track_id} {polygons}")
                        self._inspect_polygon_for_outgoing_right(frame, bounding_box, track_id, label, polygons)
                    else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                else:
                    print(f"multi-directions {track_id} {polygons}")
                    self._inspect_polygon_for_outgoing_right(frame, bounding_box, track_id, label, polygons)

    def _inspect_polygon_for_outgoing_right(self, frame, bounding_box, track_id, label, polygons):
        # corner case
        if len(polygons) == 0:
            return
        elif polygons[0] in [POLYGON_1]:
            self._on_successful_tracking(frame, bounding_box, track_id, label, self.BE)
        elif polygons[0] in [POLYGON_2]:
            self._on_successful_tracking(frame, bounding_box, track_id, label, self.DE)

    def _inspect_polygon_for_outgoing_left(self, frame, bounding_box, track_id, label, polygons):
        # corner case
        if len(polygons) == 0:
            return
        elif polygons[0] in [POLYGON_2]:
            self._on_successful_tracking(frame, bounding_box, track_id, label, self.DA)
        elif polygons[0] in [POLYGON_3]:
            self._on_successful_tracking(frame, bounding_box, track_id, label, self.FA)

    def _inspect_polygon_for_outgoing_up(self, frame, bounding_box, track_id, label, polygons):
        # corner case
        if len(polygons) == 0:
            return
        elif polygons[0] in [POLYGON_1]:
            self._on_successful_tracking(frame, bounding_box, track_id, label, self.BC)
        elif polygons[0] in [POLYGON_3]:
            self._on_successful_tracking(frame, bounding_box, track_id, label, self.FC)

    def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            areas_undergone.add(polygon)

    def _is_incoming_left(self, cur_center_coord, prev_center_coord):
        l1_cur_sign, l1_prev_sign = get_object_location_signs(
            self.l1_start,
            self.l1_end,
            cur_center_coord,
            prev_center_coord
        )
        return (is_object_in_polygon_area(cur_center_coord, self.polygon_1)
                and is_object_going_down(l1_cur_sign, l1_prev_sign) is np.True_)

    def _is_outgoing_left(self, cur_center_coord, prev_center_coord):
        l1_cur_sign, l1_prev_sign = get_object_location_signs(
            self.l1_start,
            self.l1_end,
            cur_center_coord,
            prev_center_coord
        )
        return is_object_going_up(l1_cur_sign, l1_prev_sign) is np.True_

    def _is_incoming_up(self, cur_center_coord, prev_center_coord):
        l2_cur_sign, l2_prev_sign = get_object_location_signs(
            self.l2_start,
            self.l2_end,
            cur_center_coord,
            prev_center_coord
        )
        return (is_object_in_polygon_area(cur_center_coord, self.polygon_2)
                and is_object_going_down(l2_cur_sign, l2_prev_sign) is np.True_)

    def _is_outgoing_up(self, cur_center_coord, prev_center_coord):
        l2_cur_sign, l2_prev_sign = get_object_location_signs(
            self.l2_start,
            self.l2_end,
            cur_center_coord,
            prev_center_coord
        )
        return is_object_going_up(l2_cur_sign, l2_prev_sign) is np.True_

    def _is_incoming_right(self, cur_center_coord, prev_center_coord):
        l3_cur_sign, l3_prev_sign = get_object_location_signs(
            self.l3_start,
            self.l3_end,
            cur_center_coord,
            prev_center_coord
        )
        return ((is_object_in_polygon_area(cur_center_coord, self.polygon_3))
                and is_object_going_up(l3_cur_sign, l3_prev_sign) is np.True_)

    def _is_outgoing_right(self, cur_center_coord, prev_center_coord):
        l3_cur_sign, l3_prev_sign = get_object_location_signs(
            self.l3_start,
            self.l3_end,
            cur_center_coord,
            prev_center_coord
        )
        return is_object_going_down(l3_cur_sign, l3_prev_sign) is np.True_

    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.polygon_1):
            areas.append(POLYGON_1)

        if is_object_in_polygon_area(cur_center_coord, self.polygon_2):
            areas.append(POLYGON_2)

        if is_object_in_polygon_area(cur_center_coord, self.polygon_3):
            areas.append(POLYGON_3)
        return areas


detection_class_map = {
    "18th_Crs_BsStp_JN_FIX_1": Camera4935,
    "18th_Crs_BsStp_JN_FIX_2": Camera4936,
    "18th_Crs_Bus_Stop_FIX_1": Camera4895,
    "Ayyappa_Temple_FIX_1": Camera6645,
    "Devasandra_Sgnl_JN_FIX_1": Camera6170,
    "HP_Ptrl_Bnk_BEL_Rd_FIX_2": Camera6164,
    "Stn_HD_1": Camera5816
}
