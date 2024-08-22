#!/usr/bin/env python

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Mapping, List

import cv2
import numpy as np

from .utility import (
    is_object_in_polygon_area,
    annotate_object_bounding_box,
    annotate_detection_result,
    environmental_variable_is_present
)
from .constants import (
    TARGET_CLASS_LIST,
    VEHICLE_CLASS_MAP,    
    AB,
    BA,
    AC,    
    BC, 
    BE, 
    BG, 
    DA, 
    DE, 
    DG, 
    FA, 
    FC, 
    FG, 
    HA, 
    HC, 
    HE,
    POLYGON_1,
    POLYGON_2,
    POLYGON_3,
    TURNING_PATTERN_COL,
    FRAME_COL,
    TIMESTAMP_COL,
    LEADER_BOARD_ENV,
    VEHICLE_ENTRY_COL,
    VEHICLE_EXIT_COL,
    POLYGON_A,
    POLYGON_B,
    POLYGON_C,
    POLYGON_D,
    POLYGON_E,
    POLYGON_F,
    POLYGON_G,
    POLYGON_H,    
    RESOLUTION
)

# setup the logger
logger = logging.getLogger(__name__)


class ObjectDetectionAndTracking(ABC):
    """
    base class for object detect and tracking
    """
    camera_name: str
    camera_number: int
    site_id: int

    def __init__(self):
        # this keeps track of the path covered by
        # each detected object by storing the center
        # point of its bounding box.
        self.track_history = defaultdict(lambda: deque())
        # this keeps track of the vehicle class type
        # of the object detected in the consecutive
        # frames by storing the track_id vs vehicle_class
        self.vehicle_class_history = defaultdict(lambda: deque())
        # stores the track id of the vehicles which
        # are marked as tracked
        #self.crossed_vehicles = list()
        self.crossed_vehicles = dict()
        # keeps the map of counter of detected vehicles
        # in each track by the vehicle type
        self.detected_vehicles = dict()
        # keeps the map of detected vehicles count within
        # each frame by the vehicle type
        self.detected_vehicles_in_frame = dict()
        # keeps the time series data of the detected vehicles
        # by each frame and vehicle type throughout the video
        self.detected_vehicles_time_series = list()

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
            #logger.debug(f"{track_id} Removing this track-id")
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
                TURNING_PATTERN_COL: key,
                TIMESTAMP_COL: timestamp,
                FRAME_COL: frame_number,
                **value
            }
            self.detected_vehicles_time_series.append(row_data)

    def _on_successful_tracking(self, frame, bounding_box, track_id, label, direction):
        
        #Remove the previous entry of trackid and direction from crossed vehicles
        if (track_id in self.crossed_vehicles):
            previous_direction = self.crossed_vehicles[track_id]
            self.crossed_vehicles.pop(track_id)
            if self.detected_vehicles[previous_direction][label]>0: 
                self.detected_vehicles[previous_direction][label] -= 1
            if self.detected_vehicles_in_frame[previous_direction][label]>0: 
                self.detected_vehicles_in_frame[previous_direction][label] -= 1
            logger.debug(f"function:_on_successful_tracking- {label}({track_id}) removed for {previous_direction} previous_direction")
            
            #Below code is required for some use-case where a vehicle travels from a,b,c,d. ab is valid, ac is invalid and in d it is not detected. 
            # In that case, anyway, the count will not increase for ad but ab at least it should get decremented
            """ areas_undergone = self.polygon_track_history[track_id]
            if len(areas_undergone)>2:
                first_pol = areas_undergone[0]

                #Take all the intermediate polygons and remove direction counts for them
                for pol in areas_undergone[1:-1]:
                    intermedate_pol = pol
                    previous_direction = direction_map[first_pol + "->" + intermedate_pol]

                    self.detected_vehicles[previous_direction][label] -= 1
                    self.detected_vehicles_in_frame[previous_direction][label] -= 1
                    logger.debug(f"function:_on_successful_tracking- {label}({track_id}) removed for {previous_direction} previous_direction") 
 """
        #Add latest direction for track_id
        #self.crossed_vehicles.append(track_id,direction)
        self.crossed_vehicles[track_id] = direction
        self.detected_vehicles[direction][label] += 1
        self.detected_vehicles_in_frame[direction][label] += 1
        logger.debug(f"function:_on_successful_tracking- {label}({track_id}) successfully annotated for {direction} direction")
        annotate_object_bounding_box(frame, bounding_box)


class DoubleLane(ObjectDetectionAndTracking):
    """
    object detection class implementation for camera having
    visibility to a double lane, with vehicles going in
    the opposite direction with respect to both lanes
    """
    directions: List[str]

    polygon_track_history = defaultdict(lambda: list()) 

    # key(s) used to group the dataframe specifying direction
    grouping_key: List[str] = [TURNING_PATTERN_COL]

    def __init__(self):
        super().__init__()
        self.A_polygon = np.array([], np.int32)
        self.B_polygon = np.array([], np.int32)
        
        self.A_polygon_name = "A"
        self.B_polygon_name = "B"
        self.A_polygon_name_coordinates = ()
        self.B_polygon_name_coordinates = ()

        self.left_result_origin = ()
        self.left_offset = 0
        self.right_result_origin = ()
        self.right_offset = 0
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()


    def add_detection_annotation(self, frame):
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)        


    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        areas_undergone = self.polygon_track_history[track_id]
        if (POLYGON_B in areas_undergone) and (POLYGON_A in areas_undergone):
            if areas_undergone.index(POLYGON_B)<areas_undergone.index(POLYGON_A):
                self._on_successful_tracking(frame, bounding_box, track_id, label, BA)
            elif areas_undergone.index(POLYGON_B)>areas_undergone.index(POLYGON_A):
                self._on_successful_tracking(frame, bounding_box, track_id, label, AB)

    def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        logger.debug(f"{polygons} polygons")
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon)
        logger.debug(f"{track_id} track_id {areas_undergone} areas_undergone")

    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.left_offset, self.left_result_origin, "AB")
        annotate_detection_result(frame, self.detected_vehicles, self.right_offset, self.right_result_origin,"BA")
        
    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)
        return areas

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
    # list of possible turns with starting and ending direction
    # in the format "INCOMING_LEFT->OUTGOING_RIGHT"
    directions: List[str]
    # map of possible turns with its alias for the specific camera
    #directions_map: Mapping[str, str]
    # keeps a track history of all the detected vehicles
    # by its track id and the direction it has crossed
    # direction_track_history = {
    #     "1": [INCOMING_UP]
    # }
    direction_track_history = defaultdict(lambda: deque())
    # keeps a track history of the polygon area the object
    # is detected in
    polygon_track_history = defaultdict(lambda: list())

    # key(s) used to group the dataframe specifying direction
    #grouping_key: List[str] = [VEHICLE_ENTRY_COL, VEHICLE_EXIT_COL]
    grouping_key: List[str] = [TURNING_PATTERN_COL]

    def __init__(self):
        super().__init__()
        """ self.l1_start = ()
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
        self.l4_text = () """

        self.A_polygon = np.array([], np.int32)
        self.B_polygon = np.array([], np.int32)
        self.C_polygon = np.array([], np.int32)
        self.D_polygon = np.array([], np.int32)
        self.E_polygon = np.array([], np.int32)
        self.F_polygon = np.array([], np.int32)
        self.G_polygon = np.array([], np.int32)
        self.H_polygon = np.array([], np.int32)

        self.A_polygon_name = "A"
        self.B_polygon_name = "B"
        self.C_polygon_name = "C"
        self.D_polygon_name = "D"
        self.E_polygon_name = "E"
        self.F_polygon_name = "F"
        self.G_polygon_name = "G"
        self.H_polygon_name = "H"
        self.A_polygon_name_coordinates = ()
        self.B_polygon_name_coordinates = ()
        self.C_polygon_name_coordinates = ()
        self.D_polygon_name_coordinates = ()
        self.E_polygon_name_coordinates = ()
        self.F_polygon_name_coordinates = ()
        self.G_polygon_name_coordinates = ()
        self.H_polygon_name_coordinates = ()

        #self.directions = List[str]

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

        self.dir7_result_origin = ()
        self.dir7_offset = 0

        self.dir8_result_origin = ()
        self.dir8_offset = 0

        self.dir9_result_origin = ()
        self.dir9_offset = 0

        self.dir10_result_origin = ()
        self.dir10_offset = 0

        self.dir11_result_origin = ()
        self.dir11_offset = 0

        self.dir12_result_origin = ()
        self.dir12_offset = 0


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

    """ @abstractmethod
    def update_polygon_track_history(self, track_id, center_coordinates):
        pass """
    
    def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon)

#Kuvempu_Circle_FIX_1
class Camera2853(DoubleLane):
    camera_name = "Kuvempu_Circle_FIX_1"
    camera_number = 2853
    site_id = 1354
    directions = ["AB","BA"]

    def __init__(self):
        super().__init__()
        self.A_polygon = np.array([[200,1], [650,1], [650,1000], [200,1000]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        self.B_polygon = np.array([[700,1], [1400,1], [1400,1000], [700,1000]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (350,50)
        self.B_polygon_name_coordinates = (1200,50)

        self.left_result_origin = (100,100)        
        self.left_offset = 20
        self.right_result_origin = (1550, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

#18th_Crs_BsStp_JN_FIX_2
class Camera4936(DoubleLane):
    camera_name = "18th_Crs_BsStp_JN_FIX_2"
    camera_number = 4936
    site_id = 952
    directions = ["AB", "BA"]

    def __init__(self):
        super().__init__()
        self.A_polygon = np.array([[5, 850], [5,475], [1900, 500], [1900, 850]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        self.B_polygon = np.array([[350, 450], [350, 100], [1900, 100], [1900, 450]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (50,750)
        self.B_polygon_name_coordinates = (450,300)

        self.left_result_origin = (100, 100)
        self.left_offset = 30
        self.right_result_origin = (1350, 100)
        self.right_offset = 30
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

#Devasandra_Sgnl_JN_FIX_1
class Camera6170(DoubleLane):
    camera_name = "Devasandra_Sgnl_JN_FIX_1"
    camera_number = 6170
    site_id = 1132
    directions = ["AB", "BA"]

    def __init__(self):
        super().__init__()
        self.A_polygon = np.array([[5,350], [1500,550], [1500,950], [5,950]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        self.B_polygon = np.array([[5,10], [1500,200], [1500,500], [5,300]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (15,850)
        self.B_polygon_name_coordinates = (15,250)
        
        self.left_result_origin = (100, 100)
        self.left_offset = 30
        self.right_result_origin = (1550, 100)
        self.right_offset = 30
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

#SBI_Bnk_JN_FIX_3
class Camera6179(DoubleLane):
    camera_name = "SBI_Bnk_JN_FIX_3"
    camera_number = 6179
    site_id = 1134
    directions = ["AB", "BA"]

    def __init__(self):
        super().__init__()
        
        self.A_polygon = np.array([[500,150], [1400,100], [1450,400], [500,400]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        self.B_polygon = np.array([[500,450], [1600,450], [1900,950], [500,950]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (550,200)
        self.B_polygon_name_coordinates = (550,500)
        
        self.left_result_origin = (100,100)
        self.right_result_origin = (1550, 100)
        self.left_offset = 30
        self.right_offset = 30
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()
    
#Sty_Wll_Ldge_FIX_3
class Camera6162(DoubleLane):
    camera_name = "Sty_Wll_Ldge_FIX_3"
    camera_number = 6162
    site_id = 1126
    directions = ["AB","BA"]

    def __init__(self):
        super().__init__()
        
        self.A_polygon = np.array([[1,1], [1250,1], [1250,250], [1,250]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        self.B_polygon = np.array([[1,300], [1250,300], [1250,550], [1,550]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (1300,100)
        self.B_polygon_name_coordinates = (1300,500)
        
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()
   
#Mattikere_JN_FIX_3
class Camera8064(DoubleLane):
    camera_name = "Mattikere_JN_FIX_3"
    camera_number = 8064
    site_id = 1210
    directions = ["AB","BA"]

    def __init__(self):
        super().__init__()

        self.A_polygon = np.array([[150,1], [650,1], [650,1000], [150,1000]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        
        self.B_polygon = np.array([[700,1], [1800,1], [1800,1000], [700,1000]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (400,100)
        self.B_polygon_name_coordinates = (1000,100)
        
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()
    

#Ayyappa_Temple_FIX_1
class Camera6645(DoubleLane):
    camera_name = "Ayyappa_Temple_FIX_1"
    camera_number = 6645
    site_id = 477
    directions = ["AB","BA"]

    def __init__(self):
        super().__init__()
       
        self.A_polygon = np.array([[300,550], [1900,550], [1900,900], [300,900]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
       
        self.B_polygon = np.array([[300,100], [1900,100], [1900,500], [300,500]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (250,750)
        self.B_polygon_name_coordinates = (250,450)
        
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

#Kuvempu_Circle_FIX_2
class Camera2854(DoubleLane):
    camera_name = "Kuvempu_Circle_FIX_2"
    camera_number = 2854
    site_id = 1354
    directions = ["AB","BA"]

    def __init__(self):
        super().__init__()
        self.A_polygon = np.array([[200,1], [700,1], [700,1000], [200,1000]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        
        self.B_polygon = np.array([[900,1], [1800,1], [1800,1000], [900,1000]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (250,50)
        self.B_polygon_name_coordinates = (850,50)
        
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()


#HP_Ptrl_Bnk_BEL_Rd_FIX_2
class Camera6164(DoubleLane):
    camera_name = "HP_Ptrl_Bnk_BEL_Rd_FIX_2"
    camera_number = 6164
    site_id = 1127
    directions = ["AB","BA"]

    def __init__(self):
        super().__init__()
        self.A_polygon = np.array([[750,500], [1900,500], [1900,800], [750,800]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        
        self.B_polygon = np.array([[600,50], [1900,50], [1900,450], [600,450]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (450,700)
        self.B_polygon_name_coordinates = (450,200)
        
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

#Ramaiah_BsStp_JN_FIX_1
class Camera8067(DoubleLane):
    camera_name = "Ramaiah_BsStp_JN_FIX_1"
    camera_number = 8067
    site_id = 1211
    directions = ["AB","BA"]

    def __init__(self):
        super().__init__()

        self.A_polygon = np.array([[30,550], [1400,550], [1400,900], [30,900]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        
        self.B_polygon = np.array([[350,150], [1700,150], [1700,530], [350,530]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (150,900)
        self.B_polygon_name_coordinates = (150,350)
        
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

#Ramaiah_BsStp_JN_FIX_2
class Camera8068(DoubleLane):
    camera_name = "Ramaiah_BsStp_JN_FIX_2"
    camera_number = 8068
    site_id = 1211
    directions = ["AB","BA"]

    def __init__(self):
        super().__init__()

        self.A_polygon = np.array([[200,430], [1900,430], [1900,900], [200,900]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))
        
        self.B_polygon = np.array([[300,50], [1880,50], [1880,400], [300,400]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (50,800)
        self.B_polygon_name_coordinates = (50,300)
        
        self.left_result_origin = (100, 100)
        self.left_offset = 20
        self.right_result_origin = (1650, 100)
        self.right_offset = 20
        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

#MS_Ramaiah_JN_FIX_2
class Camera6166(MultiLane):
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
    camera_name = "MS_Ramaiah_JN_FIX_2"
    camera_number = 6166
    site_id = 1125
    
    directions = [BC, BE, BG, DA, DE, DG, FA, FC, FG, HA, HC, HE]
        
    def __init__(self):
        super().__init__()
       
        self.A_polygon = np.array([[5,475], [275,475], [275,750], [5,750]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[5,50], [275,50], [275,450], [5,450]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.C_polygon = np.array([[500,10], [825,10], [825,250], [500, 250]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        self.D_polygon = np.array([[830,10], [1150,10], [1150,250], [830, 250]], np.int32)
        self.D_polygon = self.D_polygon.reshape((-1, 1, 2))
        
        self.E_polygon = np.array([[1450,200], [1800,200], [1800,550], [1450, 550]], np.int32)
        self.E_polygon = self.E_polygon.reshape((-1, 1, 2))

        self.F_polygon = np.array([[1450,575], [1800,575], [1800,1050], [1450, 1050]], np.int32)
        self.F_polygon = self.F_polygon.reshape((-1, 1, 2)) 

        self.G_polygon = np.array([[850,800], [1400,800], [1400,1050], [850, 1050]], np.int32)
        self.G_polygon = self.G_polygon.reshape((-1, 1, 2))

        self.H_polygon = np.array([[5,800], [800,800], [800,1050], [5,1050]], np.int32)
        self.H_polygon = self.H_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (50,525)
        self.B_polygon_name_coordinates = (50,100)
        self.C_polygon_name_coordinates = (550,60)
        self.D_polygon_name_coordinates = (880,60)
        self.E_polygon_name_coordinates = (1500,250)
        self.F_polygon_name_coordinates = (1500,650)
        self.G_polygon_name_coordinates = (900,850)
        self.H_polygon_name_coordinates = (50,850)

        # Direction BC
        self.dir1_result_origin = (25, 25)
        self.dir1_offset = 25
        # Direction FC
        self.dir2_result_origin = (275, 25)
        self.dir2_offset = 25
        # Direction HC
        self.dir3_result_origin = (25, 350)
        self.dir3_offset = 25

        # Direction BE
        self.dir4_result_origin = (1600, 25)
        self.dir4_offset = 25
        # Direction DE
        self.dir5_result_origin = (1325, 25)
        self.dir5_offset = 25
        # Direction HE
        self.dir6_result_origin = (1600, 350)
        self.dir6_offset = 25

        # Direction HA
        self.dir7_result_origin = (25, 700)
        self.dir7_offset = 25
        # Direction DA
        self.dir8_result_origin = (300, 700)
        self.dir8_offset = 25
        # Direction FA
        self.dir9_result_origin = (550, 700)
        self.dir9_offset = 25

        # Direction BG
        self.dir10_result_origin = (1050, 700)
        self.dir10_offset = 25
        # Direction DG
        self.dir11_result_origin = (1325, 700)
        self.dir11_offset = 25
        # Direction FG
        self.dir12_result_origin = (1600, 700)
        self.dir12_offset = 25

        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.D_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.E_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.F_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.G_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.H_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.D_polygon_name, self.D_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.E_polygon_name, self.E_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.F_polygon_name, self.F_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.G_polygon_name, self.G_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.H_polygon_name, self.H_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        #cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
 
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, BC)
        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, FC)
        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, HC)

        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, BE)
        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, DE)
        annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, HE)
        
        annotate_detection_result(frame, self.detected_vehicles, self.dir7_offset, self.dir7_result_origin, HA)
        annotate_detection_result(frame, self.detected_vehicles, self.dir8_offset, self.dir8_result_origin, DA)
        annotate_detection_result(frame, self.detected_vehicles, self.dir9_offset, self.dir9_result_origin, FA)

        annotate_detection_result(frame, self.detected_vehicles, self.dir10_offset, self.dir10_result_origin, BG)
        annotate_detection_result(frame, self.detected_vehicles, self.dir11_offset, self.dir11_result_origin, DG)
        annotate_detection_result(frame, self.detected_vehicles, self.dir12_offset, self.dir12_result_origin, FG)

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the sign of object in current frame and previous frame
        # with respect to all the lines
        #prev_center_coord = self.track_history[track_id][-2]
        #directions = self.direction_track_history[track_id]
        polygons = self.polygon_track_history[track_id]
        direction = ""
        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            try:
                direction = direction_map[first_polygon + "->" + last_polygon]
                self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
            except KeyError:
                logger.debug(f"{KeyError} KeyError")        
        logger.debug(f"{track_id} track_id {areas_undergone} areas_undergone {direction} direction")
        
    """ def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon) """

    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)

        if is_object_in_polygon_area(cur_center_coord, self.D_polygon):
            areas.append(POLYGON_D)

        if is_object_in_polygon_area(cur_center_coord, self.E_polygon):
            areas.append(POLYGON_E)

        if is_object_in_polygon_area(cur_center_coord, self.F_polygon):
            areas.append(POLYGON_F)

        if is_object_in_polygon_area(cur_center_coord, self.G_polygon):
            areas.append(POLYGON_G)

        if is_object_in_polygon_area(cur_center_coord, self.H_polygon):
            areas.append(POLYGON_H)
        return areas

#MS_Ramaiah_JN_FIX_1
class Camera6165(MultiLane):
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
    camera_name = "MS_Ramaiah_JN_FIX_1"
    camera_number = 6165
    site_id = 1125
    
    directions = ["AB","AC"]

    def __init__(self):
        super().__init__()

        self.A_polygon = np.array([[300, 200], [1525, 100], [1500, 400], [300, 500]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[1525, 200], [1850, 250], [1850, 1000], [1450, 1080]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.C_polygon = np.array([[5,800], [1350,800], [1350,1080], [5, 1080]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        self.A_polygon_name_coordinates = (350,250)
        self.B_polygon_name_coordinates = (1550,250)
        self.C_polygon_name_coordinates = (55,850)

        # Direction AC
        self.dir1_result_origin = (25, 200)
        self.dir1_offset = 30

        # Direction AB
        self.dir2_result_origin = (1525, 200)
        self.dir2_offset = 30

        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
 
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, "AC")
        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, "AB")
        

    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the sign of object in current frame and previous frame
        # with respect to all the lines
        #prev_center_coord = self.track_history[track_id][-2]
        #directions = self.direction_track_history[track_id]
        polygons = self.polygon_track_history[track_id]

        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            if (first_polygon == POLYGON_A) and (last_polygon == POLYGON_B):
                self._on_successful_tracking(frame, bounding_box, track_id, label, AB)
            elif (first_polygon == POLYGON_A) and (last_polygon == POLYGON_C):
                self._on_successful_tracking(frame, bounding_box, track_id, label, AC)
        logger.debug(f"{track_id} track_id {areas_undergone} areas_undergone")
            

    """ def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon) """

    
    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)
        return areas


#Mattikere_JN_FIX_2
class Camera8063(MultiLane):
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
    camera_name = "Mattikere_JN_FIX_2"
    camera_number = 8063
    site_id = 1210
    
    directions = ["BC", "BD", "CA", "CD", "DA", "DC"]
            
    def __init__(self):
        super().__init__()
       
        self.A_polygon = np.array([[150,5], [650,5], [650,350], [150,350]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[675,5], [1300,5], [1300,350], [675,350]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.C_polygon = np.array([[1450,200], [1900,200], [1900,545], [1450,500]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        self.D_polygon = np.array([[150,505], [1900,550], [1900,1080], [150,1080]], np.int32)
        self.D_polygon = self.D_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (200,55)
        self.B_polygon_name_coordinates = (725,55)
        self.C_polygon_name_coordinates = (1500,250)
        self.D_polygon_name_coordinates = (200,650)

        # Direction BD
        self.dir1_result_origin = (1100, 25)
        self.dir1_offset = 25
        # Direction BC
        self.dir2_result_origin = (1550, 25)
        self.dir2_offset = 25

        # Direction CD
        self.dir3_result_origin = (1100, 700)
        self.dir3_offset = 25
        # Direction CA
        self.dir4_result_origin = (1500, 700)
        self.dir4_offset = 25

        # Direction DA
        self.dir5_result_origin = (100, 700)
        self.dir5_offset = 25
        # Direction DC
        self.dir6_result_origin = (400, 700)
        self.dir6_offset = 25


        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.D_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.D_polygon_name, self.D_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, "BD")
        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, "BC")

        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, "CD")
        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, "CA")

        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, "DA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, "DC")
                
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the polygons, the track-id went through.
        # Get first and last polygon to identify direction
        #polygons = self.polygon_track_history[track_id]
        direction = ""
        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            try:
                direction = direction_map[first_polygon + "->" + last_polygon]
                if (track_id in self.crossed_vehicles):
                    if self.crossed_vehicles[track_id] == direction:
                        logger.debug(f"function:track_object- object {label}({track_id}) {direction} already detected and annotated for same direction")
                    else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
            except KeyError:
                logger.debug(f"{KeyError} KeyError")        
        logger.debug(f"{track_id} track_id {areas_undergone} areas_undergone {direction} direction")
        
    """ def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon) """

    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)

        if is_object_in_polygon_area(cur_center_coord, self.D_polygon):
            areas.append(POLYGON_D)

        return areas


#Devasandra_Sgnl_JN_FIX_3
class Camera6172(MultiLane):
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
    camera_name = "Devasandra_Sgnl_JN_FIX_3"
    camera_number = 6172
    site_id = 1132

    directions = ["AB", "AD", "CA", "CD", "DA", "DB"]

    def __init__(self):
        super().__init__()
       
        #self.A_polygon = np.array([[5,200], [400,200], [400,1050], [5,1050]], np.int32)
        self.A_polygon = np.array([[5,5], [199,5], [199,200], [400,200],[400,1050],[5,1050]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[200,5], [650,5], [650,199], [200,199]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        #self.C_polygon = np.array([[700,5], [1450,5], [1450,140], [700,199]], np.int32)
        self.C_polygon = np.array([[700,5], [1700,5], [1475,140], [700,199]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        self.D_polygon = np.array([[1500,150], [1710,5], [1900,5],[1900,1050], [1500,1050]], np.int32)
        self.D_polygon = self.D_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (50,250)
        self.B_polygon_name_coordinates = (250,55)
        self.C_polygon_name_coordinates = (750,55)
        self.D_polygon_name_coordinates = (1550,200)

        #self.directions = ["AB", "AD", "CA", "CD", "DA", "DB"]

        # Direction AB
        self.dir1_result_origin = (100, 25)
        self.dir1_offset = 25

        # Direction CA
        self.dir2_result_origin = (1100, 25)
        self.dir2_offset = 25
        # Direction CD
        self.dir3_result_origin = (1500, 25)
        self.dir3_offset = 25

        # Direction AD
        self.dir4_result_origin = (100, 700)
        self.dir4_offset = 25

        # Direction DA
        self.dir5_result_origin = (1100, 700)
        self.dir5_offset = 25
        # Direction DB
        self.dir6_result_origin = (1500, 700)
        self.dir6_offset = 25


        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    """ def add_detection_annotation(self, frame):
        
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.D_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.D_polygon_name, self.D_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
 """        
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, "AB")

        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, "CA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, "CD")

        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, "AD")

        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, "DA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, "DB")
                
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the polygons, the track-id went through.
        # Get first and last polygon to identify direction
        #polygons = self.polygon_track_history[track_id]
        direction = ""
        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            try:
                direction = direction_map[first_polygon + "->" + last_polygon]
                """ if (direction not in self.directions):
                    logger.debug(f"function: track_object- not a valid direction")
                    return """
                
                if (track_id in self.crossed_vehicles):
                    if self.crossed_vehicles[track_id] == direction:
                        logger.debug(f"function:track_object - {label}({track_id}) {direction} already detected and annotated for same direction")
                    else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
            except KeyError:
                logger.debug(f"function: track_object- {KeyError} KeyError")        
        logger.debug(f"function: track_object- {track_id} track_id {areas_undergone} areas_undergone {direction} direction")
        
    """ def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon) """

    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)

        if is_object_in_polygon_area(cur_center_coord, self.D_polygon):
            areas.append(POLYGON_D)

        return areas


#SBI_Bnk_JN_FIX_1
class Camera6177(MultiLane):
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
    camera_name = "SBI_Bnk_JN_FIX_1"
    camera_number = 6177
    site_id = 1134

    directions = ["AB", "AC", "BA", "BC", "CA", "CB"]
                
    def __init__(self):
        super().__init__()
       
        #self.A_polygon = np.array([[5,200], [400,200], [400,1050], [5,1050]], np.int32)
        self.A_polygon = np.array([[0,350], [1250,700], [1250,1080], [0,1080]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[300,0], [1250,0], [1250,250], [300,250]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        #self.C_polygon = np.array([[700,5], [1450,5], [1450,140], [700,199]], np.int32)
        self.C_polygon = np.array([[1500,200], [1920,200], [1920,1080], [1500,1080]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        #self.D_polygon = np.array([[1476,141], [1701,5], [1900,1050], [1500,1050]], np.int32)
        #self.D_polygon = self.D_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (50,750)
        self.B_polygon_name_coordinates = (450,50)
        self.C_polygon_name_coordinates = (1550,250)
        #self.D_polygon_name_coordinates = (1550,200)

        #self.directions = ["AB", "AD", "CA", "CD", "DA", "DB"]

        # Direction BA
        self.dir1_result_origin = (100, 25)
        self.dir1_offset = 25

        # Direction BC
        self.dir2_result_origin = (1200, 25)
        self.dir2_offset = 25
        # Direction CB
        self.dir3_result_origin = (1500, 25)
        self.dir3_offset = 25

        # Direction AB
        self.dir4_result_origin = (100, 700)
        self.dir4_offset = 25

        # Direction CA
        self.dir5_result_origin = (1200, 700)
        self.dir5_offset = 25
        # Direction AC
        self.dir6_result_origin = (1500, 700)
        self.dir6_offset = 25


        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        #cv2.polylines(frame, [self.D_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        #cv2.putText(frame, self.D_polygon_name, self.D_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
 
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, "BA")

        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, "BC")
        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, "CB")

        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, "AB")

        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, "CA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, "AC")
                
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the polygons, the track-id went through.
        # Get first and last polygon to identify direction
        #polygons = self.polygon_track_history[track_id]
        direction = ""
        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            try:
                direction = direction_map[first_polygon + "->" + last_polygon]
                if (direction not in self.directions):
                    logger.debug(f"function: track_object- not a valid direction")
                    return
                
                if (track_id in self.crossed_vehicles):
                    if self.crossed_vehicles[track_id] == direction:
                        logger.debug(f"function:track_object - {label}({track_id}) {direction} already detected and annotated for same direction")
                    else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
            except KeyError:
                logger.debug(f"function: track_object- {KeyError} KeyError")        
        logger.debug(f"function: track_object- {track_id} track_id {areas_undergone} areas_undergone {direction} direction")
        
    """ def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon) """

    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)

        #if is_object_in_polygon_area(cur_center_coord, self.D_polygon):
        #    areas.append(POLYGON_D)

        return areas


#Mattikere_JN_FIX_1
class Camera8063_1(MultiLane):
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
    camera_name = "Mattikere_JN_FIX_1"
    camera_number = 8065
    site_id = 1210

    directions = ["AB", "AD", "CA", "CD", "DA", "DB"]
                
    def __init__(self):
        super().__init__()
       
        #self.A_polygon = np.array([[5,200], [400,200], [400,1050], [5,1050]], np.int32)
        self.A_polygon = np.array([[0,50], [400,200], [400,1080], [0,1080]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[450,150], [900,150], [900,300], [450,300]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        #self.C_polygon = np.array([[700,5], [1450,5], [1450,140], [700,199]], np.int32)
        self.C_polygon = np.array([[950,200], [1400,300], [1550,550], [950,400]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        self.D_polygon = np.array([[1600,450], [1920,400], [1920,1080], [1400,1080]], np.int32)
        self.D_polygon = self.D_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (50,250)
        self.B_polygon_name_coordinates = (500,200)
        self.C_polygon_name_coordinates = (1050,300)
        self.D_polygon_name_coordinates = (1570,450)

        #self.directions = ["AB", "AD", "CA", "CD", "DA", "DB"]

        # Direction AB
        self.dir1_result_origin = (100, 25)
        self.dir1_offset = 25

        # Direction CA
        self.dir2_result_origin = (1200, 25)
        self.dir2_offset = 25
        # Direction CD
        self.dir3_result_origin = (1500, 25)
        self.dir3_offset = 25

        # Direction AD
        self.dir4_result_origin = (100, 700)
        self.dir4_offset = 25

        # Direction DA
        self.dir5_result_origin = (1200, 700)
        self.dir5_offset = 25
        # Direction DB
        self.dir6_result_origin = (1500, 700)
        self.dir6_offset = 25


        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.D_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.D_polygon_name, self.D_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, "AB")

        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, "CA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, "CD")

        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, "AD")

        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, "DA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, "DB")
                
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the polygons, the track-id went through.
        # Get first and last polygon to identify direction
        #polygons = self.polygon_track_history[track_id]
        direction = ""
        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            try:
                direction = direction_map[first_polygon + "->" + last_polygon]
                if (direction not in self.directions):
                    logger.debug(f"function: track_object- not a valid direction")
                    return
                
                if (track_id in self.crossed_vehicles):
                    if self.crossed_vehicles[track_id] == direction:
                        logger.debug(f"function:track_object - {label}({track_id}) {direction} already detected and annotated for same direction")
                    else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
            except KeyError:
                logger.debug(f"function: track_object- {KeyError} KeyError")        
        logger.debug(f"function: track_object- {track_id} track_id {areas_undergone} areas_undergone {direction} direction")
        
    """ def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon) """

    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)

        if is_object_in_polygon_area(cur_center_coord, self.D_polygon):
            areas.append(POLYGON_D)

        return areas


#Stn_HD_1
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

    directions = [ "BC", "BE", "DE", "DA", "FA", "FC"]
    incoming_direction_notAllowed = [POLYGON_A,POLYGON_E,POLYGON_C]

    def __init__(self):
        super().__init__()
       
        #self.A_polygon = np.array([[5,200], [400,200], [400,1050], [5,1050]], np.int32)
        #self.A_polygon = np.array([[0,350], [350,350], [350,1080], [0,1080]], np.int32)
        self.A_polygon = np.array([[0,200], [350,200], [350,1080], [0,1080]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[0,0], [400,0], [400,340], [0,340]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        #self.C_polygon = np.array([[700,5], [1450,5], [1450,140], [700,199]], np.int32)
        self.C_polygon = np.array([[401,0], [1000,0], [950,200], [401,200]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        self.D_polygon = np.array([[1050,0], [1650,0], [1650,200], [900,350]], np.int32)
        self.D_polygon = self.D_polygon.reshape((-1, 1, 2))

        self.E_polygon = np.array([[1700,0], [1920,0], [1920,450], [1700,450]], np.int32)
        self.E_polygon = self.E_polygon.reshape((-1, 1, 2))

        self.F_polygon = np.array([[1400,250], [1920,250], [1920,1080], [1400,1080]], np.int32)
        self.F_polygon = self.F_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (50,400)
        self.B_polygon_name_coordinates = (50,50)
        self.C_polygon_name_coordinates = (451,50)
        self.D_polygon_name_coordinates = (1100,50)
        self.E_polygon_name_coordinates = (1750,50)
        self.F_polygon_name_coordinates = (1450,500)

        #self.directions = ["AB", "AD", "CA", "CD", "DA", "DB"]

        # Direction BC
        self.dir1_result_origin = (100, 25)
        self.dir1_offset = 25

        # Direction BE
        self.dir2_result_origin = (1200, 25)
        self.dir2_offset = 25
        # Direction DE
        self.dir3_result_origin = (1500, 25)
        self.dir3_offset = 25

        # Direction DA
        self.dir4_result_origin = (100, 700)
        self.dir4_offset = 25

        # Direction FA
        self.dir5_result_origin = (1200, 700)
        self.dir5_offset = 25
        # Direction FC
        self.dir6_result_origin = (1500, 700)
        self.dir6_offset = 25


        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()
        

    def add_detection_annotation(self, frame):
        
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.D_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.E_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.F_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.D_polygon_name, self.D_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.E_polygon_name, self.E_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.F_polygon_name, self.F_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, "BC")

        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, "BE")
        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, "DE")

        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, "DA")

        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, "FA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, "FC")
                
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the polygons, the track-id went through.
        # Get first and last polygon to identify direction
        #polygons = self.polygon_track_history[track_id]
        direction = ""
        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            try:
                direction = direction_map[first_polygon + "->" + last_polygon]
                if (direction not in self.directions):
                    logger.debug(f"function: track_object- not a valid direction")
                    return
                
                if (track_id in self.crossed_vehicles):
                    if self.crossed_vehicles[track_id] == direction:
                        logger.debug(f"function:track_object - {label}({track_id}) {direction} already detected and annotated for same direction")
                    else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                else:
                        self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
                logger.debug(f"function: track_object- {track_id} track_id {areas_undergone} areas_undergone {direction} direction")
            except KeyError:
                logger.debug(f"function: track_object- {KeyError} KeyError")        
        
        
    def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        logger.debug(f"function:update_polygon_track_history - center_coordinates {center_coordinates} polygons {polygons} areas_undergone {areas_undergone} track_id {track_id}")
        for polygon in polygons:            
            logger.debug(f"function:update_polygon_track_history - len(areas_undergone) {len(areas_undergone)}  polygon in self.incoming_direction_notAllowed {polygon in self.incoming_direction_notAllowed})")
            if (len(areas_undergone) == 0) and (polygon in self.incoming_direction_notAllowed):
                logger.debug(f"function:update_polygon_track_history - Rejecting polygon {polygon} for  track-id {track_id} ")
                continue
            if polygon not in areas_undergone:
                areas_undergone.append(polygon)
                logger.debug(f"function:update_polygon_track_history - polygon {polygon} appended to areas_undergone {areas_undergone}")

    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):            
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)

        if is_object_in_polygon_area(cur_center_coord, self.D_polygon):
            areas.append(POLYGON_D)

        if is_object_in_polygon_area(cur_center_coord, self.E_polygon):
            areas.append(POLYGON_E)

        if is_object_in_polygon_area(cur_center_coord, self.F_polygon):
            areas.append(POLYGON_F)

        return areas


#18th_Crs_Bus_Stop_FIX_2--To be completed
class Camera4896(MultiLane):
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
    camera_name = "18th_Crs_Bus_Stop_FIX_2"
    camera_number = 4896
    site_id = 954
    
    directions = ["AB", "AD", "CD", "EB", "ED"]
    incoming_direction_notAllowed = [POLYGON_B,POLYGON_D]

    def __init__(self):
        super().__init__()
       
        self.A_polygon = np.array([[5,255], [250,255], [250,800], [5,800]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[300,5], [799,5], [799,250], [300,250]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        self.C_polygon = np.array([[801,5], [1300,5], [1300,250], [801,250]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        self.D_polygon = np.array([[1400,200], [1900,200], [1900,650], [1400,650]], np.int32)
        self.D_polygon = self.D_polygon.reshape((-1, 1, 2))

        self.E_polygon = np.array([[250,650], [1800,650], [1800,1000], [250,1000]], np.int32)
        self.E_polygon = self.E_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (55,300)
        self.B_polygon_name_coordinates = (350,55)
        self.C_polygon_name_coordinates = (851,55)
        self.D_polygon_name_coordinates = (1450,250)
        self.E_polygon_name_coordinates = (300,700)

        # Direction AB
        self.dir1_result_origin = (50, 25)
        self.dir1_offset = 25
        # Direction AD
        self.dir2_result_origin = (1700, 250)
        self.dir2_offset = 25
        # Direction CD
        self.dir3_result_origin = (1400, 25)
        self.dir3_offset = 25
        # Direction EB
        self.dir4_result_origin = (400, 700)
        self.dir4_offset = 25
        # Direction ED
        self.dir5_result_origin = (1500, 700)
        self.dir5_offset = 25
        

        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.D_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.E_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.D_polygon_name, self.D_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.E_polygon_name, self.E_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
 
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, "AB")
        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, "AD")

        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, "CD")
        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, "EB")

        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, "ED")
        #annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, "AB")
                
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the polygons, the track-id went through.
        # Get first and last polygon to identify direction
        polygons = self.polygon_track_history[track_id]
        direction = ""
        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            try:
                direction = direction_map[first_polygon + "->" + last_polygon]            
                self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
            except KeyError:
                logger.debug(f"{KeyError} KeyError")        
        logger.debug(f"{track_id} track_id {areas_undergone} areas_undergone {direction} direction")
      
        
        
    def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        logger.debug(f"function:update_polygon_track_history - center_coordinates {center_coordinates} polygons {polygons} areas_undergone {areas_undergone} track_id {track_id}")
        for polygon in polygons:            
            logger.debug(f"function:update_polygon_track_history - len(areas_undergone) {len(areas_undergone)}  polygon in self.incoming_direction_notAllowed {polygon in self.incoming_direction_notAllowed})")
            if (len(areas_undergone) == 0) and (polygon in self.incoming_direction_notAllowed):
                logger.debug(f"function:update_polygon_track_history - Rejecting polygon {polygon} for  track-id {track_id} ")
                continue
            if polygon not in areas_undergone:
                areas_undergone.append(polygon)
                logger.debug(f"function:update_polygon_track_history - polygon {polygon} appended to areas_undergone {areas_undergone}")


    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)

        if is_object_in_polygon_area(cur_center_coord, self.D_polygon):
            areas.append(POLYGON_D)
        
        if is_object_in_polygon_area(cur_center_coord, self.E_polygon):
            areas.append(POLYGON_E)

        return areas
    
#Mattikere_JN_HD_1
class Camera8065(MultiLane):
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
    camera_name = "Mattikere_JN_HD_1"
    camera_number = 8065
    site_id = 1210

    directions = ["AD", "AC", "BA", "BD", "CA", "CD"]
                
    def __init__(self):
        super().__init__()
       
        #self.A_polygon = np.array([[5,200], [400,200], [400,1050], [5,1050]], np.int32)
        self.A_polygon = np.array([[0,200], [400,200], [400,800], [0,800]], np.int32)
        self.A_polygon = self.A_polygon.reshape((-1, 1, 2))

        self.B_polygon = np.array([[405,25], [1500,25], [1500,300], [405,300]], np.int32)
        self.B_polygon = self.B_polygon.reshape((-1, 1, 2))

        #self.C_polygon = np.array([[700,5], [1450,5], [1450,140], [700,199]], np.int32)
        self.C_polygon = np.array([[1505,200], [1980,200], [1980,800], [1505,800]], np.int32)
        self.C_polygon = self.C_polygon.reshape((-1, 1, 2))

        self.D_polygon = np.array([[300,805], [1920,805], [1920,1080], [300,1080]], np.int32)
        self.D_polygon = self.D_polygon.reshape((-1, 1, 2))
        
        self.A_polygon_name_coordinates = (50,250)
        self.B_polygon_name_coordinates = (500,200)
        self.C_polygon_name_coordinates = (1570,450)
        self.D_polygon_name_coordinates = (900,900)

        #self.directions = ["AB", "AD", "CA", "CD", "DA", "DB"]

        # Direction AC
        self.dir1_result_origin = (100, 25)
        self.dir1_offset = 25

        # Direction CA
        self.dir2_result_origin = (1200, 25)
        self.dir2_offset = 25
        # Direction CD
        self.dir3_result_origin = (1500, 25)
        self.dir3_offset = 25

        # Direction AD
        self.dir4_result_origin = (100, 700)
        self.dir4_offset = 25

        # Direction BA
        self.dir5_result_origin = (1200, 700)
        self.dir5_offset = 25
        # Direction BD
        self.dir6_result_origin = (1500, 700)
        self.dir6_offset = 25


        self.detected_vehicles = self.construct_tracker_dict()
        self.detected_vehicles_in_frame = self.construct_tracker_dict()

    def add_detection_annotation(self, frame):
        
        cv2.polylines(frame, [self.A_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.B_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.C_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(frame, [self.D_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.putText(frame, self.A_polygon_name, self.A_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.B_polygon_name, self.B_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.C_polygon_name, self.C_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.D_polygon_name, self.D_polygon_name_coordinates, cv2.FONT_HERSHEY_TRIPLEX, 2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
    def add_result_annotation(self, frame):
        annotate_detection_result(frame, self.detected_vehicles, self.dir1_offset, self.dir1_result_origin, "AC")

        annotate_detection_result(frame, self.detected_vehicles, self.dir2_offset, self.dir2_result_origin, "CA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir3_offset, self.dir3_result_origin, "CD")

        annotate_detection_result(frame, self.detected_vehicles, self.dir4_offset, self.dir4_result_origin, "AD")

        annotate_detection_result(frame, self.detected_vehicles, self.dir5_offset, self.dir5_result_origin, "BA")
        annotate_detection_result(frame, self.detected_vehicles, self.dir6_offset, self.dir6_result_origin, "BD")
                
    def track_object(self, frame, bounding_box, label, track_id, cur_center_coord):
        # get the polygons, the track-id went through.
        # Get first and last polygon to identify direction
        polygons = self.polygon_track_history[track_id]
        direction = ""
        areas_undergone = self.polygon_track_history[track_id]
        if len(areas_undergone)>1:
            first_polygon = areas_undergone[0]
            last_polygon = areas_undergone[-1]
            try:                
                direction = direction_map[first_polygon + "->" + last_polygon]
                self._on_successful_tracking(frame, bounding_box, track_id, label, direction)
            except KeyError:
                logger.debug(f"{KeyError} KeyError")        
        logger.debug(f"{track_id} track_id {areas_undergone} areas_undergone {direction} direction")

    """ def update_polygon_track_history(self, track_id, center_coordinates):
        polygons = self._identity_polygon_area(center_coordinates)
        areas_undergone = self.polygon_track_history[track_id]
        for polygon in polygons:
            if polygon not in areas_undergone:
                areas_undergone.append(polygon) """

    def _identity_polygon_area(self, cur_center_coord):
        areas = []
        if is_object_in_polygon_area(cur_center_coord, self.A_polygon):
            areas.append(POLYGON_A)

        if is_object_in_polygon_area(cur_center_coord, self.B_polygon):
            areas.append(POLYGON_B)

        if is_object_in_polygon_area(cur_center_coord, self.C_polygon):
            areas.append(POLYGON_C)

        if is_object_in_polygon_area(cur_center_coord, self.D_polygon):
            areas.append(POLYGON_D)

        return areas


detection_class_map = {

    # double lane camera views
    #"18th_Crs_BsStp_JN_FIX_1": Camera4935,
    #"18th_Crs_Bus_Stop_FIX_1": Camera4895,
    "Kuvempu_Circle_FIX_1": Camera2853,
    "Kuvempu_Circle_FIX_2": Camera2854,
    "Sty_Wll_Ldge_FIX_3" :Camera6162,
    "Mattikere_JN_FIX_3":Camera8064,
    "Ayyappa_Temple_FIX_1":Camera6645,
    "HP_Ptrl_Bnk_BEL_Rd_FIX_2":Camera6164,
    
    "18th_Crs_BsStp_JN_FIX_2": Camera4936,
    #"Ayyappa_Temple_FIX_1": Camera6645,
    "Devasandra_Sgnl_JN_FIX_1": Camera6170,
    #"HP_Ptrl_Bnk_BEL_Rd_FIX_2": Camera6164,
    "SBI_Bnk_JN_FIX_3":Camera6179,
    "Ramaiah_BsStp_JN_FIX_1":Camera8067,
    "Ramaiah_BsStp_JN_FIX_2":Camera8068,

    # multi lane camera views
    "Stn_HD_1": Camera5816,
    "MS_Ramaiah_JN_FIX_1": Camera6165,
    "MS_Ramaiah_JN_FIX_2": Camera6166,

    "Mattikere_JN_FIX_2": Camera8063,
    "Mattikere_JN_HD_1":Camera8065,

    "18th_Crs_Bus_Stop_FIX_2": Camera4896,
    "Mattikere_JN_FIX_1": Camera8063_1,
    "Devasandra_Sgnl_JN_FIX_3": Camera6172,
    "SBI_Bnk_JN_FIX_1": Camera6177,
    

}

direction_map = {

    "pol_a->pol_b":"AB",
    "pol_a->pol_c":"AC",
    "pol_a->pol_d":"AD",
    
    
    "pol_b->pol_a":"BA",
    "pol_b->pol_c":"BC",
    "pol_b->pol_d":"BD",
    "pol_b->pol_e":"BE",
    "pol_b->pol_g":"BG",
    

    "pol_c->pol_a":"CA",
    "pol_c->pol_b":"CB",
    "pol_c->pol_d":"CD",
    
    "pol_d->pol_a":"DA",
    "pol_d->pol_b":"DB",    
    "pol_d->pol_c":"DC",
    "pol_d->pol_e":"DE",
    "pol_d->pol_g":"DG",

    "pol_f->pol_a":"FA",
    "pol_f->pol_c":"FC",
    "pol_f->pol_g":"FG",

    "pol_h->pol_a":"HA",
    "pol_h->pol_c":"HC",
    "pol_h->pol_e":"HE",

    "pol_e->pol_b":"EB",
    "pol_e->pol_d":"ED",

    
}



