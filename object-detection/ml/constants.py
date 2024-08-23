#!/usr/bin/env python

TEAM_GCP_PROJECT_ID = "mobility-challenge-2k24"
TEAM_GCP_PROJECT_NUMBER = 587663586906
TEAM_GCS_BUCKET = "path-protectors-datalake"
OD_RAW_RESULTS_PATH = "object-detection/raw"

RED_RGB = (0, 0, 255)
YELLOW_RGB = (0, 255, 255)
GREEN_RGB = (0, 255, 0)
BLUE_RGB = (255, 0, 0)

COLAB_RELEASE_TAG = "COLAB_RELEASE_TAG"

# Set this environmental variable when running in
# local to show the detected frames with annotations
# export RUNNING_IN_LOCAL=True
RUNNING_IN_LOCAL = "RUNNING_IN_LOCAL"
# Set this env variable to show detection annotations
# export SHOW_DETECTION_ANNOTATIONS=True
SHOW_DETECTION_ANNOTATIONS = "SHOW_DETECTION_ANNOTATIONS"
# environmental variable to claim the script execution
# for leader board evaluation.
# NOTE: this env var will be set internally
LEADER_BOARD_ENV = "LEADER_BOARD_ENV"
# Set this environmental variable to enable debug logs
# export VERBOSE_LOGGING=True
VERBOSE_LOGGING = "VERBOSE_LOGGING"
# set this environmental variable to disable yolo verbose log
# by default yolo verbose logs are enabled
# export YOLO_VERBOSE=False
YOLO_VERBOSE = "YOLO_VERBOSE"
# detection logs file name
LOG_FILE_NAME = "object-detection.log"

# define the variable for result dataframe columns
TURNING_PATTERN_COL = "Turning Pattern"
VEHICLE_ENTRY_COL = "Vehicle Entry"
VEHICLE_EXIT_COL = "Vehicle Exit"
FRAME_COL = "Frame"
TIMESTAMP_COL = "Timestamp"
CAMERA_NAME_COL = "Camera Name"


DETECTABLE_CLASSES = {
    # 0: "person",
    # 1: "bicycle",
      2: "car",
      3: "motorcycle",
      5: "bus",
    # 7: "truck",
    80: "IndianBicycle",
    81: "IndianBicyle",
    82: "IndianBus",
    83: "IndianCar",
    84: "IndianTruck",
    85: "LCV",
    86: "Three-Wheeler",
    87: "Two-Wheeler"
}

VEHICLE_CLASS_MAP = {
    "car":"oCar",
    "motorcycle":"Motorcycle",
    "bus":"oBus",    
    "IndianBicycle": "Bicycle",
    "IndianBus": "Bus",
    "IndianCar": "Cars",
    "Two-Wheeler": "Two-Wheeler",
    "Three-Wheeler": "Three-Wheeler",
    "LCV": "LCV",
    "IndianTruck": "Truck"
}
TARGET_CLASS_LIST = [
    "car",
    "motorcycle",
    "bus",
    "IndianBicycle",
    "IndianBus",
    "IndianCar",
    "Two-Wheeler",
    "Three-Wheeler",
    "LCV",
    "IndianTruck"
]


SEQUENCE_TO_TIME_MAP = {
    "000": "07:30:00",
    "001": "07:45:00",
    "002": "08:00:00",
    "003": "08:15:00",
    "004": "08:30:00",
    "005": "08:45:00",
    "006": "09:00:00",
    "007": "09:15:00",
    "008": "09:30:00",
    "009": "09:45:00",
    "010": "10:00:00",
    "011": "10:15:00",
}

POLYGON_1 = "pol1"
POLYGON_2 = "pol2"
POLYGON_3 = "pol3"
POLYGON_4 = "pol4"

POLYGON_A = "pol_a"
POLYGON_B = "pol_b"
POLYGON_C = "pol_c"
POLYGON_D = "pol_d"
POLYGON_E = "pol_e"
POLYGON_F = "pol_f"
POLYGON_G = "pol_g"
POLYGON_H = "pol_h"
#RESOLUTION = (640,384)
RESOLUTION = (1920,1080)