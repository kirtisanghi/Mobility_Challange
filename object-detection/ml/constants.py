#!/usr/bin/env python

TEAM_GCP_PROJECT_ID = "mobility-challenge-2024"
TEAM_GCP_PROJECT_NUMBER = 520718889029
TEAM_GCS_BUCKET = "path-protectors-dataset"
OD_RAW_RESULTS_PATH = "object-detection/raw"

RED_RGB = (0, 0, 255)
YELLOW_RGB = (0, 255, 255)
GREEN_RGB = (0, 255, 0)

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

# define the variable for result dataframe columns
TURNING_PATTERN_COL = "Turning Pattern"
FRAME_COL = "Frame"
TIMESTAMP_COL = "Timestamp"
CAMERA_NAME_COL = "Camera Name"


DETECTABLE_CLASSES = {
    # 0: "person",
    # 1: "bicycle",
    # 2: "car",
    # 3: "motorcycle",
    # 5: "bus",
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
    "IndianBicycle": "Bicycle",
    "IndianBus": "Bus",
    "IndianCar": "Cars",
    "Two-Wheeler": "Two-Wheeler",
    "Three-Wheeler": "Three-Wheeler",
    "LCV": "LCV",
    "IndianTruck": "Truck"
}
TARGET_CLASS_LIST = [
    "IndianBicycle",
    "IndianBus",
    "IndianCar",
    "Two-Wheeler",
    "Three-Wheeler",
    "LCV",
    "IndianTruck"
]

GOING_UP = "Going Up"
GOING_DOWN = "Going Down"
GOING_LEFT = "Going Left"
GOING_RIGHT = "Going Right"

# with resect to the leader board camera view of
# Stn_HD_1, following directions are coded with
# along with A, B, C, D, E, F (G and H if application)
INCOMING_DOWN = "Incoming Down"
INCOMING_UP = "Incoming Up"
OUTGOING_DOWN = "Outgoing Down"
OUTGOING_UP = "Outgoing Up"
INCOMING_LEFT = "Incoming Left"
INCOMING_RIGHT = "Incoming Right"
OUTGOING_LEFT = "Outgoing Left"
OUTGOING_RIGHT = "Outgoing Right"

POSSIBLE_DIRECTIONS = [
    (INCOMING_LEFT, OUTGOING_UP),  # BC
    (INCOMING_LEFT, OUTGOING_RIGHT),  # BE
    (INCOMING_LEFT, OUTGOING_DOWN),  # BG
    (INCOMING_UP, OUTGOING_RIGHT),  # DE
    (INCOMING_UP, OUTGOING_LEFT),  # DA
    (INCOMING_UP, OUTGOING_DOWN),  # DH
    (INCOMING_RIGHT, OUTGOING_LEFT),  # FA
    (INCOMING_RIGHT, OUTGOING_UP),  # FC
    (INCOMING_RIGHT, OUTGOING_DOWN),  # FH
    (INCOMING_DOWN, OUTGOING_UP),  # GA
    (INCOMING_DOWN, OUTGOING_LEFT),  # GC
    (INCOMING_DOWN, OUTGOING_RIGHT),  # GE
]

POSSIBLE_DIRECTIONS_MAP = {
    "0": f"{INCOMING_LEFT}->{OUTGOING_UP}",
    "1": f"{INCOMING_LEFT}->{OUTGOING_RIGHT}",
    "2": f"{INCOMING_LEFT}->{OUTGOING_DOWN}",
    "3": f"{INCOMING_UP}->{OUTGOING_RIGHT}",
    "4": f"{INCOMING_UP}->{OUTGOING_LEFT}",
    "5": f"{INCOMING_UP}->{OUTGOING_DOWN}",
    "6": f"{INCOMING_RIGHT}->{OUTGOING_LEFT}",
    "7": f"{INCOMING_RIGHT}->{OUTGOING_UP}",
    "8": f"{INCOMING_RIGHT}->{OUTGOING_DOWN}",
    "9": f"{INCOMING_DOWN}->{OUTGOING_UP}",
    "10": f"{INCOMING_DOWN}->{OUTGOING_LEFT}",
    "11": f"{INCOMING_DOWN}->{OUTGOING_RIGHT}"
}



U_TURNS = [
    (INCOMING_LEFT, OUTGOING_LEFT),
    (INCOMING_UP, OUTGOING_UP),
    (INCOMING_RIGHT, OUTGOING_RIGHT)
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
