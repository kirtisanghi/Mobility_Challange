#!/usr/bin/env python

ANNOTATIONS_CONFIG = "config/annotations.yaml"
TEAM_GCP_PROJECT_ID = "mobility-challenge-2024"
TEAM_GCP_PROJECT_NUMBER = 520718889029
TEAM_GCS_BUCKET = "path-protectors-dataset"
OD_RAW_RESULTS_PATH = "object-detection/raw"

RED_RGB = (0, 0, 255)
YELLOW_RGB = (0, 255, 255)
GREEN_RGB = (0, 255, 0)

ENV_VAR_TO_CHECK_COLAB_PLATFORM = "COLAB_RELEASE_TAG"

# Set this environmental variable when running in
# local to show the detected frames with annotations
# export RUNNING_IN_LOCAL=True
ENV_VAR_TO_CHECK_LOCAL_PLATFORM = "RUNNING_IN_LOCAL"

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

ALL_POSSIBLE_DIRECTIONS = [
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
