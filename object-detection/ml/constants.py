#!/usr/bin/env python

ANNOTATIONS_CONFIG = "config/annotations.yaml"
TEAM_GCP_PROJECT_ID = "mobility-challenge-2024"
TEAM_GCP_PROJECT_NUMBER = 520718889029
TEAM_GCS_BUCKET = "path-protectors-dataset"
OD_RAW_RESULTS_PATH = "object-detection/raw"

RED_RGB = (0, 0, 255)
YELLOW_RGB = (0, 255, 255)
GREEN_RGB = (0, 255, 0)

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
