#!/usr/bin/env python

ANNOTATIONS_CONFIG = "config/annotations.yaml"

RED_RGB = (0, 0, 255)
YELLOW_RGB = (0, 255, 255)
GREEN_RGB = (0, 255, 0)

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
