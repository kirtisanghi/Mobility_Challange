#!/usr/bin/env python

ANNOTATIONS_CONFIG = "config/annotations.yaml"

RED_RGB = (0, 0, 255)
YELLOW_RGB = (0, 255, 255)
GREEN_RGB = (0, 255, 0)

VEHICLE_CLASS_MAP = {
    "bicycle": "Bicycle",
    "bus": "Bus",
    "car": "Cars",
    "motorcycle": "Two-Wheeler",
    "truck": "Truck",
}
TARGET_CLASS_LIST = [
    "bicycle",
    "car",
    "bus",
    "motorcycle",
    "truck"
]

GOING_UP = "Going Up"
GOING_DOWN = "Going Down"
GOING_LEFT = "Going Left"
GOING_RIGHT = "Going Right"
