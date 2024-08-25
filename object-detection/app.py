#!/usr/bin/env python

"""
Team    - Path Protectors
Authors - Shrutika Ghosh <shrutikaghosh@gmail.com>
          Kirti Sanghi <kirti.06@gmail.com>
          Prashant Hinger <prashanthinger7@gmail.com>
          Sandeep Upadhyay <sandeepupadhyay732@gmail.com>
Date    - 14th July 2024

This script is the entrypoint for the ml python package which
has all the backend code for object detection, data processing
of the inferred objects and the utility modules.

`app.py` module internally calls the `ml` package with
required input arguments. This module to be used by the
evaluation team on code submission for leaderboard

This package can be run from command line from `object-detection`
folder with 2 required arguments
    1. path the video which should be used by the ml model for
       object detection
    2. path to the csv file where the detection results should be
       saved.

python app.py Stn_HD_1.mp4 Output_Turning_Patterns.csv
"""
import json
import os
import sys
from ml.main import main as main_prog
from ml.constants import LEADER_BOARD_ENV
import logging

logger = logging.getLogger(__name__)


def main():
    try:
        file_name, input_file, output_file = sys.argv[0], sys.argv[1], sys.argv[2]
    except IndexError:
        exit(
            f"[warn]: required arguments missing, please provide all the required arguments "
            f"and execute the script in following way\n\n"
            f"python app.py {{input_file.json}} {{output_file.json}}\n\n"
            f"input_file      -> {'Cam_ID': {'Vid_1': '/path_to_vid_1', 'Vid_2': '/path_to_vid_2'}}\n"
            f"                     https://drive.google.com/file/d/19EMxIqMXlRSYdd1uJ_d_U2fgIN4aMy7C/view"
            f"output_file     -> {'Cam_ID': {'Cumulative Counts': counts_by_class_turning_pattern, 'Predicted Counts': counts_by_class_turning_pattern }}"
            f"                     https://drive.google.com/file/d/19HL-Anae3SjFIwVypeJ-372L7teczzdd/view"
        )
    if input_file is None or input_file.strip() == "":
        exit(f"[error]: please provide a valid local path to the input file")

    if output_file is None or output_file.strip() == "":
        exit(f"[error]: please provide a valid local path to the output file")

    logger.info(f"Input File Path: {input_file}")
    logger.info(f"Output File Path: {output_file}")

    # setting this environment variable to claim the execution of
    # this script for leaderboard evaluation
    os.environ[LEADER_BOARD_ENV] = "True"
    sys.argv = [
        file_name,
        "--input-file-path",
        input_file,
        "--output-file-path",
        output_file
    ]
    main_prog()


if __name__ == "__main__":
    main()
