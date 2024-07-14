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
import sys
from ml.main import main as main_prog


def main():
    try:
        file_name, video_name, output_file_name = sys.argv[0], sys.argv[1], sys.argv[2]
    except IndexError:
        exit(
            f"[warn]: required arguments missing, please provide all the required arguments "
            f"and execute the script in following way\n\n"
            f"python app.py {{input_video}} {{output_file_name}}\n\n"
            f"input_video      -> Stn_HD_1.mp4\n"
            f"output_file_name -> Output_Turning_Patterns.csv"
        )
    if video_name is None or video_name.strip() == "":
        exit(f"[error]: please provide a valid local path to the input video")

    if output_file_name is None or output_file_name.strip() == "":
        exit(f"[error]: please provide a csv file name to store object detection results")

    print(f"Running Object Detection on video present at \"{video_name}\"")
    print(f"Detection results will be stored at \"{output_file_name}\"")
    sys.argv = [
        file_name,
        "--input-path",
        video_name,
        "--output-file-name",
        output_file_name
    ]
    main_prog()


if __name__ == "__main__":
    main()
