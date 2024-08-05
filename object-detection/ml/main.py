#!/usr/bin/env python

import argparse
import logging
import sys
from datetime import datetime as dt

from .inference import detect_and_track_with_local_file, detect_and_track_with_s3_file
from .utility import AmazonService, environmental_variable_is_present
from .constants import VERBOSE_LOGGING, LOG_FILE_NAME

# setup the logger configurations
log_level = logging.INFO
file_handler = logging.FileHandler(filename=LOG_FILE_NAME, mode="w")
stream_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [stream_handler]
if environmental_variable_is_present(VERBOSE_LOGGING):
    log_level = logging.DEBUG
    handlers.append(file_handler)

logging.basicConfig(
    level=log_level,
    handlers=[stream_handler, file_handler],
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)
logger = logging.getLogger(__name__)

# define input argument for the application
parser = argparse.ArgumentParser(description="Object Detection and Tracking Script")

# Create a mutually exclusive group
mutually_exclusive_group = parser.add_mutually_exclusive_group(required=True)

# Define flags within the group
mutually_exclusive_group.add_argument("--input-path", help="Path to the video file stored in local")
mutually_exclusive_group.add_argument("--s3-input-path-prefix",
                                      help="Prefix of the video file stored in ieee s3 bucket")

parser.add_argument("--output-file-name",
                    default=None,
                    help="csv file name for output of object detection")
parser.add_argument("--push-to-gcs",
                    action="store_true",
                    default=False,
                    help="push the output files to pre-defined gcs bucket")


def main():
    args = parser.parse_args()

    starting_time = dt.now()
    logger.info("object detection script execution starting")

    output_file_name = None
    if args.output_file_name is not None:
        output_file_name = args.output_file_name.strip()
        if output_file_name == "":
            exit(f"[error]: please provide a csv file name to store object detection results")

    # flag to control whether the output file should be pushed
    # to pre-defined gcs bucket
    push_to_gcs = args.push_to_gcs

    if args.input_path is not None:
        input_path = args.input_path.strip()
        if input_path == "":
            exit(f"[error]: please provide a valid local path to the input video")
        detect_and_track_with_local_file(input_path, output_file_name, push_to_gcs)
    else:
        s3_input_path_prefix = args.s3_input_path_prefix.strip()
        if s3_input_path_prefix == "":
            exit(f"[error]: please provide a s3 bucket prefix")

        # make sure to set required environmental variables
        # 1. configure aws profile and set an env var AWS_PROFILE=${profileName}
        # 2. set following env variable - AWS_ACCESS_KEY_ID,
        #    AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION
        aws = AmazonService()
        video_object_keys = aws.list_objects_with_prefix(prefix=s3_input_path_prefix)
        for object_key in video_object_keys:
            signed_url = aws.generate_signed_url(object_key)
            detect_and_track_with_s3_file(object_key, signed_url, output_file_name, push_to_gcs)

    ending_time = dt.now()
    execution_time = ending_time - starting_time
    logger.info(f"object detection script execution completed after {execution_time.seconds} seconds")


if __name__ == "__main__":
    main()
