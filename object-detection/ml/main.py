#!/usr/bin/env python
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from .inference import detect_and_track_with_local_file, detect_and_track_with_s3_file
from .utility import AmazonService

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

    print("object detection script execution starting")

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

        # make sure an aws profile named hackathon is created
        aws = AmazonService("hackathon")
        video_object_keys = aws.list_objects_with_prefix(prefix=s3_input_path_prefix)
        signed_urls = [aws.generate_signed_url(object_key) for object_key in video_object_keys]

        futures_to_detection = list()
        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="ml-executor-") as executor:
            for object_key, signed_url in zip(video_object_keys, signed_urls):
                futures_to_detection.append(
                    executor.submit(
                        detect_and_track_with_s3_file, object_key, signed_url, output_file_name, push_to_gcs))
            for future in as_completed(futures_to_detection):
                result = future.result()
                print(f"future completed with result {result}")

    print("object detection script execution completed")


if __name__ == "__main__":
    main()
