#!/usr/bin/env python
import argparse

from inference import detect_and_track_with_local_file, detect_and_track_with_s3_file
from utility import AmazonService

# define input argument for the application
parser = argparse.ArgumentParser(description="Object Detection and Tracking Script")

# Create a mutually exclusive group
mutually_exclusive_group = parser.add_mutually_exclusive_group(required=True)

# Define flags within the group
mutually_exclusive_group.add_argument("--local-path", help="Local Path to the video file")
mutually_exclusive_group.add_argument("--s3-prefix", help="Prefix of the video file stored in ieee s3 bucket")


def main():
    args = parser.parse_args()

    if args.local_path:
        video_path = args.local_path.strip()
        if video_path == "":
            exit(f"[error]: please provide a valid local path to the video")
        detect_and_track_with_local_file(video_path)
    else:
        s3_prefix = args.s3_prefix.strip()
        if s3_prefix == "":
            exit(f"[error]: please provide a s3 bucket prefix")

        # make sure an aws profile named hackathon is created
        aws = AmazonService("hackathon")
        video_object_keys = aws.list_objects_with_prefix(prefix=s3_prefix)
        for object_key in video_object_keys:
            signed_url = aws.generate_signed_url(object_key)
            detect_and_track_with_s3_file(object_key, signed_url)
            break

    print("object detection script completed successfully")


if __name__ == "__main__":
    main()
