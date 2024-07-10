#!/usr/bin/env python

from absl import app
from absl import flags

from inference import detect_and_track

FLAGS = flags.FLAGS

# define input argument for the application
flags.DEFINE_string("video", None, "Path to video for object detection", required=True)


def main(argv):

    video_path = FLAGS.video.strip()
    if video_path == "":
        print(f"please provide a valid video path")

    print(f"initiating object detection and tracking for {video_path}")

    detect_and_track(video_path)

    print("object detection completed successfully")


if __name__ == "__main__":
    app.run(main)
