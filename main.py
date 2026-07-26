import argparse

from monocular_vision import fsd


def parse_args():
    parser = argparse.ArgumentParser(description="Run the vision FSD video pipeline.")
    parser.add_argument(
        "video_path",
        nargs="?",
        default="data/realtime.MOV",
        help="Path to the input video.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stream",
        action="store_true",
        help="Process and display the video on the fly.",
    )
    mode.add_argument(
        "--save",
        action="store_true",
        help="Process offline and save the annotated video.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Maximum number of processed frames to run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for --save mode. Defaults to outputs/<video>_processed_<timestamp>.mp4.",
    )
    parser.add_argument(
        "--target-motion-fps",
        type=float,
        default=20,
        help="Target sampled FPS for high-FPS videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = "save" if args.save else "stream"
    fsd.driver(
        args.video_path,
        mode=mode,
        output_path=args.output,
        max_frames=args.frames,
        target_motion_fps=args.target_motion_fps,
    )


if __name__ == "__main__":
    main()
