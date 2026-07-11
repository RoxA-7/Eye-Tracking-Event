"""Main entry point for webcam/video gaze tracking."""

import argparse
import tkinter as tk
from tkinter import filedialog

from eye_tracker import EyeTracker


def select_video_source():
    """Open a file dialog for video selection; return webcam if cancelled."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a video file (cancel to use webcam)",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")],
    )
    return path if path else 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run webcam/video eye tracking.")
    parser.add_argument("--source", default=None, help="Video path or webcam index. Omit for file dialog.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--process-every-n-frames", type=int, default=1)
    parser.add_argument("--flush-interval", type=int, default=300)
    parser.add_argument("--show-debug-windows", action="store_true")
    parser.add_argument("--open-output-dir", action="store_true")
    return parser.parse_args()


def parse_source(source):
    if source is None:
        return select_video_source()
    if str(source).isdigit():
        return int(source)
    return source


if __name__ == "__main__":
    args = parse_args()
    source = parse_source(args.source)
    print(f"Using video source: {source}")

    tracker = EyeTracker(
        video_source=source,
        output_dir=args.output_dir,
        process_every_n_frames=args.process_every_n_frames,
        flush_interval=args.flush_interval,
        show_debug_windows=args.show_debug_windows,
        open_output_dir=args.open_output_dir,
    )
    tracker.run()
