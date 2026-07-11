"""Calibrated gaze-to-screen control entry point."""

import argparse

from eye_tracker import CalibratedEyeTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibrated gaze mouse control.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--calibration-points", type=int, choices=[5, 9, 13], default=5)
    parser.add_argument("--auto-click-duration", type=float, default=3.0)
    parser.add_argument("--click-cooldown", type=float, default=1.0)
    parser.add_argument("--min-click-confidence", type=float, default=0.6)
    parser.add_argument("--process-every-n-frames", type=int, default=1)
    parser.add_argument("--flush-interval", type=int, default=300)
    parser.add_argument("--open-output-dir", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tracker = CalibratedEyeTracker(
        video_source=0,
        output_dir=args.output_dir,
        calibration_points=args.calibration_points,
        auto_click_duration=args.auto_click_duration,
        click_cooldown=args.click_cooldown,
        min_click_confidence=args.min_click_confidence,
        process_every_n_frames=args.process_every_n_frames,
        flush_interval=args.flush_interval,
        open_output_dir=args.open_output_dir,
    )
    tracker.run()
