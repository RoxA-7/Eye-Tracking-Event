"""
Eye Tracking (Calibrated) — calibration + mouse control entry point.

1. 5-point calibration: look at each yellow dot on screen.
2. Mouse follows your gaze in real time.
3. Hold gaze for 3 seconds to auto-click.
Press Q to quit.
"""

from eye_tracker import CalibratedEyeTracker

if __name__ == "__main__":
    tracker = CalibratedEyeTracker(video_source=0, auto_click_duration=3.0)
    tracker.run()
