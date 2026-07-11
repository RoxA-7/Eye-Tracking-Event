"""Controlled target-selection experiment for mouse baseline collection.

This is intentionally simple and deterministic. It provides a real experiment
surface and trial-level CSV logs before gaze conditions are integrated.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime

import cv2
import numpy as np

from experiment_logger import TrialLogger, TrialRecord, elapsed_since, iso_now


WINDOW_NAME = "Target Selection Experiment"


class TargetSelectionExperiment:
    def __init__(
        self,
        participant_id,
        session_id,
        output_dir="results",
        width=1280,
        height=720,
        trials_per_radius=8,
        radii=(30, 50, 80),
        seed=42,
        timeout_sec=5.0,
    ):
        self.participant_id = participant_id
        self.session_id = session_id
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.trials_per_radius = trials_per_radius
        self.radii = tuple(radii)
        self.seed = seed
        self.timeout_sec = timeout_sec
        self.click = None
        self.false_click_count = 0
        self.rng = random.Random(seed)

    def build_trials(self):
        trials = []
        trial_id = 1
        for radius in self.radii:
            margin = max(radius + 20, 80)
            for _ in range(self.trials_per_radius):
                x = self.rng.randint(margin, self.width - margin)
                y = self.rng.randint(margin, self.height - margin)
                trials.append(
                    {
                        "trial_id": trial_id,
                        "target_id": f"r{radius}_t{trial_id}",
                        "target_x": x,
                        "target_y": y,
                        "target_radius": radius,
                    }
                )
                trial_id += 1
        self.rng.shuffle(trials)
        return trials

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def draw_frame(self, trial, elapsed):
        frame = np.full((self.height, self.width, 3), 245, dtype=np.uint8)
        center = (trial["target_x"], trial["target_y"])
        radius = trial["target_radius"]
        cv2.circle(frame, center, radius, (40, 130, 220), -1)
        cv2.circle(frame, center, radius, (20, 80, 160), 2)
        cv2.putText(
            frame,
            f"Trial {trial['trial_id']}  radius={radius}px  time={elapsed:.1f}s",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
        )
        cv2.putText(
            frame,
            "Click the blue target. Press Q to stop.",
            (24, self.height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 30, 30),
            2,
        )
        return frame

    @staticmethod
    def distance(click, target_x, target_y):
        return math.sqrt((click[0] - target_x) ** 2 + (click[1] - target_y) ** 2)

    def run(self):
        trials = self.build_trials()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.width, self.height)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        with TrialLogger(self.output_dir) as logger:
            for trial in trials:
                self.click = None
                self.false_click_count = 0
                start = time.time()
                start_iso = iso_now()
                success = False
                missed = False
                distance_px = None
                selection_time = None

                while True:
                    elapsed = time.time() - start
                    frame = self.draw_frame(trial, elapsed)
                    cv2.imshow(WINDOW_NAME, frame)

                    key = cv2.waitKey(16) & 0xFF
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        return

                    if self.click is not None:
                        distance_px = self.distance(self.click, trial["target_x"], trial["target_y"])
                        if distance_px <= trial["target_radius"]:
                            success = True
                            selection_time = elapsed_since(start)
                            break
                        self.false_click_count += 1
                        self.click = None

                    if elapsed >= self.timeout_sec:
                        missed = True
                        selection_time = elapsed_since(start)
                        break

                logger.write(
                    TrialRecord(
                        participant_id=self.participant_id,
                        session_id=self.session_id,
                        trial_id=trial["trial_id"],
                        condition="mouse_baseline",
                        input_method="mouse",
                        calibration_mode="not_applicable",
                        gaze_smoothing_window=None,
                        dwell_method="not_applicable",
                        dwell_time_sec=None,
                        target_id=trial["target_id"],
                        target_x=trial["target_x"],
                        target_y=trial["target_y"],
                        target_radius=trial["target_radius"],
                        trial_start_time=start_iso,
                        selection_time_sec=selection_time,
                        success=success,
                        false_click_count=self.false_click_count,
                        missed_selection=missed,
                        click_x=self.click[0] if self.click else None,
                        click_y=self.click[1] if self.click else None,
                        distance_to_target_px=round(distance_px, 3) if distance_px is not None else None,
                    )
                )

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a mouse target-selection baseline.")
    parser.add_argument("--participant-id", default="pilot")
    parser.add_argument("--session-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--trials-per-radius", type=int, default=8)
    parser.add_argument("--radii", type=int, nargs="+", default=[30, 50, 80])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TargetSelectionExperiment(
        participant_id=args.participant_id,
        session_id=args.session_id,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        trials_per_radius=args.trials_per_radius,
        radii=args.radii,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
    ).run()
