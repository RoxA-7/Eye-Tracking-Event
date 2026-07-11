"""Controlled mouse and gaze target-selection experiment.

The experiment uses one deterministic trial generator for every input method.
Gaze trials run in full-screen coordinates so calibration targets, gaze mapping,
and experiment targets share the same coordinate system.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime

import cv2
import numpy as np

from experiment_logger import (
    TrialLogger,
    TrialRecord,
    elapsed_since,
    iso_now,
    save_session_config,
)


WINDOW_NAME = "Target Selection Experiment"


class DwellSelector:
    """Emit a selection after a gaze point remains within a stable region."""

    def __init__(self, duration_sec=1.0, stability_radius_px=45):
        if duration_sec <= 0:
            raise ValueError("duration_sec must be greater than zero")
        if stability_radius_px <= 0:
            raise ValueError("stability_radius_px must be greater than zero")
        self.duration_sec = duration_sec
        self.stability_radius_px = stability_radius_px
        self.reference = None
        self.started_at = None

    def reset(self):
        self.reference = None
        self.started_at = None

    def update(self, point, valid, now=None):
        now = time.monotonic() if now is None else now
        if not valid or point is None:
            self.reset()
            return None

        if self.reference is None:
            self.reference = point
            self.started_at = now
            return None

        if TargetSelectionExperiment.distance(point, *self.reference) > self.stability_radius_px:
            self.reference = point
            self.started_at = now
            return None

        if now - self.started_at >= self.duration_sec:
            selected = point
            self.reset()
            return selected
        return None

    def progress(self, now=None):
        if self.started_at is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return min(1.0, max(0.0, (now - self.started_at) / self.duration_sec))


class TargetSelectionExperiment:
    def __init__(
        self,
        participant_id,
        session_id,
        output_dir="results",
        input_method="mouse",
        width=1280,
        height=720,
        trials_per_radius=8,
        radii=(30, 50, 80),
        seed=42,
        timeout_sec=5.0,
        video_source=0,
        calibration_points=9,
        calibration_samples_per_point=12,
        gaze_smoothing=5,
        dwell_method="confidence_aware",
        dwell_time_sec=1.0,
        dwell_stability_radius_px=45,
        min_confidence=0.6,
    ):
        self.participant_id = participant_id
        self.session_id = session_id
        self.output_dir = output_dir
        self.input_method = input_method
        self.width = width
        self.height = height
        self.trials_per_radius = trials_per_radius
        self.radii = tuple(radii)
        self.seed = seed
        self.timeout_sec = timeout_sec
        self.video_source = video_source
        self.calibration_points = calibration_points
        self.calibration_samples_per_point = calibration_samples_per_point
        self.gaze_smoothing = gaze_smoothing
        self.dwell_method = dwell_method
        self.dwell_time_sec = dwell_time_sec
        self.dwell_stability_radius_px = dwell_stability_radius_px
        self.min_confidence = min_confidence
        self.click = None
        self.rng = random.Random(seed)

        if input_method not in {"mouse", "gaze"}:
            raise ValueError("input_method must be mouse or gaze")
        if dwell_method not in {"fixed", "confidence_aware"}:
            raise ValueError("dwell_method must be fixed or confidence_aware")
        if trials_per_radius < 1 or not self.radii or any(radius <= 0 for radius in self.radii):
            raise ValueError("trials_per_radius and all target radii must be positive")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        if calibration_samples_per_point < 3:
            raise ValueError("calibration_samples_per_point must be at least 3")
        if gaze_smoothing < 1:
            raise ValueError("gaze_smoothing must be at least 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

    def build_trials(self):
        trials = []
        trial_id = 1
        for radius in self.radii:
            margin = max(radius + 20, 80)
            if self.width <= 2 * margin or self.height <= 2 * margin:
                raise ValueError(f"Experiment surface is too small for radius {radius}")
            for _ in range(self.trials_per_radius):
                trials.append(
                    {
                        "trial_id": trial_id,
                        "target_id": f"r{radius}_t{trial_id}",
                        "target_x": self.rng.randint(margin, self.width - margin),
                        "target_y": self.rng.randint(margin, self.height - margin),
                        "target_radius": radius,
                    }
                )
                trial_id += 1
        self.rng.shuffle(trials)
        return trials

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def draw_frame(self, trial, elapsed, gaze_point=None, dwell_progress=0.0, valid_gaze=False):
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
        instruction = "Click the blue target." if self.input_method == "mouse" else "Look at the blue target."
        cv2.putText(
            frame,
            f"{instruction} Press Q to stop.",
            (24, self.height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 30, 30),
            2,
        )
        if gaze_point is not None:
            color = (40, 170, 40) if valid_gaze else (80, 80, 180)
            cv2.circle(frame, gaze_point, 10, color, 2)
            if dwell_progress > 0:
                cv2.ellipse(
                    frame,
                    gaze_point,
                    (18, 18),
                    -90,
                    0,
                    360 * dwell_progress,
                    color,
                    3,
                )
        return frame

    @staticmethod
    def distance(point, target_x, target_y):
        return math.hypot(point[0] - target_x, point[1] - target_y)

    def session_config(self):
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "input_method": self.input_method,
            "width": self.width,
            "height": self.height,
            "trials_per_radius": self.trials_per_radius,
            "radii": list(self.radii),
            "seed": self.seed,
            "timeout_sec": self.timeout_sec,
            "video_source": self.video_source,
            "calibration_points": self.calibration_points if self.input_method == "gaze" else None,
            "calibration_samples_per_point": (
                self.calibration_samples_per_point if self.input_method == "gaze" else None
            ),
            "gaze_smoothing": self.gaze_smoothing if self.input_method == "gaze" else None,
            "dwell_method": self.dwell_method if self.input_method == "gaze" else None,
            "dwell_time_sec": self.dwell_time_sec if self.input_method == "gaze" else None,
            "dwell_stability_radius_px": (
                self.dwell_stability_radius_px if self.input_method == "gaze" else None
            ),
            "min_confidence": self.min_confidence if self.input_method == "gaze" else None,
        }

    def _create_window(self, fullscreen=False):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW_NAME, self.width, self.height)

    def run_mouse(self):
        trials = self.build_trials()
        self._create_window()
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        with TrialLogger(self.output_dir) as logger:
            for trial in trials:
                self.click = None
                false_click_count = 0
                start = time.monotonic()
                start_iso = iso_now()
                success = False
                missed = False
                distance_px = None

                while True:
                    elapsed = time.monotonic() - start
                    cv2.imshow(WINDOW_NAME, self.draw_frame(trial, elapsed))
                    if cv2.waitKey(16) & 0xFF == ord("q"):
                        return

                    if self.click is not None:
                        distance_px = self.distance(self.click, trial["target_x"], trial["target_y"])
                        if distance_px <= trial["target_radius"]:
                            success = True
                            break
                        false_click_count += 1
                        self.click = None

                    if elapsed >= self.timeout_sec:
                        missed = True
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
                        selection_time_sec=elapsed_since(start),
                        success=success,
                        false_click_count=false_click_count,
                        missed_selection=missed,
                        click_x=self.click[0] if self.click else None,
                        click_y=self.click[1] if self.click else None,
                        distance_to_target_px=round(distance_px, 3) if distance_px is not None else None,
                    )
                )

    def run_gaze(self):
        from eye_tracker import CalibratedEyeTracker

        tracker = CalibratedEyeTracker(
            video_source=self.video_source,
            output_dir=self.output_dir,
            enable_heatmap=False,
            enable_fixation=False,
            enable_blink_detection=False,
            flush_interval=0,
            gaze_smoothing=self.gaze_smoothing,
            calibration_points=self.calibration_points,
            calibration_samples_per_point=self.calibration_samples_per_point,
            min_click_confidence=self.min_confidence,
        )
        if not tracker.run_calibration():
            return

        self.width, self.height = tracker.screen_w, tracker.screen_h
        save_session_config(self.output_dir, self.session_id, self.session_config())
        trials = self.build_trials()
        if not tracker._open_capture():
            return
        tracker.start_time = time.time()

        self._create_window(fullscreen=True)
        try:
            with TrialLogger(self.output_dir) as logger:
                for trial in trials:
                    if not self._run_gaze_trial(tracker, logger, trial):
                        return
        finally:
            tracker.cleanup()

    def _run_gaze_trial(self, tracker, logger, trial):
        selector = DwellSelector(self.dwell_time_sec, self.dwell_stability_radius_px)
        start = time.monotonic()
        start_iso = iso_now()
        first_valid_gaze_time = None
        confidences = []
        fps_values = []
        total_frames = 0
        lost_frames = 0
        false_click_count = 0
        success = False
        selected_point = None
        last_gaze = None
        last_record = None
        distance_px = None

        while True:
            ret, camera_frame = tracker.cap.read()
            if not ret:
                if not isinstance(self.video_source, int):
                    print("Video source ended during the experiment.")
                    return False
                continue

            record = tracker.process_frame(camera_frame)
            last_record = record
            total_frames += 1
            confidence = float(record["gaze_confidence"] or 0.0)
            confidences.append(confidence)
            if record["fps"] is not None:
                fps_values.append(float(record["fps"]))

            screen_x, screen_y = tracker.map_to_screen(record["gaze_x"], record["gaze_y"])
            last_gaze = (screen_x, screen_y) if screen_x is not None else None
            tracking_available = last_gaze is not None
            valid_gaze = tracking_available
            if self.dwell_method == "confidence_aware":
                valid_gaze = valid_gaze and confidence >= self.min_confidence

            if valid_gaze and first_valid_gaze_time is None:
                first_valid_gaze_time = iso_now()
            if not tracking_available:
                lost_frames += 1

            selected_point = selector.update(last_gaze, valid_gaze)
            if selected_point is not None:
                distance_px = self.distance(selected_point, trial["target_x"], trial["target_y"])
                if distance_px <= trial["target_radius"]:
                    success = True
                    break
                false_click_count += 1

            elapsed = time.monotonic() - start
            display = self.draw_frame(
                trial,
                elapsed,
                gaze_point=last_gaze,
                dwell_progress=selector.progress(),
                valid_gaze=valid_gaze,
            )
            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
            if elapsed >= self.timeout_sec:
                break

        logger.write(
            TrialRecord(
                participant_id=self.participant_id,
                session_id=self.session_id,
                trial_id=trial["trial_id"],
                condition=f"gaze_{self.calibration_points}pt_{self.dwell_method}",
                input_method="gaze",
                calibration_mode=f"{self.calibration_points}-point",
                gaze_smoothing_window=self.gaze_smoothing,
                dwell_method=self.dwell_method,
                dwell_time_sec=self.dwell_time_sec,
                target_id=trial["target_id"],
                target_x=trial["target_x"],
                target_y=trial["target_y"],
                target_radius=trial["target_radius"],
                trial_start_time=start_iso,
                first_valid_gaze_time=first_valid_gaze_time,
                selection_time_sec=elapsed_since(start),
                success=success,
                false_click_count=false_click_count,
                missed_selection=not success,
                gaze_x=last_record["gaze_x"] if last_record else None,
                gaze_y=last_record["gaze_y"] if last_record else None,
                screen_x=selected_point[0] if selected_point else (last_gaze[0] if last_gaze else None),
                screen_y=selected_point[1] if selected_point else (last_gaze[1] if last_gaze else None),
                click_x=selected_point[0] if selected_point else None,
                click_y=selected_point[1] if selected_point else None,
                distance_to_target_px=round(distance_px, 3) if distance_px is not None else None,
                face_detected=last_record["face_detected"] if last_record else False,
                pupil_detected_left=last_record["pupil_detected_left"] if last_record else False,
                pupil_detected_right=last_record["pupil_detected_right"] if last_record else False,
                tracking_lost_ratio=round(lost_frames / total_frames, 3) if total_frames else 1.0,
                mean_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
                fps_mean=round(sum(fps_values) / len(fps_values), 2) if fps_values else None,
            )
        )
        return True

    def run(self):
        try:
            if self.input_method == "gaze":
                self.run_gaze()
            else:
                save_session_config(self.output_dir, self.session_id, self.session_config())
                self.run_mouse()
        finally:
            cv2.destroyAllWindows()


def parse_video_source(value):
    return int(value) if value.isdigit() else value


def parse_args():
    parser = argparse.ArgumentParser(description="Run a controlled target-selection experiment.")
    parser.add_argument("--participant-id", default="pilot")
    parser.add_argument("--session-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--input-method", choices=["mouse", "gaze"], default="mouse")
    parser.add_argument("--width", type=int, default=1280, help="Mouse-mode surface width")
    parser.add_argument("--height", type=int, default=720, help="Mouse-mode surface height")
    parser.add_argument("--trials-per-radius", type=int, default=8)
    parser.add_argument("--radii", type=int, nargs="+", default=[30, 50, 80])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--video-source", type=parse_video_source, default=0)
    parser.add_argument("--calibration-points", type=int, choices=[5, 9, 13], default=9)
    parser.add_argument("--calibration-samples-per-point", type=int, default=12)
    parser.add_argument("--gaze-smoothing", type=int, default=5)
    parser.add_argument("--dwell-method", choices=["fixed", "confidence_aware"], default="confidence_aware")
    parser.add_argument("--dwell-time-sec", type=float, default=1.0)
    parser.add_argument("--dwell-stability-radius-px", type=int, default=45)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TargetSelectionExperiment(
        participant_id=args.participant_id,
        session_id=args.session_id,
        output_dir=args.output_dir,
        input_method=args.input_method,
        width=args.width,
        height=args.height,
        trials_per_radius=args.trials_per_radius,
        radii=args.radii,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
        video_source=args.video_source,
        calibration_points=args.calibration_points,
        calibration_samples_per_point=args.calibration_samples_per_point,
        gaze_smoothing=args.gaze_smoothing,
        dwell_method=args.dwell_method,
        dwell_time_sec=args.dwell_time_sec,
        dwell_stability_radius_px=args.dwell_stability_radius_px,
        min_confidence=args.min_confidence,
    ).run()
