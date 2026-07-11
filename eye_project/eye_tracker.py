"""
Shared eye-tracking engine for pupil detection, gaze analysis, calibration,
and dwell-based mouse control.

The module is intentionally lightweight: the core tracker can run from a
webcam or video file, saves traceable CSV/PNG outputs, and exposes enough
quality fields to support later experiment logging and paper analysis.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
import math
import os
import platform
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd


class EyeTracker:
    """Core eye-tracking engine.

    Parameters
    ----------
    video_source:
        ``0`` for webcam, or a video file path.
    enable_heatmap:
        Overlay and save a gaze heatmap.
    enable_fixation:
        Detect and record fixations.
    enable_blink_detection:
        Count blink-like pupil-loss events.
    gaze_smoothing:
        Number of recent gaze points used for smoothing.
    process_every_n_frames:
        Run MediaPipe every N frames and reuse recent landmarks between runs.
    flush_interval:
        Save partial CSV files every N processed frames. ``0`` disables this.
    show_debug_windows:
        Show per-eye threshold windows.
    output_dir:
        Directory for CSV/PNG outputs.
    open_output_dir:
        Open the output folder after cleanup. Disabled by default for
        reproducible batch runs.
    max_stale_landmark_frames:
        Number of consecutive MediaPipe misses before cached landmarks are
        discarded.
    """

    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]

    def __init__(
        self,
        video_source=0,
        enable_heatmap=True,
        enable_fixation=True,
        enable_blink_detection=True,
        gaze_smoothing=5,
        process_every_n_frames=1,
        flush_interval=300,
        show_debug_windows=False,
        output_dir="results",
        open_output_dir=False,
        max_stale_landmark_frames=5,
    ):
        self.video_source = video_source
        self.enable_heatmap = enable_heatmap
        self.enable_fixation = enable_fixation
        self.enable_blink_detection = enable_blink_detection
        self.process_every_n_frames = max(1, process_every_n_frames)
        self.flush_interval = flush_interval
        self.show_debug_windows = show_debug_windows
        self.output_dir = output_dir
        self.open_output_dir = open_output_dir
        self.max_stale_landmark_frames = max(0, max_stale_landmark_frames)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )

        self.gaze_buffer = deque(maxlen=max(1, gaze_smoothing))
        self.heatmap_layer = None

        self.fixations = []
        self.fixation_threshold = 15
        self.min_fixation_duration = 0.5
        self.fixation_start_time = None
        self.fixation_reference = None
        self._jitter_counter = 0
        self._jitter_tolerance = 3

        self.blink_count = 0
        self.fatigue_alert_triggered = False
        self._blink_loss_frames = 0
        self._blink_loss_threshold = 3

        self.data_records = []
        self.frame_count = 0
        self.start_time = time.time()
        self._last_landmarks = None
        self._stale_landmark_frames = 0
        self.cap = None

    # ------------------------------------------------------------------
    # Geometry and detection helpers
    # ------------------------------------------------------------------

    def extract_eye_roi(self, image, landmarks, eye_indices, margin=10):
        """Crop the eye region from the frame using normalized landmarks."""
        h, w = image.shape[:2]
        x1 = int(landmarks[eye_indices[0]].x * w)
        y1 = int(landmarks[eye_indices[0]].y * h)
        x2 = int(landmarks[eye_indices[1]].x * w)
        y2 = int(landmarks[eye_indices[1]].y * h)

        x_min = max(min(x1, x2) - margin, 0)
        y_min = max(min(y1, y2) - margin, 0)
        x_max = min(max(x1, x2) + margin, w)
        y_max = min(max(y1, y2) + margin, h)
        return image[y_min:y_max, x_min:x_max], (x_min, y_min)

    def detect_pupil(self, roi, draw_on=None, offset=(0, 0), label=None):
        """Detect pupil center in an eye ROI via adaptive thresholding.

        The returned coordinate is always in the same coordinate system as the
        provided ``offset``. Pass ``offset=(0, 0)`` for ROI-local coordinates or
        the ROI top-left offset for full-frame coordinates.
        """
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.equalizeHist(gray)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        if label and self.show_debug_windows:
            cv2.imshow(label, thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        ox, oy = offset
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            if draw_on is not None:
                cv2.rectangle(draw_on, (ox + x, oy + y), (ox + x + w, oy + y + h), (255, 0, 0), 2)
                cv2.line(draw_on, (ox + cx, oy + y), (ox + cx, oy + y + h), (0, 255, 0), 1)
                cv2.line(draw_on, (ox + x, oy + cy), (ox + x + w, oy + cy), (0, 255, 0), 1)

            return ox + cx, oy + cy
        return None

    def compute_gaze(self, left_center, right_center):
        """Average left and right pupil centers, then smooth over time."""
        if left_center is None or right_center is None:
            return None, None

        raw_x = (left_center[0] + right_center[0]) // 2
        raw_y = (left_center[1] + right_center[1]) // 2
        self.gaze_buffer.append((raw_x, raw_y))
        gaze_x = int(sum(point[0] for point in self.gaze_buffer) / len(self.gaze_buffer))
        gaze_y = int(sum(point[1] for point in self.gaze_buffer) / len(self.gaze_buffer))
        return gaze_x, gaze_y

    def estimate_gaze_confidence(self, face_detected, left_center, right_center):
        """Estimate a simple, interpretable tracking-confidence score."""
        if not face_detected:
            return 0.0
        detected_eyes = int(left_center is not None) + int(right_center is not None)
        if detected_eyes == 0:
            return 0.0

        pupil_score = detected_eyes / 2.0
        stability_score = 1.0
        if len(self.gaze_buffer) >= 3:
            xs = np.array([point[0] for point in self.gaze_buffer], dtype=float)
            ys = np.array([point[1] for point in self.gaze_buffer], dtype=float)
            jitter = math.sqrt(float(np.var(xs) + np.var(ys)))
            stability_score = max(0.0, min(1.0, 1.0 - jitter / 50.0))
        return round(pupil_score * stability_score, 3)

    def update_heatmap(self, gaze_point, frame):
        """Add gaze point to a persistent heatmap layer and blend onto frame."""
        if not self.enable_heatmap:
            return frame

        gx, gy = gaze_point
        if self.heatmap_layer is None:
            self.heatmap_layer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        cv2.circle(self.heatmap_layer, (gx, gy), 3, 1, -1)
        self.heatmap_layer *= 0.995
        heatmap_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(self.heatmap_layer, alpha=10),
            cv2.COLORMAP_JET,
        )
        return cv2.addWeighted(frame, 0.7, heatmap_vis, 0.3, 0)

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def detect_fixation(self, gaze_point):
        """Detect I-VT style fixation from smoothed gaze points."""
        if not self.enable_fixation or gaze_point is None:
            return None

        gx, gy = gaze_point
        if self.fixation_reference is None:
            self.fixation_reference = (gx, gy)
            self.fixation_start_time = time.time()
            return None

        dx = abs(gx - self.fixation_reference[0])
        dy = abs(gy - self.fixation_reference[1])
        if dx <= self.fixation_threshold and dy <= self.fixation_threshold:
            self._jitter_counter = 0
            duration = time.time() - self.fixation_start_time
            if duration >= self.min_fixation_duration:
                fix = {
                    "start_time": self.fixation_start_time,
                    "duration_sec": round(duration, 3),
                    "gaze_x": gx,
                    "gaze_y": gy,
                }
                self.fixation_start_time = time.time()
                self.fixation_reference = (gx, gy)
                return fix
        else:
            self._jitter_counter += 1
            if self._jitter_counter > self._jitter_tolerance:
                self.fixation_start_time = time.time()
                self.fixation_reference = (gx, gy)
                self._jitter_counter = 0
        return None

    def detect_blink(self, left_ok, right_ok):
        """Return 1 only after sustained pupil loss, reducing false blinks."""
        if not self.enable_blink_detection:
            return 0
        if not left_ok or not right_ok:
            self._blink_loss_frames += 1
            return 0
        if self._blink_loss_frames >= self._blink_loss_threshold:
            self._blink_loss_frames = 0
            self.blink_count += 1
            return 1
        self._blink_loss_frames = 0
        return 0

    def check_fatigue(self):
        """Print a fatigue warning when blink rate is high after warmup."""
        elapsed = time.time() - self.start_time
        if elapsed > 10:
            blink_rate = self.blink_count / elapsed
            if blink_rate > 0.3 and not self.fatigue_alert_triggered:
                print("Fatigue warning: blink rate is high; consider resting.")
                self.fatigue_alert_triggered = True

    # ------------------------------------------------------------------
    # Frame processing and persistence
    # ------------------------------------------------------------------

    def _update_landmarks(self, rgb_frame):
        if self.frame_count % self.process_every_n_frames != 0:
            return self._last_landmarks, self._last_landmarks is not None

        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            self._last_landmarks = results.multi_face_landmarks[0].landmark
            self._stale_landmark_frames = 0
            return self._last_landmarks, True

        self._stale_landmark_frames += 1
        if self._stale_landmark_frames > self.max_stale_landmark_frames:
            self._last_landmarks = None
        return self._last_landmarks, False

    def process_frame(self, frame):
        """Run the full pipeline on one BGR frame and append one CSV record."""
        if self.heatmap_layer is None and self.enable_heatmap:
            self.heatmap_layer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, fresh_face_detected = self._update_landmarks(rgb)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        elapsed_sec = time.time() - self.start_time
        left_center = None
        right_center = None
        gaze_x = None
        gaze_y = None
        fixation = None
        face_detected = landmarks is not None

        if landmarks is not None:
            roi_l, offset_l = self.extract_eye_roi(frame, landmarks, self.LEFT_EYE)
            left_center = self.detect_pupil(roi_l, draw_on=frame, offset=offset_l, label="Left Eye")

            roi_r, offset_r = self.extract_eye_roi(frame, landmarks, self.RIGHT_EYE)
            right_center = self.detect_pupil(roi_r, draw_on=frame, offset=offset_r, label="Right Eye")

            gaze_x, gaze_y = self.compute_gaze(left_center, right_center)
            if gaze_x is not None:
                cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
                cv2.line(frame, (gaze_x, 0), (gaze_x, frame.shape[0]), (0, 255, 0), 1)
                cv2.line(frame, (0, gaze_y), (frame.shape[1], gaze_y), (0, 255, 0), 1)
                frame = self.update_heatmap((gaze_x, gaze_y), frame)

                fixation = self.detect_fixation((gaze_x, gaze_y))
                if fixation:
                    self.fixations.append(fixation)

        blink = self.detect_blink(left_center is not None, right_center is not None)
        confidence = self.estimate_gaze_confidence(face_detected, left_center, right_center)
        fps = round(self.frame_count / elapsed_sec, 2) if elapsed_sec > 0 else None

        record = {
            "timestamp": timestamp,
            "elapsed_sec": round(elapsed_sec, 3),
            "frame": self.frame_count,
            "face_detected": bool(face_detected),
            "fresh_face_detected": bool(fresh_face_detected),
            "pupil_detected_left": left_center is not None,
            "pupil_detected_right": right_center is not None,
            "left_x": left_center[0] if left_center else None,
            "left_y": left_center[1] if left_center else None,
            "right_x": right_center[0] if right_center else None,
            "right_y": right_center[1] if right_center else None,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "gaze_confidence": confidence,
            "blink": blink,
            "fps": fps,
        }
        self.data_records.append(record)

        self.check_fatigue()
        self.frame_count += 1
        return record

    def _open_capture(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"Unable to open video source: {self.video_source}")
            return False
        return True

    def _flush_data(self, output_dir=None):
        """Incrementally save data to disk."""
        output_dir = output_dir or self.output_dir
        if not self.data_records:
            return
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(self.data_records).to_csv(
            os.path.join(output_dir, "eye_tracking_data_partial.csv"),
            index=False,
        )
        if self.fixations:
            pd.DataFrame(self.fixations).to_csv(
                os.path.join(output_dir, "fixations_partial.csv"),
                index=False,
            )

    def run(self):
        """Start the main tracking loop. Press Q to quit."""
        if not self._open_capture():
            return

        print("Eye tracking started. Press Q to quit.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if not isinstance(self.video_source, int):
                        print("Video source ended.")
                        break
                    continue

                self.process_frame(frame)
                src_label = "webcam" if self.video_source == 0 else str(self.video_source)
                cv2.imshow(f"Eye Tracking - {src_label}", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if self.flush_interval > 0 and self.frame_count % self.flush_interval == 0:
                    self._flush_data()

                if self.frame_count % 30 == 0:
                    elapsed = max(time.time() - self.start_time, 1e-9)
                    print(f"  FPS: {self.frame_count / elapsed:.1f}")

        except KeyboardInterrupt:
            print("\nInterrupted; saving data.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release camera, destroy windows, and save data."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.save_results()

    def save_results(self, output_dir=None):
        """Export raw data CSV, fixations CSV, and gaze heatmap PNG."""
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        data_path = os.path.join(output_dir, f"eye_tracking_data_{ts}.csv")
        pd.DataFrame(self.data_records).to_csv(data_path, index=False)

        fix_path = os.path.join(output_dir, f"fixations_{ts}.csv")
        pd.DataFrame(self.fixations).to_csv(fix_path, index=False)

        if self.heatmap_layer is not None and np.max(self.heatmap_layer) > 0:
            heat_path = os.path.join(output_dir, f"gaze_heatmap_{ts}.png")
            plt.figure(figsize=(10, 6))
            heat_display = cv2.convertScaleAbs(
                self.heatmap_layer,
                alpha=255.0 / np.max(self.heatmap_layer),
            )
            plt.imshow(heat_display, cmap="jet")
            plt.title("Gaze Heatmap")
            plt.axis("off")
            plt.savefig(heat_path)
            plt.close()

        print(f"Data saved to {os.path.abspath(output_dir)}")
        if self.open_output_dir:
            self._open_folder(output_dir)

    def _open_folder(self, output_dir):
        abs_path = os.path.abspath(output_dir)
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", abs_path])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", abs_path])
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", abs_path])


class CalibratedEyeTracker(EyeTracker):
    """EyeTracker with calibration, screen mapping, and dwell mouse control."""

    def __init__(
        self,
        auto_click_duration=3.0,
        calibration_points=5,
        calibration_samples_per_point=12,
        click_cooldown=1.0,
        min_click_confidence=0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gaze_model = None
        self.auto_click_duration = auto_click_duration
        self.calibration_points = calibration_points
        self.calibration_samples_per_point = max(3, calibration_samples_per_point)
        self.click_cooldown = click_cooldown
        self.min_click_confidence = min_click_confidence
        self.screen_w = None
        self.screen_h = None
        self._click_start_time = None
        self._click_reference = None
        self._last_click_time = 0.0
        self.calibration_report = []

    @staticmethod
    def get_calibration_points(count):
        """Return normalized calibration points for 5, 9, or 13 point modes."""
        if count == 5:
            return [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
        if count == 9:
            values = [0.1, 0.5, 0.9]
            return [(x, y) for y in values for x in values]
        if count == 13:
            return [
                (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
                (0.25, 0.3), (0.75, 0.3),
                (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
                (0.25, 0.7), (0.75, 0.7),
                (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
            ]
        raise ValueError("calibration_points must be one of: 5, 9, 13")

    def run_calibration(self):
        """Collect multi-frame samples on a full-screen calibration canvas."""
        import pyautogui
        from sklearn.linear_model import LinearRegression

        calib_points = self.get_calibration_points(self.calibration_points)
        print(f"Calibration started: {len(calib_points)} points.")
        calib_data = []
        started = time.time()
        self.screen_w, self.screen_h = pyautogui.size()

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Unable to open video source for calibration: {self.video_source}")
            return False

        window_name = "Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        idx = 0
        point_started = time.monotonic()
        point_samples = []
        while idx < len(calib_points):
            ret, frame = cap.read()
            if not ret:
                if not isinstance(self.video_source, int):
                    print("Video source ended during calibration.")
                    break
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            norm_pt = calib_points[idx]
            px = int(norm_pt[0] * (self.screen_w - 1))
            py = int(norm_pt[1] * (self.screen_h - 1))
            canvas = np.full((self.screen_h, self.screen_w, 3), 245, dtype=np.uint8)
            cv2.circle(canvas, (px, py), 18, (0, 170, 230), -1)
            cv2.circle(canvas, (px, py), 4, (20, 20, 20), -1)
            cv2.putText(
                canvas,
                f"Point {idx + 1}/{len(calib_points)} - keep looking at the center",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 30, 30),
                2,
            )

            settling = time.monotonic() - point_started < 0.6
            if res.multi_face_landmarks and not settling:
                lm = res.multi_face_landmarks[0].landmark
                roi_l, off_l = self.extract_eye_roi(frame, lm, self.LEFT_EYE)
                roi_r, off_r = self.extract_eye_roi(frame, lm, self.RIGHT_EYE)
                pl = self.detect_pupil(roi_l, offset=off_l)
                pr = self.detect_pupil(roi_r, offset=off_r)

                if pl and pr:
                    gx = (pl[0] + pr[0]) // 2
                    gy = (pl[1] + pr[1]) // 2
                    point_samples.append((gx, gy))

            sample_count = len(point_samples)
            cv2.putText(
                canvas,
                f"Samples: {sample_count}/{self.calibration_samples_per_point}",
                (24, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (30, 30, 30),
                2,
            )
            if sample_count >= self.calibration_samples_per_point:
                samples = np.asarray(point_samples, dtype=float)
                gx, gy = np.median(samples, axis=0)
                calib_data.append(
                    [
                        idx + 1,
                        float(gx),
                        float(gy),
                        norm_pt[0],
                        norm_pt[1],
                        sample_count,
                        round(float(np.std(samples[:, 0])), 3),
                        round(float(np.std(samples[:, 1])), 3),
                    ]
                )
                print(f"  collected point {idx + 1}/{len(calib_points)} -> gaze=({gx:.1f},{gy:.1f})")
                idx += 1
                point_samples = []
                point_started = time.monotonic()

            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(calib_data) < len(calib_points):
            print("Insufficient calibration data.")
            return False

        df = pd.DataFrame(
            calib_data,
            columns=["point_index", "gx", "gy", "sx", "sy", "sample_count", "gaze_std_x", "gaze_std_y"],
        )
        self.gaze_model = LinearRegression().fit(df[["gx", "gy"]], df[["sx", "sy"]])
        self._save_calibration_report(df, time.time() - started)
        print("Calibration complete.")
        return True

    def _save_calibration_report(self, df, duration_sec):
        predictions = self.gaze_model.predict(df[["gx", "gy"]])
        report = df.copy()
        report["pred_sx"] = predictions[:, 0]
        report["pred_sy"] = predictions[:, 1]
        report["error_px"] = np.sqrt(
            ((report["pred_sx"] - report["sx"]) * self.screen_w) ** 2
            + ((report["pred_sy"] - report["sy"]) * self.screen_h) ** 2
        ).round(3)
        report["calibration_mode"] = f"{self.calibration_points}-point"
        report["calibration_duration_sec"] = round(duration_sec, 3)

        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"calibration_report_{ts}.csv")
        report.to_csv(path, index=False)
        self.calibration_report = report.to_dict("records")

        print(
            "Calibration error px: "
            f"mean={report['error_px'].mean():.1f}, "
            f"median={report['error_px'].median():.1f}, "
            f"max={report['error_px'].max():.1f}"
        )

    def map_to_screen(self, gaze_x, gaze_y):
        """Map camera gaze coordinates to clamped screen pixel coordinates."""
        if self.gaze_model is None or gaze_x is None or gaze_y is None:
            return None, None
        sx, sy = self.gaze_model.predict(pd.DataFrame([[gaze_x, gaze_y]], columns=["gx", "gy"]))[0]
        screen_x = int(np.clip(sx * self.screen_w, 0, self.screen_w - 1))
        screen_y = int(np.clip(sy * self.screen_h, 0, self.screen_h - 1))
        return screen_x, screen_y

    def _reset_click_state(self, gaze_x=None, gaze_y=None):
        self._click_start_time = time.time()
        self._click_reference = (gaze_x, gaze_y) if gaze_x is not None and gaze_y is not None else None

    def _update_dwell_click(self, frame, gaze_x, gaze_y, confidence, pyautogui_module):
        if gaze_x is None or gaze_y is None or confidence < self.min_click_confidence:
            self._reset_click_state()
            return

        if self._click_reference is None:
            self._reset_click_state(gaze_x, gaze_y)
            return

        dx = abs(gaze_x - self._click_reference[0])
        dy = abs(gaze_y - self._click_reference[1])
        if dx > self.fixation_threshold or dy > self.fixation_threshold:
            self._reset_click_state(gaze_x, gaze_y)
            return

        duration = time.time() - self._click_start_time
        remaining = max(0.0, self.auto_click_duration - duration)
        progress = 1.0 - remaining / self.auto_click_duration
        radius = int(30 + 15 * progress)
        color = (0, int(255 * progress), int(255 * (1 - progress)))
        cv2.circle(frame, (gaze_x, gaze_y), radius, color, 2)
        cv2.putText(
            frame,
            f"{remaining:.1f}s",
            (gaze_x + 20, gaze_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        now = time.time()
        if duration >= self.auto_click_duration and now - self._last_click_time >= self.click_cooldown:
            print("Dwell click.")
            pyautogui_module.click()
            self._last_click_time = now
            self._reset_click_state(gaze_x, gaze_y)

    def run(self):
        """Calibrate first, then track with mouse control. Press Q to quit."""
        if not self.run_calibration():
            return

        import pyautogui

        if not self._open_capture():
            return

        print("Calibrated tracking started. Press Q to quit.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if not isinstance(self.video_source, int):
                        print("Video source ended.")
                        break
                    continue

                record = self.process_frame(frame)
                gaze_x = record["gaze_x"]
                gaze_y = record["gaze_y"]
                confidence = record["gaze_confidence"]

                sx, sy = self.map_to_screen(gaze_x, gaze_y)
                if sx is not None and confidence >= self.min_click_confidence:
                    pyautogui.moveTo(sx, sy, duration=0.05)
                    cv2.putText(
                        frame,
                        f"Screen: ({sx},{sy}) conf={confidence:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                self._update_dwell_click(frame, gaze_x, gaze_y, confidence, pyautogui)

                cv2.imshow("Gaze Tracking (Calibrated)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if self.flush_interval > 0 and self.frame_count % self.flush_interval == 0:
                    self._flush_data()

        except KeyboardInterrupt:
            print("\nInterrupted; saving data.")
        finally:
            self.cleanup()
