"""
Eye Tracking Engine — shared module for pupil detection and gaze analysis.

Uses MediaPipe Face Mesh to locate eyes, adaptive thresholding to find pupils,
and provides gaze smoothing, fixation detection, blink counting, heatmap generation,
and data export.

Usage:
    from eye_tracker import EyeTracker
    tracker = EyeTracker(video_source=0)
    tracker.run()
"""

import cv2
import numpy as np
import time
import mediapipe as mp
import pandas as pd
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import os
import platform
import subprocess


class EyeTracker:
    """Core eye tracking engine.

    Parameters
    ----------
    video_source : int or str
        0 for webcam, or path to video file.
    enable_heatmap : bool
        Overlay a gaze heatmap on the frame.
    enable_fixation : bool
        Detect and record fixations.
    enable_blink_detection : bool
        Count blinks and trigger fatigue alerts.
    gaze_smoothing : int
        Number of frames to average for gaze smoothing (buffer size).
    process_every_n_frames : int
        Run MediaPipe only every N frames; reuse last landmarks on others.
        Higher = faster but less responsive (1 = every frame, 2 = every 2nd).
    flush_interval : int
        Auto-save data to disk every N frames (0 = only at exit).
    show_debug_windows : bool
        Show per-eye threshold debug windows (adds latency).
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
    ):
        self.video_source = video_source
        self.enable_heatmap = enable_heatmap
        self.enable_fixation = enable_fixation
        self.enable_blink_detection = enable_blink_detection
        self.process_every_n_frames = max(1, process_every_n_frames)
        self.flush_interval = flush_interval
        self.show_debug_windows = show_debug_windows

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )

        # Gaze smoothing
        self.gaze_buffer = deque(maxlen=gaze_smoothing)

        # Heatmap
        self.heatmap_layer = None

        # Fixation state
        self.fixations = []
        self.fixation_threshold = 15  # pixels
        self.min_fixation_duration = 0.5  # seconds
        self.fixation_start_time = None
        self.fixation_reference = None
        # Jitter tolerance: allow N out-of-threshold frames before resetting fixation
        self._jitter_counter = 0
        self._jitter_tolerance = 3

        # Blink / fatigue
        self.blink_count = 0
        self.fatigue_alert_triggered = False
        # Require N consecutive pupil-loss frames to count as one blink
        self._blink_loss_frames = 0
        self._blink_loss_threshold = 3

        # Recording
        self.data_records = []
        self.frame_count = 0
        self.start_time = time.time()

        # Frame-skip state: reuse last landmarks on non-key frames
        self._last_landmarks = None

        # Video capture
        self.cap = None

    # ── geometry helpers ──────────────────────────────────────────────

    def extract_eye_roi(self, image, landmarks, eye_indices, margin=10):
        """Crop the eye region from the frame using normalized landmarks.

        Returns (roi_image, (x_offset, y_offset)).
        """
        h, w = image.shape[:2]
        x1 = int(landmarks[eye_indices[0]].x * w)
        y1 = int(landmarks[eye_indices[0]].y * h)
        x2 = int(landmarks[eye_indices[1]].x * w)
        y2 = int(landmarks[eye_indices[1]].y * h)

        x_min = max(min(x1, x2) - margin, 0)
        y_min = max(min(y1, y2) - margin, 0)
        x_max = min(max(x1, x2) + margin, w)
        y_max = min(max(y1, y2) + margin, h)

        roi = image[y_min:y_max, x_min:x_max]
        return roi, (x_min, y_min)

    # ── pupil detection ───────────────────────────────────────────────

    def detect_pupil(self, roi, draw_on=None, offset=(0, 0), label=None):
        """Detect pupil centre in an eye ROI via adaptive thresholding.

        Parameters
        ----------
        roi : np.ndarray
            Eye region image (BGR).
        draw_on : np.ndarray or None
            Frame to draw detection overlay on (optional).
        offset : tuple
            (x, y) offset to convert ROI-local coords to full-frame coords.
        label : str or None
            If given, show the threshold debug window with this title.

        Returns
        -------
        (cx, cy) in full-frame pixel coordinates, or None.
        """
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.equalizeHist(gray)

        # Adaptive threshold — robust to varying lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        if label and self.show_debug_windows:
            cv2.imshow(label, thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest valid contour = pupil
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            ox, oy = offset

            if draw_on is not None:
                # Bounding box
                cv2.rectangle(
                    draw_on, (ox + x, oy + y), (ox + x + w, oy + y + h),
                    (255, 0, 0), 2,
                )
                # Crosshair
                cv2.line(draw_on, (ox + cx, oy + y), (ox + cx, oy + y + h), (0, 255, 0), 1)
                cv2.line(draw_on, (ox + x, oy + cy), (ox + x + w, oy + cy), (0, 255, 0), 1)

            return (ox + cx, oy + cy)

        return None

    # ── gaze computation ──────────────────────────────────────────────

    def compute_gaze(self, left_center, right_center):
        """Average left & right pupil centres + smooth over time.

        Returns (gaze_x, gaze_y) or (None, None).
        """
        if left_center is None or right_center is None:
            return None, None

        raw_x = (left_center[0] + right_center[0]) // 2
        raw_y = (left_center[1] + right_center[1]) // 2
        self.gaze_buffer.append((raw_x, raw_y))

        gaze_x = int(sum(p[0] for p in self.gaze_buffer) / len(self.gaze_buffer))
        gaze_y = int(sum(p[1] for p in self.gaze_buffer) / len(self.gaze_buffer))
        return gaze_x, gaze_y

    # ── heatmap ───────────────────────────────────────────────────────

    def update_heatmap(self, gaze_point, frame):
        """Add gaze point to persistent heatmap layer and blend onto frame.

        Returns the blended frame (modified in place).
        """
        if not self.enable_heatmap:
            return frame

        gx, gy = gaze_point
        if self.heatmap_layer is None:
            self.heatmap_layer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        cv2.circle(self.heatmap_layer, (gx, gy), 3, 1, -1)
        # Decay: prevent unbounded saturation over long sessions
        self.heatmap_layer *= 0.995
        heatmap_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(self.heatmap_layer, alpha=10), cv2.COLORMAP_JET
        )
        return cv2.addWeighted(frame, 0.7, heatmap_vis, 0.3, 0)

    # ── fixation detection ────────────────────────────────────────────

    def detect_fixation(self, gaze_point):
        """I-VT (Velocity-Threshold) fixation detector.

        Records a fixation when gaze stays within `fixation_threshold` pixels
        for at least `min_fixation_duration` seconds.

        Returns a dict if a new fixation was just completed, else None.
        """
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
            self._jitter_counter = 0  # reset jitter on stable frame
            duration = time.time() - self.fixation_start_time
            if duration >= self.min_fixation_duration:
                fix = {
                    "start_time": self.fixation_start_time,
                    "duration_sec": round(duration, 3),
                    "gaze_x": gx,
                    "gaze_y": gy,
                }
                # Reset for next fixation
                self.fixation_start_time = time.time()
                self.fixation_reference = (gx, gy)
                self._jitter_counter = 0
                return fix
        else:
            # Allow a few jitter frames before truly resetting
            self._jitter_counter += 1
            if self._jitter_counter > self._jitter_tolerance:
                self.fixation_start_time = time.time()
                self.fixation_reference = (gx, gy)
                self._jitter_counter = 0

        return None

    # ── blink detection ───────────────────────────────────────────────

    def detect_blink(self, left_ok, right_ok):
        """Return 1 if a blink was completed (N consecutive pupil-loss frames).

        Uses hysteresis: pupil must be lost for `_blink_loss_threshold` frames
        to count as one blink, preventing single-frame dropouts from inflating
        the count.
        """
        if not self.enable_blink_detection:
            return 0
        if not left_ok or not right_ok:
            self._blink_loss_frames += 1
            return 0  # still in potential blink — don't count yet
        else:
            if self._blink_loss_frames >= self._blink_loss_threshold:
                self._blink_loss_frames = 0
                self.blink_count += 1
                return 1
            self._blink_loss_frames = 0
            return 0

    def check_fatigue(self):
        """Print a fatigue warning if blink rate exceeds threshold (after 10 s)."""
        elapsed = time.time() - self.start_time
        if elapsed > 10:
            blink_rate = self.blink_count / elapsed
            if blink_rate > 0.3 and not self.fatigue_alert_triggered:
                print("⚠ 疲劳警告：眨眼频率偏高，请休息！")
                self.fatigue_alert_triggered = True

    # ── frame processing (orchestrator) ───────────────────────────────

    def process_frame(self, frame):
        """Run the full pipeline on one BGR frame.

        Returns
        -------
        dict with keys:
            timestamp, left_x, left_y, right_x, right_y,
            gaze_x, gaze_y, blink, fixation (dict or None)
        """
        if self.heatmap_layer is None and self.enable_heatmap:
            self.heatmap_layer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- MediaPipe: only run on key frames (frame-skip optimization) ---
        if self.frame_count % self.process_every_n_frames == 0:
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                self._last_landmarks = results.multi_face_landmarks[0].landmarks
        landmarks = self._last_landmarks
        results = None  # not used beyond this point

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        left_center = None
        right_center = None
        gaze_x = None
        gaze_y = None
        blink = 0
        fixation = None

        if landmarks is not None:

            # Left eye
            roi_l, offset_l = self.extract_eye_roi(frame, landmarks, self.LEFT_EYE)
            left_center = self.detect_pupil(roi_l, draw_on=frame, offset=offset_l, label="Left Eye")

            # Right eye
            roi_r, offset_r = self.extract_eye_roi(frame, landmarks, self.RIGHT_EYE)
            right_center = self.detect_pupil(roi_r, draw_on=frame, offset=offset_r, label="Right Eye")

            # Gaze
            gaze_x, gaze_y = self.compute_gaze(left_center, right_center)

            if gaze_x is not None:
                # Draw gaze crosshair
                cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
                cv2.line(frame, (gaze_x, 0), (gaze_x, frame.shape[0]), (0, 255, 0), 1)
                cv2.line(frame, (0, gaze_y), (frame.shape[1], gaze_y), (0, 255, 0), 1)

                # Heatmap
                frame = self.update_heatmap((gaze_x, gaze_y), frame)

                # Fixation
                fixation = self.detect_fixation((gaze_x, gaze_y))
                if fixation:
                    self.fixations.append(fixation)

        # Blink
        blink = self.detect_blink(left_center is not None, right_center is not None)

        # Record
        record = {
            "timestamp": timestamp,
            "left_x": left_center[0] if left_center else None,
            "left_y": left_center[1] if left_center else None,
            "right_x": right_center[0] if right_center else None,
            "right_y": right_center[1] if right_center else None,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "blink": blink,
        }
        self.data_records.append(record)

        self.check_fatigue()
        self.frame_count += 1

        return record

    # ── main loop ─────────────────────────────────────────────────────

    def _open_capture(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"无法打开视频源: {self.video_source}")
            return False
        return True

    def _flush_data(self, output_dir="results"):
        """Incrementally save data to disk (crash-safe)."""
        if not self.data_records:
            return
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(self.data_records).to_csv(
            os.path.join(output_dir, "eye_tracking_data_partial.csv"), index=False
        )
        if self.fixations:
            pd.DataFrame(self.fixations).to_csv(
                os.path.join(output_dir, "fixations_partial.csv"), index=False
            )

    def run(self):
        """Start the main tracking loop. Press 'q' to quit."""
        if not self._open_capture():
            return

        print("眼动追踪已启动 — 按 Q 退出")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                self.process_frame(frame)

                # Window
                src_label = "摄像头" if self.video_source == 0 else str(self.video_source)
                cv2.imshow(f"Eye Tracking - {src_label}", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Periodic incremental save
                if self.flush_interval > 0 and self.frame_count % self.flush_interval == 0:
                    self._flush_data()

                if self.frame_count % 30 == 0:
                    fps = self.frame_count / (time.time() - self.start_time)
                    print(f"  FPS: {fps:.1f}")

        except KeyboardInterrupt:
            print("\n检测到中断信号，正在保存数据…")

        finally:
            self.cleanup()

    def cleanup(self):
        """Release camera, destroy windows, save data."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.save_results()

    # ── data export ───────────────────────────────────────────────────

    def save_results(self, output_dir="results"):
        """Export raw data CSV, fixations CSV, and gaze heatmap PNG."""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Raw data
        data_path = os.path.join(output_dir, f"eye_tracking_data_{ts}.csv")
        pd.DataFrame(self.data_records).to_csv(data_path, index=False)

        # Fixations
        fix_path = os.path.join(output_dir, f"fixations_{ts}.csv")
        pd.DataFrame(self.fixations).to_csv(fix_path, index=False)

        # Heatmap image
        if self.heatmap_layer is not None and np.max(self.heatmap_layer) > 0:
            heat_path = os.path.join(output_dir, f"gaze_heatmap_{ts}.png")
            plt.figure(figsize=(10, 6))
            heat_display = cv2.convertScaleAbs(
                self.heatmap_layer, alpha=255.0 / np.max(self.heatmap_layer)
            )
            plt.imshow(heat_display, cmap="jet")
            plt.title("Gaze Heatmap")
            plt.axis("off")
            plt.savefig(heat_path)
            plt.close()

        print(f"数据已保存 → {os.path.abspath(output_dir)}")

        # Open folder
        if platform.system() == "Windows":
            subprocess.Popen(f'explorer {os.path.abspath(output_dir)}')
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", os.path.abspath(output_dir)])
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", os.path.abspath(output_dir)])


class CalibratedEyeTracker(EyeTracker):
    """EyeTracker with calibration → screen mapping + PyAutoGUI mouse control.

    Adds a 5-point calibration phase followed by gaze-to-screen mapping
    using linear regression, real-time mouse movement, and auto-click on
    sustained fixation (default 3 seconds).
    """

    def __init__(self, auto_click_duration=3.0, **kwargs):
        super().__init__(**kwargs)
        self.gaze_model = None
        self.auto_click_duration = auto_click_duration
        self.screen_w = None
        self.screen_h = None
        # Independent auto-click state (separate from base fixation state)
        self._click_start_time = None
        self._click_reference = None

    # ── calibration ───────────────────────────────────────────────────

    def run_calibration(self):
        """5-point calibration: user looks at each point, model is trained.

        Returns True on success.
        """
        import pyautogui  # lazy import — only needed for calibration mode

        print("校准阶段 — 请依次注视屏幕上的黄点")
        calib_data = []
        calib_points = [
            (0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)
        ]
        idx = 0

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return False

        while idx < len(calib_points):
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                roi_l, off_l = self.extract_eye_roi(frame, lm, self.LEFT_EYE)
                roi_r, off_r = self.extract_eye_roi(frame, lm, self.RIGHT_EYE)
                pl = self.detect_pupil(roi_l)
                pr = self.detect_pupil(roi_r)

                if pl and pr:
                    abs_l = (off_l[0] + pl[0], off_l[1] + pl[1])
                    abs_r = (off_r[0] + pr[0], off_r[1] + pr[1])
                    gx = (abs_l[0] + abs_r[0]) // 2
                    gy = (abs_l[1] + abs_r[1]) // 2

                    norm_pt = calib_points[idx]
                    px, py = int(norm_pt[0] * w), int(norm_pt[1] * h)
                    cv2.circle(frame, (px, py), 15, (0, 255, 255), -1)
                    cv2.putText(frame, f"Point {idx+1}/5", (px - 40, py - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    calib_data.append([gx, gy, norm_pt[0], norm_pt[1]])
                    print(f"  已采集点 {idx+1}/5 → 注视({gx},{gy})")
                    idx += 1
                    time.sleep(0.8)

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(calib_data) < 5:
            print("校准数据不足")
            return False

        from sklearn.linear_model import LinearRegression
        df = pd.DataFrame(calib_data, columns=["gx", "gy", "sx", "sy"])
        self.gaze_model = LinearRegression().fit(df[["gx", "gy"]], df[["sx", "sy"]])
        self.screen_w, self.screen_h = pyautogui.size()
        print("校准完成 ✓")
        return True

    # ── screen mapping ────────────────────────────────────────────────

    def map_to_screen(self, gaze_x, gaze_y):
        """Map camera gaze coords → screen pixel coords via calibration model."""
        if self.gaze_model is None:
            return None, None
        sx, sy = self.gaze_model.predict([[gaze_x, gaze_y]])[0]
        return int(sx * self.screen_w), int(sy * self.screen_h)

    # ── tracking loop (override) ──────────────────────────────────────

    def run(self):
        """Calibrate first, then track with mouse control. Press 'q' to quit."""
        if not self.run_calibration():
            return

        import pyautogui

        if not self._open_capture():
            return

        print("追踪已启动 — 鼠标将跟随你的视线，注视 3 秒自动点击 — 按 Q 退出")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                self.process_frame(frame)

                # Get the latest gaze from buffer
                if self.gaze_buffer:
                    gaze_x = int(sum(p[0] for p in self.gaze_buffer) / len(self.gaze_buffer))
                    gaze_y = int(sum(p[1] for p in self.gaze_buffer) / len(self.gaze_buffer))

                    sx, sy = self.map_to_screen(gaze_x, gaze_y)
                    if sx is not None:
                        pyautogui.moveTo(sx, sy, duration=0.05)
                        cv2.putText(frame, f"Screen: ({sx},{sy})", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Auto-click countdown ring — uses independent state
                if self._click_reference is None:
                    self._click_start_time = time.time()
                    self._click_reference = (gaze_x, gaze_y)
                else:
                    dx = abs(gaze_x - self._click_reference[0])
                    dy = abs(gaze_y - self._click_reference[1])
                    if dx <= self.fixation_threshold and dy <= self.fixation_threshold:
                        duration = time.time() - self._click_start_time
                        remaining = max(0, self.auto_click_duration - duration)
                        progress = 1.0 - remaining / self.auto_click_duration
                        radius = int(30 + 15 * progress)
                        color = (0, int(255 * progress), int(255 * (1 - progress)))
                        cv2.circle(frame, (gaze_x, gaze_y), radius, color, 2)
                        cv2.putText(frame, f"{remaining:.1f}s",
                                    (gaze_x + 20, gaze_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        if duration >= self.auto_click_duration:
                            print("🖱 自动点击！")
                            pyautogui.click()
                            self._click_start_time = time.time()
                            self._click_reference = (gaze_x, gaze_y)
                    else:
                        self._click_start_time = time.time()
                        self._click_reference = (gaze_x, gaze_y)

                cv2.imshow("Gaze Tracking (Calibrated)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Periodic incremental save
                if self.flush_interval > 0 and self.frame_count % self.flush_interval == 0:
                    self._flush_data()

        except KeyboardInterrupt:
            print("\n中断信号…")

        finally:
            self.cleanup()
