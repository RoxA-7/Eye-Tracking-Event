"""Trial-level CSV logger for gaze and mouse experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import csv
import os
import time


TRIAL_FIELDS = [
    "participant_id",
    "session_id",
    "trial_id",
    "condition",
    "input_method",
    "calibration_mode",
    "gaze_smoothing_window",
    "dwell_method",
    "dwell_time_sec",
    "target_id",
    "target_x",
    "target_y",
    "target_radius",
    "trial_start_time",
    "first_valid_gaze_time",
    "selection_time_sec",
    "success",
    "false_click_count",
    "missed_selection",
    "gaze_x",
    "gaze_y",
    "screen_x",
    "screen_y",
    "click_x",
    "click_y",
    "distance_to_target_px",
    "face_detected",
    "pupil_detected_left",
    "pupil_detected_right",
    "tracking_lost_ratio",
    "mean_confidence",
    "fps_mean",
    "notes",
]


@dataclass
class TrialRecord:
    participant_id: str
    session_id: str
    trial_id: int
    condition: str
    input_method: str
    calibration_mode: str
    gaze_smoothing_window: int | None
    dwell_method: str
    dwell_time_sec: float | None
    target_id: str
    target_x: int
    target_y: int
    target_radius: int
    trial_start_time: str
    first_valid_gaze_time: str | None = None
    selection_time_sec: float | None = None
    success: bool = False
    false_click_count: int = 0
    missed_selection: bool = False
    gaze_x: int | None = None
    gaze_y: int | None = None
    screen_x: int | None = None
    screen_y: int | None = None
    click_x: int | None = None
    click_y: int | None = None
    distance_to_target_px: float | None = None
    face_detected: bool | None = None
    pupil_detected_left: bool | None = None
    pupil_detected_right: bool | None = None
    tracking_lost_ratio: float | None = None
    mean_confidence: float | None = None
    fps_mean: float | None = None
    notes: str = ""


class TrialLogger:
    """Append-only CSV logger with a stable schema."""

    def __init__(self, output_dir="results", filename=None):
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = f"trial_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.path = os.path.join(output_dir, filename)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=TRIAL_FIELDS)
        self._writer.writeheader()

    def write(self, record: TrialRecord):
        row = asdict(record)
        self._writer.writerow({field: row.get(field) for field in TRIAL_FIELDS})
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def iso_now():
    return datetime.now().isoformat(timespec="milliseconds")


def elapsed_since(start_time):
    return round(time.time() - start_time, 3)
