from __future__ import annotations

import os
import sys
import unittest

import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from analysis.analyze_results import summarize_trials
from target_selection_experiment import DwellSelector, TargetSelectionExperiment


class DwellSelectorTests(unittest.TestCase):
    def test_selects_after_stable_duration(self):
        selector = DwellSelector(duration_sec=1.0, stability_radius_px=10)
        self.assertIsNone(selector.update((100, 100), True, now=0.0))
        self.assertIsNone(selector.update((105, 103), True, now=0.5))
        self.assertEqual((104, 102), selector.update((104, 102), True, now=1.0))

    def test_resets_on_invalid_or_unstable_gaze(self):
        selector = DwellSelector(duration_sec=1.0, stability_radius_px=5)
        selector.update((10, 10), True, now=0.0)
        selector.update((20, 20), True, now=0.8)
        self.assertIsNone(selector.update((20, 20), True, now=1.1))
        selector.update(None, False, now=1.2)
        self.assertEqual(0.0, selector.progress(now=2.0))


class TrialGenerationTests(unittest.TestCase):
    def test_trial_generation_is_seeded_and_within_bounds(self):
        kwargs = dict(
            participant_id="P001",
            session_id="test",
            width=800,
            height=600,
            trials_per_radius=2,
            radii=(30, 50),
            seed=7,
        )
        first = TargetSelectionExperiment(**kwargs).build_trials()
        second = TargetSelectionExperiment(**kwargs).build_trials()
        self.assertEqual(first, second)
        self.assertEqual(4, len(first))
        for trial in first:
            radius = trial["target_radius"]
            self.assertGreaterEqual(trial["target_x"], radius)
            self.assertLessEqual(trial["target_x"], 800 - radius)
            self.assertGreaterEqual(trial["target_y"], radius)
            self.assertLessEqual(trial["target_y"], 600 - radius)


class AnalysisTests(unittest.TestCase):
    def test_summary_includes_gaze_quality_metrics(self):
        frame = pd.DataFrame(
            [
                {
                    "condition": "gaze_9pt_confidence_aware",
                    "input_method": "gaze",
                    "calibration_mode": "9-point",
                    "gaze_smoothing_window": 5,
                    "dwell_method": "confidence_aware",
                    "target_radius": 50,
                    "trial_id": 1,
                    "success": "True",
                    "selection_time_sec": 1.2,
                    "false_click_count": 0,
                    "missed_selection": "False",
                    "distance_to_target_px": 12.0,
                    "mean_confidence": 0.8,
                    "tracking_lost_ratio": 0.1,
                    "fps_mean": 29.0,
                }
            ]
        )
        summary = summarize_trials(frame)
        self.assertEqual(1.0, summary.loc[0, "success_rate"])
        self.assertEqual(0.8, summary.loc[0, "mean_confidence"])
        self.assertEqual(0.1, summary.loc[0, "mean_tracking_lost_ratio"])


if __name__ == "__main__":
    unittest.main()
