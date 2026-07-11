"""Generate simple summaries from trial-level CSV logs."""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def load_trial_logs(input_glob):
    paths = glob.glob(input_glob)
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = os.path.basename(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_trials(df):
    df = df.copy()
    group_cols = [
        "condition",
        "input_method",
        "calibration_mode",
        "gaze_smoothing_window",
        "dwell_method",
        "target_radius",
    ]
    for column in ("success", "missed_selection"):
        if column in df:
            df[column] = df[column].map(
                lambda value: value
                if isinstance(value, bool)
                else str(value).strip().lower() in {"true", "1", "yes"}
            )

    aggregations = {
        "trials": ("trial_id", "count"),
        "success_rate": ("success", "mean"),
        "mean_selection_time_sec": ("selection_time_sec", "mean"),
        "median_selection_time_sec": ("selection_time_sec", "median"),
        "false_clicks_mean": ("false_click_count", "mean"),
        "missed_rate": ("missed_selection", "mean"),
        "mean_distance_px": ("distance_to_target_px", "mean"),
    }
    optional_metrics = {
        "mean_confidence": ("mean_confidence", "mean"),
        "mean_tracking_lost_ratio": ("tracking_lost_ratio", "mean"),
        "mean_fps": ("fps_mean", "mean"),
    }
    aggregations.update(
        {name: spec for name, spec in optional_metrics.items() if spec[0] in df.columns}
    )
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    return summary.round(3)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize experiment trial logs.")
    parser.add_argument("--input", default="results/trial_log_*.csv")
    parser.add_argument("--output", default="results/summary_trials.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_trial_logs(args.input)
    summary = summarize_trials(data)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary to {args.output}")
