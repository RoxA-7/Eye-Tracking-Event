# Eye Tracking Project

Low-cost webcam-based gaze tracking and gaze interaction prototype built with
OpenCV, MediaPipe, pandas, matplotlib, scikit-learn, and PyAutoGUI. The current
dependency set is validated against Python 3.12.

## Features

- Pupil detection with MediaPipe Face Mesh eye landmarks and OpenCV adaptive thresholding.
- Real-time gaze point estimation from both pupil centers with configurable smoothing.
- I-VT style fixation detection with jitter tolerance.
- Blink-like pupil-loss counting and fatigue warning.
- Traceable CSV outputs for gaze, pupil, face-detection, confidence, blink, FPS, and fixation data.
- Gaze heatmap export.
- 5-point, 9-point, and 13-point calibration modes for screen mapping.
- Confidence-gated dwell click with click cooldown and clamped screen coordinates.
- Mouse baseline target-selection experiment with trial-level logging.
- Basic analysis script for trial-level experiment summaries.

## Project Structure

```text
eye_project/
├── eye_tracker.py                 # Core tracker and calibrated gaze-control classes
├── eye_tracking.py                # Webcam/video tracking entry point
├── eye_tracking-test.py           # Calibrated gaze mouse-control entry point
├── experiment_logger.py           # Stable trial-level CSV schema and logger
├── target_selection_experiment.py # Mouse baseline target-selection experiment
├── analysis/
│   └── analyze_results.py         # Summary script for trial logs
├── requirements.txt               # Python dependencies
└── setup_env.ps1                  # PowerShell environment setup
```

Runtime outputs are written to `results/`, which is ignored by Git.

## Setup

```powershell
cd eye_project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or run:

```powershell
cd eye_project
.\setup_env.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Run Gaze Tracking

Use the file picker, or cancel to use the webcam:

```powershell
python eye_tracking.py
```

Use webcam directly:

```powershell
python eye_tracking.py --source 0
```

Use a video file:

```powershell
python eye_tracking.py --source path\to\video.mp4
```

Useful options:

```powershell
python eye_tracking.py --source 0 --process-every-n-frames 2 --flush-interval 150
```

Press `Q` to quit. The tracker writes gaze CSV, fixation CSV, and heatmap PNG files to `results/`.

## Run Calibrated Gaze Mouse Control

```powershell
python eye_tracking-test.py --calibration-points 9 --auto-click-duration 3.0
```

Supported calibration modes:

- `--calibration-points 5`
- `--calibration-points 9`
- `--calibration-points 13`

The calibrated mode writes a `calibration_report_*.csv` file with per-point
training residuals. Dwell clicks are suppressed when gaze confidence is below
`--min-click-confidence`, and repeated clicks are limited by `--click-cooldown`.

## Run Mouse Baseline Experiment

This provides a controlled target-selection surface before gaze conditions are
integrated.

```powershell
python target_selection_experiment.py --participant-id pilot01 --trials-per-radius 8
```

It logs trial-level CSV files with target size, click location, success, false
click count, timeout/miss, and selection time.

## Analyze Trial Logs

```powershell
python analysis\analyze_results.py --input "results/trial_log_*.csv" --output results\summary_trials.csv
```

The script reports success rate, selection time, false clicks, miss rate, and
distance-to-target summaries grouped by condition, input method, and target size.

## Research Notes

For conference submission planning, see `conference_submission_plan.md`.
Do not report experiment results in a paper unless they are traceable to raw
CSV logs and reproducible analysis scripts.

## License

MIT
