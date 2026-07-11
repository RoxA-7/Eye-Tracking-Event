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
- Mouse and full-screen gaze target-selection conditions with trial-level logging.
- Reproducible session configuration files and gaze-quality summaries.

## Project Structure

```text
eye_project/
├── eye_tracker.py                 # Core tracker and calibrated gaze-control classes
├── eye_tracking.py                # Webcam/video tracking entry point
├── eye_tracking-test.py           # Calibrated gaze mouse-control entry point
├── experiment_logger.py           # Stable trial-level CSV schema and logger
├── target_selection_experiment.py # Mouse/gaze target-selection experiment
├── experiment_protocol.md         # Reproducible study conditions and run rules
├── tests/                         # Hardware-independent experiment tests
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

## Run Target-Selection Experiments

Run the mouse baseline:

```powershell
python target_selection_experiment.py --participant-id pilot01 --session-id pilot01_mouse --input-method mouse --trials-per-radius 8 --seed 42
```

Run a 9-point gaze condition with confidence-aware dwell:

```powershell
python target_selection_experiment.py --participant-id pilot01 --session-id pilot01_gaze9 --input-method gaze --calibration-points 9 --dwell-method confidence_aware --seed 42
```

Use `--dwell-method fixed` for the dwell ablation and
`--calibration-points 5` for the calibration-density comparison. Gaze mode uses
a full-screen surface so calibration, mapped gaze, and targets share screen
coordinates. Each run saves a session configuration JSON and trial-level CSV;
gaze runs additionally save calibration and raw tracking data.

The formal condition matrix, exclusion rules, and output checks are documented
in `eye_project/experiment_protocol.md`.

## Analyze Trial Logs

```powershell
python analysis\analyze_results.py --input "results/trial_log_*.csv" --output results\summary_trials.csv
```

The script groups results by experiment condition, calibration, smoothing,
dwell method, and target size. It reports success, selection time, false clicks,
misses, distance, confidence, tracking loss, and FPS when those fields exist.

## Run Hardware-Independent Tests

```powershell
python -m unittest discover -s tests -v
```

## Research Notes

For conference submission planning, see `conference_submission_plan.md`.
Do not report experiment results in a paper unless they are traceable to raw
CSV logs and reproducible analysis scripts.

## License

MIT
