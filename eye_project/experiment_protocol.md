# Target-Selection Experiment Protocol

## Purpose

Compare mouse selection with webcam-gaze selection while varying calibration
density, gaze smoothing, and dwell activation. Treat all runs before the final
protocol freeze as pilots.

## Conditions

Run these minimum conditions for each participant:

1. Mouse baseline.
2. Gaze with 5-point calibration and confidence-aware dwell.
3. Gaze with 9-point calibration and confidence-aware dwell.
4. Gaze with 9-point calibration and fixed dwell.

Keep target radii, trial count, timeout, and dwell duration constant. Use the
same random seed across conditions so target locations are comparable. Rotate
condition order across participants to reduce learning and fatigue effects.

## Standard Commands

Run commands from `eye_project/` and replace the participant/session IDs.

```powershell
python target_selection_experiment.py --participant-id P001 --session-id P001_mouse --input-method mouse --seed 42
python target_selection_experiment.py --participant-id P001 --session-id P001_gaze5_conf --input-method gaze --calibration-points 5 --dwell-method confidence_aware --seed 42
python target_selection_experiment.py --participant-id P001 --session-id P001_gaze9_conf --input-method gaze --calibration-points 9 --dwell-method confidence_aware --seed 42
python target_selection_experiment.py --participant-id P001 --session-id P001_gaze9_fixed --input-method gaze --calibration-points 9 --dwell-method fixed --seed 42
```

## Before Each Run

- Record only anonymous participant IDs.
- Keep camera position, screen resolution, seating distance, and room lighting
  stable or document the change.
- Confirm the participant can see every calibration point and target.
- Ask the participant to keep their head in a comfortable, repeatable position.
- Stop and restart calibration if multiple points cannot collect stable samples.

## Exclusion and Failure Rules

- Do not edit raw CSV or JSON files manually.
- Mark a run invalid if the wrong participant ID, condition, or screen was used.
- Preserve timed-out and false-activation trials; they are outcomes, not missing
  data.
- Exclude a gaze run only by a protocol rule decided before formal collection,
  such as calibration failure or a documented camera interruption.
- Record every excluded session and reason in the study log.

## Outputs and Verification

Each session must produce a `session_config_*.json` and `trial_log_*.csv`. Gaze
sessions also produce a `calibration_report_*.csv` and raw eye-tracking CSV.
Generate summaries without editing source data:

```powershell
python analysis\analyze_results.py --input "results/trial_log_*.csv" --output results\summary_trials.csv
```

Before analysis, verify participant IDs, expected trial counts, condition names,
screen dimensions, random seed, and calibration mode against the JSON files.
