# Eye-Tracking-Event Conference Submission Plan

本文档是 Eye-Tracking-Event 项目的会议投稿执行计划，也可作为同类研究项目的通用模板。它把项目事实、论文主张、实验任务、复现要求、写作进度、投稿检查和投稿后工作放在同一份可维护文档中。

## 1. Document Maintenance Rules

### 1.1 Purpose and Scope

本计划用于支持以下工作：

- 研究问题和论文贡献冻结；
- 实验设计、实验执行和结果追踪；
- 论文写作、内部审稿和投稿准备；
- 双盲、复现、代码与数据声明检查；
- 投稿后 rebuttal、revision、camera-ready 和 artifact release。

它不替代论文正文，也不替代实验日志。凡是论文中出现的核心结论，都必须能追溯到本文档中的 claim-evidence 矩阵、实验记录或外部文献证据。

### 1.2 Fact Sources Used in This Revision

本次修订只把以下仓库内容作为可确认事实：

| Source | Verified Facts |
|---|---|
| `README.md` | 项目是基于 OpenCV、MediaPipe、pandas、matplotlib 的普通摄像头眼动追踪项目，支持 gaze tracking、fixation、blink、heatmap、CSV 输出和校准版鼠标控制入口 |
| `eye_project/eye_tracker.py` | 核心类为 `EyeTracker` 和 `CalibratedEyeTracker`，包含 MediaPipe Face Mesh、瞳孔阈值检测、gaze smoothing、I-VT fixation、blink 计数、heatmap、CSV/PNG 输出、5/9/13 点校准、线性回归映射、gaze confidence、screen clamp 和 confidence-gated dwell auto-click |
| `eye_project/eye_tracking.py` | 普通追踪入口，支持摄像头或视频文件输入 |
| `eye_project/eye_tracking-test.py` | 校准和鼠标控制入口，默认 3 秒 dwell click |
| `eye_project/requirements.txt` | 已声明 `opencv-python`、`mediapipe==0.10.35`、`pandas`、`matplotlib`、`scikit-learn`、`pyautogui` |
| `eye_project/experiment_logger.py` | 已定义 trial-level CSV schema 和 append-only logger |
| `eye_project/target_selection_experiment.py` | 已提供可控 mouse baseline target-selection 实验入口 |
| `eye_project/analysis/analyze_results.py` | 已提供 trial log 摘要分析脚本 |
| `plan_iui_2027.md`、`plan_chi_2027.md`、`plan_ieee_vr_2027.md` | 现有投稿路线建议，作为待确认的战略参考，不作为已验证会议要求 |

未在仓库中找到 `paper/`、`docs/`、`manuscript/`、`experiments/`、`analysis/`、数据集说明、正式实验结果或论文源文件。因此，任何实验结果、样本量结论、会议截止日期、页数限制和模板要求都必须标记为待确认。

### 1.3 Priority System

| Priority | Meaning |
|---|---|
| `P0` | 影响论文有效性、研究诚信或投稿资格，必须完成 |
| `P1` | 直接支撑核心贡献，正常应完成 |
| `P2` | 增强说服力，时间允许时完成 |
| `P3` | 长期优化，不要求当前投稿完成 |

### 1.4 Status System

| Status | Meaning |
|---|---|
| `NOT_STARTED` | 尚未开始 |
| `IN_PROGRESS` | 正在进行 |
| `BLOCKED` | 被明确依赖阻塞，必须记录原因、所需输入和下一步 |
| `NEEDS_REVIEW` | 已有初稿或实现，等待审查 |
| `DONE` | 已完成并可追溯验证 |
| `DEFERRED` | 暂缓，不进入当前关键路径 |
| `NOT_APPLICABLE` | 不适用于当前投稿路线 |

### 1.5 Rules for Uncertain Information

- 不编造实验结果、统计显著性、性能提升、样本量结论、会议要求或引用。
- 对无法从仓库验证的信息使用 `To Be Confirmed`、`TODO` 或 `BLOCKED`。
- 计划中的实验不得写成已完成实验。
- 论文主张必须先进入 claim-evidence 矩阵，再进入摘要、贡献列表或结论。
- 如果会议官网、模板或截止日期发生变化，以官方 CFP 和投稿系统为准。

## 2. Project Snapshot

### 2.1 Current Project Overview

| Field | Current Value |
|---|---|
| Repository | `https://github.com/RoxA-7/Eye-Tracking-Event` |
| Local root | `D:\Eye-Tracking-Event` |
| Current system | Low-cost webcam-based gaze tracking and interaction prototype |
| Main implementation | `eye_project/eye_tracker.py` |
| Main entry point | `eye_project/eye_tracking.py` |
| Calibrated entry point | `eye_project/eye_tracking-test.py` |
| Confirmed output | `results/eye_tracking_data_*.csv`, `results/fixations_*.csv`, `results/gaze_heatmap_*.png` when the program runs |
| Confirmed calibration | 5-point, 9-point, and 13-point calibration in `CalibratedEyeTracker.run_calibration()` |
| Confirmed dwell behavior | Default 3-second auto-click after sustained fixation in calibrated mode, with confidence threshold and click cooldown |
| Paper source | `NOT_STARTED`: no paper source file found |
| Formal experiment dataset | `NOT_STARTED`: no formal dataset or participant data found |
| Analysis scripts | `IN_PROGRESS`: `eye_project/analysis/analyze_results.py` summarizes trial logs |

### 2.2 Current Research Framing

Recommended positioning, based on current implementation:

```text
We present and evaluate a low-cost webcam-based gaze interaction system for hands-free pointing and selection.
```

避免把项目写成新的眼动追踪算法。当前方法主要整合 MediaPipe Face Mesh、OpenCV 阈值瞳孔检测、线性回归校准、gaze smoothing、fixation 和 dwell-click，合理贡献应放在低成本交互系统、实验评估和可用性边界上。

### 2.3 Factual Correction from Earlier Plan

旧计划中提到 `CalibratedEyeTracker.run_calibration()` 对 pupil 坐标重复加 offset。代码复核结果如下：

- `detect_pupil(roi, offset=(0, 0))` 默认返回 ROI 内坐标；
- 普通追踪路径调用 `detect_pupil(..., offset=offset_l)`，返回全局帧坐标；
- 校准路径调用 `detect_pupil(roi_l)`，随后手动加 `off_l`，当前看起来是单次 offset 转换，不是已确认的重复 offset bug。

因此，校准坐标问题不应作为已确认 bug 写入论文或实验前提。更稳妥的 P0 任务是统一坐标 API、增加回归检查，并记录校准误差。

## 3. Submission Target

### 3.1 Target Options

| Route | Role | Current Fit | Submission Type | Status | Notes |
|---|---|---:|---|---|---|
| ACM IUI 2027 | Primary candidate | High | Short paper / full paper / demo | `To Be Confirmed` | 与智能用户界面、gaze interaction 和 dwell selection 最匹配 |
| ACM CHI 2027 | Backup / stretch | Medium-High | Poster / interactive demo / full paper | `To Be Confirmed` | full paper 需要更强 HCI framing、更大用户研究和设计启示 |
| IEEE VR 2027 | Conditional backup | Medium | Poster / demo / paper | `To Be Confirmed` | 当前项目没有 VR/AR/3D 场景，必须扩展 3D UI 才适合 |

### 3.2 Conference Requirement Checklist

在确认具体会议之前，不填写具体截止日期或页数。

| Requirement | ACM IUI 2027 | ACM CHI 2027 | IEEE VR 2027 |
|---|---|---|---|
| Official CFP URL | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Abstract deadline | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Main paper deadline | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Supplement deadline | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Time zone | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Page limit | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Double-blind rule | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Anonymous repository rule | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Template version | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Artifact / code / data policy | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |
| Ethics requirement | `To Be Confirmed` | `To Be Confirmed` | `To Be Confirmed` |

### 3.3 Route Decision Rule

- Choose IUI if the project can deliver a stable target-selection experiment, trial-level logs, baseline comparison, and a credible intelligent dwell or confidence mechanism.
- Choose CHI poster/demo if formal experiments are limited but the toolkit and demonstration can be polished.
- Choose CHI full paper only if there is enough time for a broader HCI study, realistic tasks, qualitative analysis, and transferable design implications.
- Choose IEEE VR only after a real 3D UI task exists and the claims are framed around 3D user interfaces rather than immersive VR hardware.

## 4. Research Questions and Contribution Freeze

### 4.1 Candidate Research Questions

| ID | Research Question | Priority | Status |
|---|---|---|---|
| RQ1 | Can a low-cost webcam-based gaze system support usable hands-free target selection? | `P0` | `NOT_STARTED` |
| RQ2 | How do calibration density and gaze smoothing affect pointing accuracy and selection performance? | `P1` | `NOT_STARTED` |
| RQ3 | Can confidence-aware dwell selection reduce false activations under noisy webcam gaze estimates? | `P1` | `NOT_STARTED` |
| RQ4 | What usability limits appear under ordinary webcam conditions? | `P2` | `NOT_STARTED` |

### 4.2 Candidate Contributions

| Contribution | Current Status | Evidence Needed |
|---|---|---|
| A low-cost webcam-based gaze interaction prototype combining face landmark detection, pupil localization, smoothing, calibration, fixation, and dwell selection | `NEEDS_REVIEW` | Code review, system diagram, reproducible demo commands |
| A target-selection experimental protocol for evaluating webcam gaze interaction | `NEEDS_REVIEW` | Mouse/gaze conditions, trial schema, session configs, and protocol exist; pilot validation still needed |
| A comparison of mouse, 5-point gaze, and 9-point gaze selection | `IN_PROGRESS` | 9-point calibration exists; formal participant data still needed |
| A confidence-aware dwell-click mechanism to reduce false activation | `IN_PROGRESS` | Confidence metric and dwell suppression exist; ablation still needed |
| Practical usability limits and design implications | `NOT_STARTED` | Quantitative results, participant feedback, failure cases |

## 5. Claim-Evidence Matrix

No claim below is ready for the paper abstract or conclusion until its status becomes `DONE` or `NEEDS_REVIEW` with evidence attached.

| Paper Claim | Supporting Experiment / Evidence | Current Status | Evidence Location | Risk |
|---|---|---|---|---|
| The system can track gaze using a commodity webcam and export gaze/fixation data | Existing implementation smoke test | `NEEDS_REVIEW` | `eye_project/eye_tracker.py`, runtime `results/` output | Needs real run log and example output |
| The system supports hands-free pointing and dwell-based selection | Calibrated mode demo | `NEEDS_REVIEW` | `eye_project/eye_tracking-test.py` | Needs calibration validation and screen-bound handling |
| 9-point calibration improves selection performance over 5-point calibration | Controlled target-selection study | `IN_PROGRESS` | `eye_project/eye_tracker.py`, `TODO: formal trial logs` | Calibration mode exists, but no participant comparison data yet |
| Confidence-aware dwell reduces false clicks | Ablation: fixed dwell vs confidence-aware dwell | `IN_PROGRESS` | `eye_project/target_selection_experiment.py`, `TODO: ablation logs` | Both conditions are runnable, but participant data is still required |
| Webcam gaze has practical usability limits under small targets or unstable tracking | Error and failure-case analysis | `NOT_STARTED` | `eye_project/analysis/analyze_results.py`, `TODO: formal trial logs` | Requires participant data and honest negative results |
| The project is reproducible by other researchers | Reproducibility checklist and clean environment run | `NEEDS_REVIEW` | `README.md`, `requirements.txt`, `setup_env.ps1` | Needs clean-machine installation test |

Blocked item details:

| Item | Reason | Required Input | Next Step | Critical Path |
|---|---|---|---|---|
| Clean environment verification | Dependencies are declared, but a fresh environment install has not been run in this task | Clean install result | Run `.\setup_env.ps1` or equivalent in a fresh environment | Yes |

## 6. Experiment Plan

### 6.1 Experiment Backlog

| ID | Experiment | Objective | Priority | Status | Dependencies | Output |
|---|---|---|---|---|---|---|
| E0 | System smoke test | Verify webcam/video input, gaze CSV, fixation CSV, and heatmap output | `P0` | `NOT_STARTED` | Camera or sample video | Example `results/` folder and run notes |
| E1 | Calibration validation | Measure 5-point calibration error and verify coordinate mapping | `P0` | `NOT_STARTED` | Stable calibrated run | Calibration report CSV |
| E2 | Target-selection UI | Provide controlled target selection instead of raw desktop mouse control | `P0` | `NEEDS_REVIEW` | Experiment UI | Mouse and full-screen gaze conditions are implemented; hardware pilot remains |
| E3 | Mouse baseline | Establish upper-bound baseline for target selection | `P0` | `NEEDS_REVIEW` | E2 | Mouse runner and reproducible session config exist; pilot data still needed |
| E4 | 5-point vs 9-point gaze | Compare calibration density | `P1` | `IN_PROGRESS` | 9-point calibration | 9-point calibration exists; formal comparison still needed |
| E5 | Smoothing ablation | Compare with and without gaze smoothing | `P1` | `NOT_STARTED` | Configurable smoothing | Stability and selection plots |
| E6 | Fixed dwell vs confidence-aware dwell | Test false activation reduction | `P1` | `NEEDS_REVIEW` | Confidence metric | Both experiment conditions and false-activation logging exist; pilot data still needed |
| E7 | Robustness / failure cases | Characterize lighting, glasses, face loss, pupil loss, head movement | `P2` | `NOT_STARTED` | Logging fields | Failure-case table |
| E8 | Realistic tasks | Test web navigation, media control, or communication-board tasks | `P2` | `NOT_STARTED` | Stable toolkit/demo | Task success and qualitative notes |
| E9 | 3D UI selection | Prepare IEEE VR route | `P3` | `DEFERRED` | 3D UI implementation | 2D/3D comparison |

### 6.2 Required Trial-Level Log Schema

At minimum, target-selection experiments should record:

```text
participant_id
session_id
trial_id
condition
input_method
calibration_mode
gaze_smoothing_window
dwell_method
dwell_time_sec
target_id
target_x
target_y
target_radius
trial_start_time
first_valid_gaze_time
selection_time_sec
success
false_click_count
missed_selection
gaze_x
gaze_y
screen_x
screen_y
click_x
click_y
distance_to_target_px
face_detected
pupil_detected_left
pupil_detected_right
tracking_lost_ratio
mean_confidence
fps_mean
notes
```

### 6.3 Experiment Definition of Done

An experiment is not `DONE` just because code ran. It must satisfy:

- Protocol, condition order, participant criteria, and exclusion rules are documented.
- Configuration and randomization seed are saved.
- Raw trial logs are saved without manual value editing.
- Data cleaning script is versioned.
- Tables and figures can be regenerated from raw data.
- Abnormal results and failed trials are documented.
- Reported paper values match source CSV values.
- Any user data is anonymized before sharing or submission.

## 7. Reproducibility and Engineering Quality Plan

| Check | Current Status | Required Action | Priority |
|---|---|---|---|
| Environment can be recreated | `NEEDS_REVIEW` | Confirm Python version, test `pip install -r eye_project/requirements.txt` in a clean venv | `P0` |
| Dependencies are complete | `DONE` | `scikit-learn` and `pyautogui` are now declared in requirements | `P0` |
| Minimal run command exists | `DONE` | README documents tracking, calibrated mode, mouse/gaze experiments, tests, and analysis commands | `P0` |
| Data acquisition method is clear | `NOT_STARTED` | Define webcam/video inputs, participant consent, storage policy | `P0` |
| Logs are traceable | `NEEDS_REVIEW` | Mouse/gaze trial logs and per-session JSON configs exist; pilot output audit remains | `P0` |
| Results are generated by scripts | `IN_PROGRESS` | Factor-aware summary script exists; figure generation remains | `P1` |
| Random seeds are recorded | `DONE` | Mouse and gaze conditions share a seeded trial generator and save the seed in session config | `P1` |
| Local paths are avoided | `NEEDS_REVIEW` | Keep output relative to project or configurable output directory | `P1` |
| Secrets and private data are excluded | `NEEDS_REVIEW` | Keep `.env`, `results/`, CSV and PNG outputs ignored unless anonymized examples are intentionally added | `P0` |
| Double-blind repository is possible | `NOT_STARTED` | Prepare anonymous copy after conference route is selected | `P1` |

## 8. Paper Writing Plan

| Section | Objective | Input Materials | Status | Completion Criteria | Review Status |
|---|---|---|---|---|---|
| Title | State low-cost webcam gaze interaction contribution | Route decision | `NOT_STARTED` | No overclaiming algorithm novelty | `NOT_STARTED` |
| Abstract | Summarize problem, system, study, results, limits | DONE experiments only | `BLOCKED` | Contains no unverified numbers | `NOT_STARTED` |
| Introduction | Motivate hands-free low-cost gaze interaction | Related work and RQs | `NOT_STARTED` | Ends with evidence-backed contributions | `NOT_STARTED` |
| Related Work | Position against gaze interaction, low-cost eye tracking, dwell selection | Literature review | `NOT_STARTED` | No fabricated citations | `NOT_STARTED` |
| System / Method | Explain pipeline and interaction design | Code, diagrams, parameters | `NOT_STARTED` | Matches implementation | `NOT_STARTED` |
| Experimental Setup | Define protocol and metrics | E2-E6 | `BLOCKED` | Reproducible study design | `NOT_STARTED` |
| Results | Report quantitative results | Analysis scripts | `BLOCKED` | Values trace to raw data | `NOT_STARTED` |
| Ablation | Report calibration/smoothing/dwell comparisons | E4-E6 | `BLOCKED` | Claims tied to tables/figures | `NOT_STARTED` |
| Discussion | Explain tradeoffs and failures | Results and interview notes | `BLOCKED` | Includes limitations honestly | `NOT_STARTED` |
| Limitations | State webcam, lighting, participant, task limits | Failure analysis | `NOT_STARTED` | No defensive wording | `NOT_STARTED` |
| Ethics Statement | Explain participant handling and data privacy | Study protocol | `BLOCKED` | Matches conference policy | `NOT_STARTED` |
| Conclusion | Close with supported claims only | Final results | `BLOCKED` | No new unsupported claim | `NOT_STARTED` |
| References | Provide real citations | Literature database | `NOT_STARTED` | Every citation exists and is relevant | `NOT_STARTED` |
| Appendix / Supplement | Add protocol, extra figures, demo details | Final artifacts | `NOT_STARTED` | Matches page/supplement policy | `NOT_STARTED` |

## 9. Review Plan

| Review Type | Reviewer Role | Checklist | Status |
|---|---|---|---|
| Technical correctness | Developer familiar with OpenCV/MediaPipe | API behavior, coordinate mapping, calibration, failure handling | `NOT_STARTED` |
| Experimental completeness | Research collaborator | Baselines, conditions, metrics, counterbalancing, sample exclusions | `NOT_STARTED` |
| Paper narrative | HCI / IUI reader | Research question clarity, contribution fit, no overclaiming | `NOT_STARTED` |
| Figure and table consistency | Independent checker | Values match CSV, axes labeled, captions accurate | `NOT_STARTED` |
| Citation completeness | Paper author | Claims are cited, references are real and formatted | `NOT_STARTED` |
| Language and formatting | Native/experienced reviewer | Clarity, grammar, template compliance | `NOT_STARTED` |
| Double-blind | Submission checker | Author names, GitHub links, paths, acknowledgements removed | `NOT_STARTED` |
| Reproducibility | External-style tester | Clean setup, commands, example data, expected outputs | `NOT_STARTED` |
| Final submission | Lead author | PDF, supplement, forms, conflicts, ethics, license | `NOT_STARTED` |

## 10. Submission Checklist

| Item | Status | Notes |
|---|---|---|
| Official template version confirmed | `To Be Confirmed` | Check after target conference is selected |
| Page count confirmed | `To Be Confirmed` | Do not assume page limits |
| PDF font embedding checked | `NOT_STARTED` | Requires final PDF |
| Figures are readable | `NOT_STARTED` | Requires final figures |
| Tables match result files | `NOT_STARTED` | Requires analysis scripts |
| Anonymous repository prepared | `NOT_STARTED` | Required if double-blind route demands it |
| Author information removed for blind review | `NOT_STARTED` | Depends on conference policy |
| Code/data statement written | `NOT_STARTED` | Must match actual release plan |
| Ethics statement written | `NOT_STARTED` | Must match participant study |
| Conflict of Interest completed | `NOT_STARTED` | Submission system step |
| Supplementary material checked | `NOT_STARTED` | Video, appendix, code archive |
| Final PDF hash/version recorded | `NOT_STARTED` | Record after final export |
| Submission archive stored | `NOT_STARTED` | Save exact submitted files |

## 11. Risk Register

| Risk | Probability | Impact | Trigger Condition | Mitigation | Contingency Plan | Status |
|---|---|---|---|---|---|---|
| Core gaze selection is not stable enough | Medium | High | Pilot shows high false clicks or low success rate | Add confidence gating, larger targets, clearer feedback | Reframe as usability limits or demo/toolkit | `OPEN` |
| Calibration comparison cannot be implemented in time | Medium | High | 9-point calibration remains unavailable | Keep 5-point validation and smoothing ablation | Submit demo/poster instead of full paper | `OPEN` |
| Optional dependencies are missing from reproducible setup | High | Medium | Clean environment cannot run calibrated mode | Document or add `scikit-learn` and `pyautogui` | Restrict paper claims to non-calibrated mode until fixed | `OPEN` |
| No formal participant data before deadline | Medium | High | Recruitment or IRB/ethics process delayed | Run pilot early and lock smaller protocol | Submit demo/poster or workshop-style artifact | `OPEN` |
| Results do not support strong performance claims | Medium | High | Gaze performs much worse than mouse baseline | Report tradeoffs honestly | Change contribution to characterization and design implications | `OPEN` |
| Double-blind leakage | Medium | High | README, code paths, GitHub URL, video, metadata reveal authors | Prepare anonymous repo and metadata checklist | Delay submission until blind package is clean | `OPEN` |
| Figures/tables diverge from raw data | Medium | High | Manual edits or stale plots | Generate all figures from scripts | Remove unsupported figure/table | `OPEN` |
| Conference requirements change | Medium | Medium | Official CFP differs from assumptions | Re-check official CFP before writing final checklist | Adjust target or submission type | `OPEN` |
| Participant privacy exposure | Low | High | Raw videos, names, or personal notes enter repo | Keep raw personal data outside public repo; anonymize logs | Exclude data release or release synthetic/example data only | `OPEN` |
| IEEE VR route overclaims VR relevance | Medium | Medium | No actual 3D/VR task exists | Keep IEEE VR as deferred route | Use IUI/CHI route | `OPEN` |

## 12. Timeline and Critical Path

No confirmed official deadline is recorded in the repository. Use `T` as the official submission deadline once selected.

| Relative Time | Critical Tasks | Exit Criteria |
|---|---|---|
| `T-6 weeks` | Confirm target venue, paper type, template, double-blind policy; freeze RQs and claims | Submission target table complete |
| `T-5 weeks` | Implement or validate experiment UI, logging, dependency setup, calibration validation | E0-E3 runnable |
| `T-4 weeks` | Pilot with 2-3 users; repair logging and task difficulty | Pilot report and revised protocol |
| `T-3 weeks` | Run formal experiment or demo evaluation | Raw logs saved and backed up |
| `T-2 weeks` | Clean data, generate figures/tables, write method/setup/results | Analysis scripts produce paper figures |
| `T-1 week` | Complete full draft, internal reviews, double-blind cleanup | Draft ready for final edits |
| `T-3 days` | Final PDF build, supplement check, submission form dry run | All checklist items reviewed |
| `T-0` | Submit and archive exact files | Submission ID and archive recorded |

Critical path for IUI-style submission:

```text
target venue confirmation
-> experiment UI and logger
-> calibration / dwell validation
-> pilot
-> formal data collection
-> analysis scripts
-> paper results and discussion
-> blind package and final PDF
```

## 13. Post-Submission Plan

| Phase | Tasks | Status |
|---|---|---|
| Submission confirmation | Record submission ID, timestamp, submitted files, PDF hash, supplement hash | `NOT_STARTED` |
| Review tracking | Track notification date, reviewer scores, confidence, major concerns | `NOT_STARTED` |
| Rebuttal preparation | Classify comments as misunderstanding, missing evidence, additional experiment, writing issue, limitation | `NOT_STARTED` |
| Evidence mapping | Link every rebuttal point to code, logs, figures, tables, or citations | `NOT_STARTED` |
| Additional experiments | Prioritize only experiments that answer reviewer concerns | `NOT_STARTED` |
| Revision | Update paper, appendix, code statement, figures, and limitations | `NOT_STARTED` |
| Camera-ready | Restore author info, acknowledgements, artifact links, final formatting | `NOT_STARTED` |
| Artifact evaluation | Prepare reproducible package if applicable | `NOT_STARTED` |
| Release | Publish code, anonymized data, models or demo after acceptance if policy permits | `NOT_STARTED` |
| Archive | Store final paper, source, data, logs, review response, and release tag | `NOT_STARTED` |

## 14. Immediate Action List

| Priority | Task | Owner | Status | Definition of Done |
|---|---|---|---|---|
| `P0` | Decide primary submission route and paper type | TODO | `NOT_STARTED` | Target table has official CFP facts |
| `P0` | Run a clean environment smoke test | TODO | `NEEDS_REVIEW` | Commands and outputs recorded |
| `P0` | Resolve calibrated-mode dependencies | TODO | `DONE` | `scikit-learn` and `pyautogui` are in `requirements.txt` |
| `P0` | Add trial-level experiment logger | TODO | `DONE` | CSV contains required schema |
| `P0` | Build target-selection experiment UI | TODO | `NEEDS_REVIEW` | Mouse/gaze modes, full-screen coordinate mapping, dwell feedback, and trial logs are implemented; camera pilot remains |
| `P1` | Add 9-point calibration mode or defer claim | TODO | `DONE` | 5/9/13-point calibration modes are available |
| `P1` | Add analysis scripts | TODO | `IN_PROGRESS` | Factor-aware trial and gaze-quality summary exists; figure generation remains |
| `P1` | Draft system/method section | TODO | `NOT_STARTED` | Text matches actual implementation |
| `P2` | Add subjective questionnaire and interview notes | TODO | `NOT_STARTED` | SUS/NASA-TLX/fatigue fields are defined |
| `P3` | Explore 3D UI demo for IEEE VR route | TODO | `DEFERRED` | Demo scope is decided after IUI/CHI route |

## 15. Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-12 | Reworked the document from conference-route advice into a general executable submission plan with project snapshot, status system, claim-evidence matrix, experiment backlog, reproducibility plan, review workflow, submission checklist, risk register, timeline, and post-submission workflow. Corrected the unverified calibration offset bug claim. | Codex |
| 2026-07-12 | Implemented core engineering optimizations: completed calibrated-mode dependencies, added confidence fields, 5/9/13-point calibration, calibration reports, click cooldown, screen clamping, trial logger, mouse target-selection baseline, analysis summary script, and updated README/setup commands. | Codex |
| 2026-07-12 | Added full-screen gaze target selection, fixed/confidence-aware dwell conditions, multi-sample full-screen calibration, reproducible session configs, experiment protocol, gaze-quality analysis, and hardware-independent tests. | Codex |
