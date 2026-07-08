# Eye Tracking Project

基于 OpenCV + MediaPipe 的瞳孔追踪与注视点分析系统。

## 功能

- **瞳孔检测**：使用 MediaPipe Face Mesh 定位眼部区域，自适应阈值检测瞳孔中心
- **注视点追踪**：实时计算双眼瞳孔中心 → 画面注视坐标，叠加十字线 + 热力图（带衰减防饱和）
- **注视分析 (Fixation)**：I-VT 算法 + 3 帧抖动容错，准确记录注视起止时间和位置
- **眨眼检测**：滞后滤波（连续 3 帧瞳孔丢失才算一次眨眼），疲劳告警
- **数据导出**：自动保存 CSV 数据、注视记录和热力图 PNG；支持增量存盘防崩溃
- **校准 + 鼠标控制** (`eye_tracking-test.py`)：5 点校准 → 线性回归模型 → PyAutoGUI 屏幕映射 + 注视 3 秒自动点击
- **性能优化**：可配置帧跳过（MediaPipe 隔帧运行）、调试窗口可选关闭

## 项目结构

```
eye_project/
├── eye_tracker.py           # 核心引擎（EyeTracker + CalibratedEyeTracker 类）
├── eye_tracking.py          # 主入口（视频文件 / 摄像头）
├── eye_tracking-test.py     # 校准版入口（PyAutoGUI 鼠标控制）
├── requirements.txt         # Python 依赖
├── setup_env.ps1            # PowerShell 一键环境配置脚本
├── eye_tracking_data.csv    # 输出：追踪数据
├── fixations.csv            # 输出：注视分析数据
└── gaze_heatmap*.png        # 输出：注视热力图
```

## 快速开始

### 1. 创建虚拟环境

```powershell
cd eye_project
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> 若遇到权限错误，先运行：`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

或直接运行一键脚本：

```powershell
.\setup_env.ps1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

`eye_tracking-test.py` 额外需要：

```bash
pip install scikit-learn pyautogui
```

### 3. 运行

**主程序**（有 UI 文件选择对话框，支持视频文件或摄像头）：

```bash
python eye_tracking.py
```

**校准 + 鼠标控制版**（注视 3 秒自动点击）：

```bash
python eye_tracking-test.py
```

按 `Q` 键退出，数据自动保存到 `results/`。

### 4. 高级用法

```python
from eye_tracker import EyeTracker

# 每 2 帧跑一次 MediaPipe（提升性能）
tracker = EyeTracker(video_source=0, process_every_n_frames=2)

# 开启调试窗口
tracker = EyeTracker(show_debug_windows=True)

# 每 150 帧增量存盘（防崩溃丢数据）
tracker = EyeTracker(flush_interval=150)

tracker.run()
```

## 依赖

| 包 | 用途 |
|---|---|
| opencv-python | 视频处理、图像处理、可视化 |
| mediapipe | 人脸网格 → 眼部关键点 |
| pandas | 数据记录与导出 |
| matplotlib | 热力图渲染 |
| scikit-learn (可选) | 校准阶段线性回归 |
| pyautogui (可选) | 屏幕鼠标映射 |

## License

MIT
