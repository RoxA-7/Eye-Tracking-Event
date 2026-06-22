# Eye Tracking Event 👁️

基于 MediaPipe 和 OpenCV 的实时眼动追踪系统，支持注视点热力图、注视行为分析、眨眼检测与疲劳预警。

A real-time eye tracking system based on MediaPipe and OpenCV, featuring gaze heatmap overlay, fixation analysis, blink detection, and fatigue alerts.

## ✨ 功能特性 / Features

- **👁️ 瞳孔检测 / Pupil Detection** — 使用 MediaPipe Face Mesh 定位眼部关键点，自适应阈值提取瞳孔中心
- **🎯 注视追踪 / Gaze Tracking** — 双眼瞳孔中心融合计算注视坐标，平滑滤波
- **🔥 实时热力图 / Real-time Heatmap** — 注视点叠加生成动态热力图，直观展示注意力分布
- **📍 注视行为分析 / Fixation Analysis** — 检测持续注视（Fixation），记录时长与位置
- **😴 疲劳检测 / Fatigue Detection** — 基于眨眼频率的疲劳预警（>0.3次/秒触发警报）
- **📊 数据导出 / Data Export** — 每条记录输出为 CSV（注视数据 + 注视行为），热力图导出为 PNG
- **🎥 多输入源 / Multi-source Input** — 支持摄像头实时采集或视频文件分析

## 🚀 快速开始 / Quick Start

### 环境要求 / Requirements

- Python 3.8+
- 摄像头（实时模式）或视频文件

### 一键安装 / One-click Setup

`powershell
# Windows PowerShell
.\setup_env.ps1
`

### 手动安装 / Manual Setup

`ash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
`

### 运行 / Run

`ash
# 主程序（支持选择视频文件或摄像头）
python eye_tracking.py

# 摄像头直连版本
python eye_tracking-saved.py

# 高级版本（含标定 + 屏幕映射 + 自动点击）
python eye_tracking-test.py
`

## 📁 项目结构 / Project Structure

`
eye_project/
├── eye_tracking.py          # 主程序：视频/摄像头选择，实时追踪，数据导出
├── eye_tracking-saved.py    # 简化版：摄像头直连，适合日常使用
├── eye_tracking-test.py     # 实验版：标定校准 + 注视点屏幕映射 + 自动点击
├── requirements.txt         # Python 依赖
├── setup_env.ps1            # Windows 一键环境配置脚本
├── .gitignore
└── results/                 # 输出结果目录
    ├── eye_tracking_data_*.csv   # 逐帧注视数据
    ├── fixations_*.csv           # 注视行为记录
    └── gaze_heatmap_*.png        # 注视热力图
`

## 🎮 操作说明 / Controls

| 按键 | 功能 |
|------|------|
| Q | 退出程序并自动保存数据 |
| Ctrl+C | 终端中断，同样触发数据保存 |

### 主程序 (eye_tracking.py)

1. 启动后弹出文件选择对话框 — 选择视频文件，或取消使用摄像头
2. 或直接在终端输入视频路径（留空使用摄像头）
3. 实时窗口展示追踪效果：
   - 红色圆点 = 当前注视点
   - 绿色十字线 = 注视坐标
   - 蓝色矩形 = 瞳孔检测框
   - 彩色热力叠加 = 历史注视分布

### 实验版 (eye_tracking-test.py)

1. **标定阶段** — 依次注视屏幕上 5 个提示点，建立注视-屏幕映射模型
2. **追踪阶段** — 视线控制鼠标移动，持续注视 3 秒触发自动点击（带圆形倒计时动画）

## 📋 输出数据格式 / Output Format

### 注视数据 CSV (eye_tracking_data_*.csv)

| 字段 | 说明 |
|------|------|
| 	imestamp | 时间戳 |
| left_x, left_y | 左眼瞳孔坐标 |
| ight_x, ight_y | 右眼瞳孔坐标 |
| gaze_x, gaze_y | 融合注视坐标 |
| link | 眨眼标记（0/1） |

### 注视行为 CSV (ixations_*.csv)

| 字段 | 说明 |
|------|------|
| start_time | 注视开始时间（Unix时间戳） |
| duration_sec | 注视持续时间（秒） |
| gaze_x, gaze_y | 注视点坐标 |

## 🔧 技术栈 / Tech Stack

- [OpenCV](https://opencv.org/) — 图像处理、瞳孔检测、可视化
- [MediaPipe](https://mediapipe.dev/) — 人脸网格关键点检测
- [Pandas](https://pandas.pydata.org/) — 数据记录与导出
- [Matplotlib](https://matplotlib.org/) — 热力图渲染
- [scikit-learn](https://scikit-learn.org/) — 标定模型（实验版）
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — 鼠标控制（实验版）

## 📝 License

MIT
