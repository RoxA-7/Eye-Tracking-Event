# Eye Tracking Project

基于 OpenCV + MediaPipe 的瞳孔追踪与注视点分析系统。

## 功能

- **瞳孔检测**：使用 MediaPipe Face Mesh 定位眼部区域，自适应阈值检测瞳孔中心
- **注视点追踪**：实时计算双眼瞳孔中心 → 画面注视坐标，叠加十字线 + 热力图
- **注视分析 (Fixation)**：检测持续注视行为，记录起止时间和位置
- **眨眼检测**：统计眨眼频率，超出阈值发出疲劳警告
- **数据导出**：自动保存 CSV 数据、注视记录和热力图 PNG
- **校准 + 鼠标控制** (`eye_tracking-test.py`)：5 点校准 → 线性回归模型 → PyAutoGUI 屏幕映射 + 自动点击

## 项目结构

```
eye_project/
├── eye_tracking.py          # 主程序（视频文件 / 摄像头 + 全功能）
├── eye_tracking-test.py     # 进阶版（校准 + PyAutoGUI 鼠标控制）
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

按 `Q` 键退出，数据自动保存。

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
