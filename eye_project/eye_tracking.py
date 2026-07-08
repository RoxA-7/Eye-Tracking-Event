"""
Eye Tracking — main entry point.

Select a video file or use the webcam, then track gaze in real time.
Press Q to quit; data is auto-saved to results/ on exit.
"""

import tkinter as tk
from tkinter import filedialog
from eye_tracker import EyeTracker


def select_video_source():
    """Open a file dialog for video selection; return 0 for webcam if cancelled."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="选择视频文件（取消则使用摄像头）",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")],
    )
    return path if path else 0


if __name__ == "__main__":
    source = select_video_source()
    if source == 0:
        print("使用摄像头作为输入源")
    else:
        print(f"使用视频文件：{source}")

    tracker = EyeTracker(video_source=source)
    tracker.run()
