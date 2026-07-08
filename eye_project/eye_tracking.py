import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import time
import mediapipe as mp
import pandas as pd
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import os
import platform
import subprocess

def select_video_source():
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口
    file_path = filedialog.askopenfilename(
        title="选择视频文件（取消则使用摄像头）",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
    )
    return file_path if file_path else 0

video_source = select_video_source()
if video_source == 0:
    print("使用摄像头作为输入源")
else:
    print(f"使用视频文件：{video_source}")

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("无法打开摄像头或视频文件，程序退出。")
    exit()


# 初始化组件
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

# 配置
gaze_buffer = deque(maxlen=5)
heatmap_layer = None
fixations = []
fixation_threshold = 15
min_fixation_duration = 0.5
fixation_start_time = None
fixation_reference = None

data_records = []
frame_count = 0
blink_count = 0
fatigue_alert_triggered = False
start_time = time.time()

def extract_eye_roi(image, landmarks, eye_points, margin=10):
    h, w, _ = image.shape
    x1 = int(landmarks[eye_points[0]].x * w)
    y1 = int(landmarks[eye_points[0]].y * h)
    x2 = int(landmarks[eye_points[1]].x * w)
    y2 = int(landmarks[eye_points[1]].y * h)
    x_min = max(min(x1, x2) - margin, 0)
    y_min = max(min(y1, y2) - margin, 0)
    x_max = min(max(x1, x2) + margin, w)
    y_max = min(max(y1, y2) + margin, h)
    roi = image[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min)

def process_eye(roi, draw_frame, offset, label="Threshold"):
    if roi.size == 0:
        return None
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (7, 7), 0)
    roi_gray = cv2.equalizeHist(roi_gray)
    threshold = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow(label, threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if not contours:
        return None
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        offset_x, offset_y = offset
        center = (offset_x + cx, offset_y + cy)
        cv2.rectangle(draw_frame, (offset_x + x, offset_y + y),
                      (offset_x + x + w, offset_y + y + h), (255, 0, 0), 2)
        cv2.line(draw_frame, (center[0], offset_y + y), (center[0], offset_y + y + h), (0, 255, 0), 1)
        cv2.line(draw_frame, (offset_x + x, center[1]), (offset_x + x + w, center[1]), (0, 255, 0), 1)
        return center
    return None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if heatmap_layer is None:
            heatmap_layer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        left_coords = None
        right_coords = None
        gaze_x = None
        gaze_y = None
        blink = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            eye_roi_l, pt_l = extract_eye_roi(frame, landmarks, LEFT_EYE)
            eye_roi_r, pt_r = extract_eye_roi(frame, landmarks, RIGHT_EYE)

            left_coords = process_eye(eye_roi_l, frame, pt_l, label="Left Eye Threshold")
            right_coords = process_eye(eye_roi_r, frame, pt_r, label="Right Eye Threshold")

            if left_coords and right_coords:
                raw_x = (left_coords[0] + right_coords[0]) // 2
                raw_y = (left_coords[1] + right_coords[1]) // 2
                gaze_buffer.append((raw_x, raw_y))
                gaze_x = int(sum(x for x, y in gaze_buffer) / len(gaze_buffer))
                gaze_y = int(sum(y for x, y in gaze_buffer) / len(gaze_buffer))

                cv2.circle(heatmap_layer, (gaze_x, gaze_y), 3, 1, -1)
                heatmap_vis = cv2.applyColorMap(cv2.convertScaleAbs(heatmap_layer, alpha=10), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 0.7, heatmap_vis, 0.3, 0)

                cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
                cv2.line(frame, (gaze_x, 0), (gaze_x, frame.shape[0]), (0, 255, 0), 1)
                cv2.line(frame, (0, gaze_y), (frame.shape[1], gaze_y), (0, 255, 0), 1)

                if fixation_reference is None:
                    fixation_reference = (gaze_x, gaze_y)
                    fixation_start_time = time.time()
                else:
                    dx = abs(gaze_x - fixation_reference[0])
                    dy = abs(gaze_y - fixation_reference[1])
                    if dx <= fixation_threshold and dy <= fixation_threshold:
                        duration = time.time() - fixation_start_time
                        if duration >= min_fixation_duration:
                            fixations.append({
                                "start_time": fixation_start_time,
                                "duration_sec": duration,
                                "gaze_x": gaze_x,
                                "gaze_y": gaze_y
                            })
                            fixation_start_time = time.time()
                            fixation_reference = (gaze_x, gaze_y)
                    else:
                        fixation_start_time = time.time()
                        fixation_reference = (gaze_x, gaze_y)

        if not left_coords or not right_coords:
            blink = 1
            blink_count += 1

        data_records.append({
            "timestamp": timestamp,
            "left_x": left_coords[0] if left_coords else None,
            "left_y": left_coords[1] if left_coords else None,
            "right_x": right_coords[0] if right_coords else None,
            "right_y": right_coords[1] if right_coords else None,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "blink": blink
        })

        elapsed = time.time() - start_time
        if elapsed > 10:
            blink_rate = blink_count / elapsed
            if blink_rate > 0.3 and not fatigue_alert_triggered:
                print("疲劳警告：眨眼频率偏高，请休息")
                fatigue_alert_triggered = True

        window_title = f"Eye Tracking - {'摄像头' if video_source == 0 else video_source}"
        cv2.imshow(window_title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"当前 FPS: {fps:.2f}")

except KeyboardInterrupt:
    print("\n检测到中断信号，正在保存数据...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    data_file = os.path.join(save_dir, f"eye_tracking_data_{timestamp_str}.csv")
    fix_file = os.path.join(save_dir, f"fixations_{timestamp_str}.csv")
    heatmap_file = os.path.join(save_dir, f"gaze_heatmap_{timestamp_str}.png")

    df = pd.DataFrame(data_records)
    df.to_csv(data_file, index=False)
    fix_df = pd.DataFrame(fixations)
    fix_df.to_csv(fix_file, index=False)

    plt.figure(figsize=(10, 6))
    heatmap_display = cv2.convertScaleAbs(heatmap_layer, alpha=255.0 / np.max(heatmap_layer))
    plt.imshow(heatmap_display, cmap='jet')
    plt.title("Gaze Heatmap")
    plt.axis('off')
    plt.savefig(heatmap_file)
    plt.close()

    print(f"数据保存完成：{data_file} + {fix_file} + {heatmap_file}")

    if platform.system() == "Windows":
        subprocess.Popen(f'explorer {os.path.abspath(save_dir)}')
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", os.path.abspath(save_dir)])
    elif platform.system() == "Linux":
        subprocess.Popen(["xdg-open", os.path.abspath(save_dir)])
