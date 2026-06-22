# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture("眼动.mp4")
#
# while (True):
#     ret, frame = cap.read()
#     if ret is False:
#         break
#     roi = frame[100: 500, 157: 800]  # 利用切片工具，选出感兴趣roi区域
#     #  cv2.imshow("show",roi)
#
#     rows, cols, _ = roi.shape  # 保存视频尺寸以备用
#     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 转灰度
#     gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)  # 高斯滤波一次
#
#     _, threshold = cv2.threshold(gray_roi, 8, 255, cv2.THRESH_BINARY_INV)  # 二值化，依据需要改变阈值
#     contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 画连通域
#     contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
#
#     for cnt in contours:
#         (x, y, w, h) = cv2.boundingRect(cnt)
#
#         # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
#         cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
#         cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)
#         break
#
#     cv2.imshow("Roi", roi)
#     cv2.imshow("Threshold", threshold)
#     key = cv2.waitKey(30)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# cv2.namedWindow("camera", 1)
# # 开启ip摄像头
# video = "rtsp://admin:admin@192.168.43.32:8554/live"  # 此处@后的ipv4 地址需要修改为自己的地址
# #！！！！划重点了！！！！这个地址就是上面记下来的局域网IP
# cap = cv2.VideoCapture(video)

#
#cap = cv2.VideoCapture(0)  # 使用默认摄像头















# --- Enhanced Eye Tracking with Fixation Visualization, Head Pose Estimation, Calibration, and Gaze-to-Screen Mapping ---
# Eye Tracking with Separate Calibration and Tracking Modes
# Eye Tracking with Calibration, PyAutoGUI Interaction, and Auto Click on Fixation (with Visual Countdown)

import cv2
import numpy as np
import time
import mediapipe as mp
import pandas as pd
from datetime import datetime
from collections import deque
from sklearn.linear_model import LinearRegression
import pyautogui
import os

# 初始化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

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

def process_eye(roi):
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        return x + w // 2, y + h // 2
    return None

def run_calibration():
    print("🟡 校准阶段开始：请依次注视提示位置")
    calibration_data = []
    calibration_points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
    calibration_index = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头无法启动")
        return None

    while calibration_index < len(calibration_points):
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            eye_roi_l, offset_l = extract_eye_roi(frame, landmarks, LEFT_EYE)
            eye_roi_r, offset_r = extract_eye_roi(frame, landmarks, RIGHT_EYE)
            rel_l = process_eye(eye_roi_l)
            rel_r = process_eye(eye_roi_r)

            if rel_l and rel_r:
                abs_l = (offset_l[0] + rel_l[0], offset_l[1] + rel_l[1])
                abs_r = (offset_r[0] + rel_r[0], offset_r[1] + rel_r[1])
                gaze_x = (abs_l[0] + abs_r[0]) // 2
                gaze_y = (abs_l[1] + abs_r[1]) // 2

                norm_pt = calibration_points[calibration_index]
                px = int(norm_pt[0] * w)
                py = int(norm_pt[1] * h)
                cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)
                calibration_data.append([gaze_x, gaze_y, norm_pt[0], norm_pt[1]])
                calibration_index += 1
                time.sleep(1)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 校准完成，开始训练模型")
    df = pd.DataFrame(calibration_data, columns=["gx", "gy", "sx", "sy"])
    model = LinearRegression().fit(df[["gx", "gy"]], df[["sx", "sy"]])
    return model

def run_tracking(gaze_model):
    screen_w, screen_h = pyautogui.size()
    gaze_buffer = deque(maxlen=5)
    fixation_reference = None
    fixation_start_time = None
    fixation_threshold = 15
    min_fixation_duration = 0.5
    auto_click_duration = 3.0  # 改为 3 秒注视点击
    fixations = []
    heatmap_layer = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头无法启动")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        if heatmap_layer is None:
            heatmap_layer = np.zeros((h, w), dtype=np.float32)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            eye_roi_l, offset_l = extract_eye_roi(frame, landmarks, LEFT_EYE)
            eye_roi_r, offset_r = extract_eye_roi(frame, landmarks, RIGHT_EYE)
            rel_l = process_eye(eye_roi_l)
            rel_r = process_eye(eye_roi_r)

            if rel_l and rel_r:
                abs_l = (offset_l[0] + rel_l[0], offset_l[1] + rel_l[1])
                abs_r = (offset_r[0] + rel_r[0], offset_r[1] + rel_r[1])
                raw_x = (abs_l[0] + abs_r[0]) // 2
                raw_y = (abs_l[1] + abs_r[1]) // 2
                gaze_buffer.append((raw_x, raw_y))
                gaze_x = int(np.mean([x for x, y in gaze_buffer]))
                gaze_y = int(np.mean([y for x, y in gaze_buffer]))

                mapped = gaze_model.predict([[gaze_x, gaze_y]])[0]
                screen_x = int(mapped[0] * screen_w)
                screen_y = int(mapped[1] * screen_h)

                pyautogui.moveTo(screen_x, screen_y, duration=0.05)

                if fixation_reference is None:
                    fixation_reference = (gaze_x, gaze_y)
                    fixation_start_time = time.time()
                else:
                    dx = abs(gaze_x - fixation_reference[0])
                    dy = abs(gaze_y - fixation_reference[1])
                    if dx <= fixation_threshold and dy <= fixation_threshold:
                        duration = time.time() - fixation_start_time
                        if duration >= auto_click_duration:
                            print("🖱 自动点击触发")
                            pyautogui.click()
                            fixation_start_time = time.time()
                        elif duration >= min_fixation_duration:
                            fixations.append({
                                "start": fixation_start_time,
                                "duration": duration,
                                "gx": gaze_x,
                                "gy": gaze_y
                            })

                        # 🟡 添加视觉提示环（倒计时圈）
                        remaining = max(0, auto_click_duration - duration)
                        progress = 1.0 - remaining / auto_click_duration
                        radius = int(40 + 10 * progress)
                        color = (0, int(255 * progress), int(255 * (1 - progress)))
                        cv2.circle(frame, (gaze_x, gaze_y), radius, color, 2)
                        cv2.putText(frame, f"{remaining:.1f}s", (gaze_x + 20, gaze_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        fixation_reference = (gaze_x, gaze_y)
                        fixation_start_time = time.time()

                cv2.putText(frame, f"Screen: ({screen_x},{screen_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
                cv2.circle(heatmap_layer, (gaze_x, gaze_y), 3, 1, -1)

        heatmap_vis = cv2.applyColorMap(cv2.convertScaleAbs(heatmap_layer, alpha=10), cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.7, heatmap_vis, 0.3, 0)

        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(fixations).to_csv("fixations.csv", index=False)
    print("✅ gaze tracking 数据保存完毕")

if __name__ == "__main__":
    model = run_calibration()
    if model:
        run_tracking(model)
