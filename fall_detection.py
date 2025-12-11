import cv2
import cvzone
import math
import os
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


source = 'vid3.mp4'  

model = YOLO("yolov8n.pt")  

classnames = [line.strip() for line in open('classes.txt')]

tracker = DeepSort(max_age=15)

fall_memory = {}

excel_file = "fall_log.xlsx"
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=["ID", "Timestamp", "Confidence"])
    df.to_excel(excel_file, index=False)

def log_fall(tid, confidence):
    df = pd.read_excel(excel_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat([df, pd.DataFrame([{"ID": tid, "Timestamp": timestamp, "Confidence": confidence}])], ignore_index=True)
    df.to_excel(excel_file, index=False)

def check_fall(width, height):
    return (height - width) < 0

def update_fall_animation(tid, detected):
    mem = fall_memory[tid]
    if detected:
        mem['fall_color'] = (0, 255, 0)
        mem['animation_counter'] = 5
    elif mem['animation_counter'] > 0:
        mem['animation_counter'] -= 1
    else:
        mem['fall_color'] = (0, 0, 255)

def process_frame(frame):
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    detections = []

    for info in results:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = classnames[cls]
            if cls_name != 'person' or confidence < 0.5:
                continue
            width, height = x2 - x1, y2 - y1
            detections.append(([x1, y1, width, height], confidence, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)
    any_fall = False  

    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        width, height = x2 - x1, y2 - y1

        if tid not in fall_memory:
            fall_memory[tid] = {'fall_frames': 0, 'fall_color': (0,0,255), 'animation_counter': 0}

        mem = fall_memory[tid]

        aspect_fall = check_fall(width, height)
        if aspect_fall:
            mem['fall_frames'] += 1
        else:
            mem['fall_frames'] = max(0, mem['fall_frames'] - 1)

        fall_detected = mem['fall_frames'] > 4 

        update_fall_animation(tid, fall_detected)

        any_fall = any_fall or fall_detected

        cvzone.cornerRect(frame, [x1, y1, width, height], l=20, rt=6, colorR=(255,0,0))

        if fall_detected:
            log_fall(tid, round(track.det_confidence, 2) if hasattr(track, 'det_confidence') else 1.0)

    big_box_color = (0, 255, 0) if any_fall else (0, 0, 255)
    cv2.rectangle(frame, (10,10), (200,70), big_box_color, -1)
    text = "Fall Detected" if any_fall else "No Fall"
    cv2.putText(frame, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return frame


is_image = os.path.splitext(source)[1].lower() in ['.jpg', '.jpeg', '.png']

if is_image:
    frame = cv2.imread(source)
    frame = process_frame(frame)
    cv2.imshow("Fall Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()