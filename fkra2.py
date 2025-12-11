import cv2
import cvzone
import math
import os
import time
import pandas as pd
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tensorflow as tf
from tensorflow.keras.models import load_model # لاستيراد الموديل المُدرب

# ============================
# 0) إعدادات النماذج والبيانات
# ============================

source = 'vid3.mp4'  
IMG_SIZE = 128  # الحجم الذي تم تدريب موديل CNN عليه

# نموذج اكتشاف الأجسام (YOLO) لتحديد موقع الشخص
model_yolo = YOLO("yolov8n.pt")  

# نموذج تصنيف السقوط (CNN) الذي تدرب على صور السقوط/غير السقوط
try:
    # تحميل الموديل الذي تم تدريبه على بيانات السقوط/غير السقوط
    model_fall_cnn = load_model("fall_model_improved.h5")
except:
    print("FATAL ERROR: Could not load 'fall_model_improved.h5'. Ensure the file is present.")
    exit()

classnames = [line.strip() for line in open('classes.txt')]

tracker = DeepSort(max_age=15) # مُتابع DeepSort

fall_memory = {}

# إعداد ملف Log السقوط
excel_file = "fall_log_ai.xlsx"
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=["ID", "Timestamp", "Confidence_CNN"])
    df.to_excel(excel_file, index=False)

def log_fall(tid, confidence):
    """تسجيل حالة السقوط في ملف Excel."""
    df = pd.read_excel(excel_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat([df, pd.DataFrame([{"ID": tid, "Timestamp": timestamp, "Confidence_CNN": confidence}])], ignore_index=True)
    df.to_excel(excel_file, index=False)

# ====================================================
# دالة اكتشاف السقوط الجديدة باستخدام نموذج CNN (التعلم الآلي)
# ====================================================

def check_fall_cnn(person_crop):
    """
    تقييم حالة السقوط باستخدام نموذج CNN المُدرب.
    
    :param person_crop: جزء الصورة الذي يمثل الشخص (مصفوفة NumPy).
    :return: (is_fall_detected, fall_confidence)
    """
    # 1. تغيير حجم الجزء المقصوص ليناسب حجم إدخال الموديل (128x128)
    img = cv2.resize(person_crop, (IMG_SIZE, IMG_SIZE))
    
    # 2. تطبيع قيم البكسل (Normalization)
    img = img / 255.0
    
    # 3. إضافة بُعد الدفعة (Batch dimension)
    img = np.expand_dims(img, axis=0)
    
    # 4. التنبؤ باستخدام الموديل المُدرب
    pred = model_fall_cnn.predict(img, verbose=0)[0]
    
    # التصنيف 1 هو "FALL"، التصنيف 0 هو "NO FALL"
    fall_confidence = pred[1] 
    
    # نعتبر السقوط مكتشفًا إذا كانت الثقة أعلى من عتبة (Threshold) معينة
    fall_threshold = 0.8  
    is_fall_detected = fall_confidence >= fall_threshold
    
    return is_fall_detected, fall_confidence

def update_fall_animation(tid, detected):
    """تحديث مؤشر الرسوم المتحركة واللون للمربع."""
    mem = fall_memory[tid]
    if detected:
        mem['fall_color'] = (0, 255, 255) # لون تنبيه (أصفر)
        mem['animation_counter'] = 5
    elif mem['animation_counter'] > 0:
        mem['animation_counter'] -= 1
    else:
        mem['fall_color'] = (0, 0, 255) # لون عادي (أزرق)

def process_frame(frame):
    """العملية الرئيسية لكل إطار فيديو."""
    frame = cv2.resize(frame, (640, 480))
    results = model_yolo(frame, verbose=False) # استخدام YOLO لاكتشاف الأشخاص

    detections = []

    for info in results:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = classnames[cls]
            # التأكد من أنه شخص والثقة عالية
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
        
        # التأكد من أن المربع صالح
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue
        
        # قص جزء الصورة للشخص (Crop)
        person_crop = frame[y1:y2, x1:x2]

        if tid not in fall_memory:
            fall_memory[tid] = {'fall_frames': 0, 'fall_color': (0,0,255), 'animation_counter': 0, 'last_conf': 0.0}

        mem = fall_memory[tid]
        
        # =======================================================
        # استبدال منطق check_fall(width, height) بـ check_fall_cnn
        # =======================================================
        
        fall_detected_cnn, fall_conf = check_fall_cnn(person_crop)
        mem['last_conf'] = fall_conf
        
        if fall_detected_cnn:
            mem['fall_frames'] += 1
        else:
            # تقليل عداد الإطارات تدريجياً
            mem['fall_frames'] = max(0, mem['fall_frames'] - 1)

        # الكشف الحقيقي للسقوط (يجب أن يستمر لعدة إطارات لمنع الإنذارات الكاذبة)
        fall_detected = mem['fall_frames'] > 4 

        update_fall_animation(tid, fall_detected)

        any_fall = any_fall or fall_detected

        # رسم المربع حول الشخص
        rect_color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cvzone.cornerRect(frame, [x1, y1, width, height], l=20, rt=6, colorR=rect_color)
        
        # كتابة التصنيف وثقة الموديل
        text_label = f"ID:{tid} | {'FALL' if fall_detected else 'No Fall'} ({mem['last_conf']:.2f})"
        cv2.putText(frame, text_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)

        if fall_detected and mem['fall_frames'] == 5: # سجل السقوط مرة واحدة عند الكشف الأولي
            log_fall(tid, mem['last_conf'])

    # وضع مربع إجمالي للحالة العامة
    big_box_color = (0, 0, 255) if any_fall else (0, 255, 0)
    cv2.rectangle(frame, (10,10), (200,70), big_box_color, -1)
    text = "FALL" if any_fall else "No Fall"
    cv2.putText(frame, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return frame

# ============================
# 7) تشغيل الفيديو (بدون تغيير)
# ============================

is_image = os.path.splitext(source)[1].lower() in ['.jpg', '.jpeg', '.png']

if is_image:
    frame = cv2.imread(source)
    frame = process_frame(frame)
    cv2.imshow("Fall Detection (AI)", frame)
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
        cv2.imshow("Fall Detection (AI)", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
