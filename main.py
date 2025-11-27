import cv2
import cvzone
import math
from ultralytics import YOLO
import os

#دا مش احسن حاجه عدلت عليه كتير شوف fkra2 افضل في الاداء 


#source = 'img.jpeg'  
#source = 'vid1.mp4'  
#source = 'vid2.mp4'  
#source = 'vid3.mp4'  
source = 'vid4.mp4'  

model = YOLO("yolov8n.pt")


classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()


is_image = os.path.splitext(source)[1].lower() in ['.jpg', '.jpeg', '.png']

if is_image:
    frame = cv2.imread(source)
    frame = cv2.resize(frame, (980, 740))
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = int(box.cls[0])
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6  , colorR=(255,0,0))
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2 , colorR=(0,255,0))
            
            if threshold < 0:
                cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2 , colorR=(0,0,255))

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    cap = cv2.VideoCapture(source)
    
  
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_detect = int(box.cls[0])
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                height = y2 - y1
                width = x2 - x1
                threshold = height - width

                if conf > 80 and class_detect == 'person':
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6 ,colorR=(255,0,0))
                    cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2 , colorR=(0,255,0))

                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2  ,colorR=(0,0,255))

        cv2.imshow('frame', frame)

        if cv2.waitKey(delay) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()
