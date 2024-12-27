import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt") 

rtsp_url = "rtsp://admin:Hhappyy11!!@192.168.1.1:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not connect to the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model.predict(frame, conf=0.6, iou=0.5)  # Set confidence and IoU thresholds

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  
        confs = result.boxes.conf.cpu().numpy()  
        classes = result.boxes.cls.cpu().numpy()  

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLOv8 - RTSP Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
