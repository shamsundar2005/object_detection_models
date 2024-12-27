import cv2
from ultralytics import YOLO

model_path = "yolov8n.pt"
model = YOLO(model_path)

rtsp_url = "rtsp://admin:Hhappyy11!!@192.168.1.1:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to establish connection with the camera.")
    exit()

window_title = "YOLOv8 Nano - Real-Time Detection"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame could not be read from the camera.")
        break

    frame_height, frame_width, _ = frame.shape
    print(f"Processing frame of dimensions: {frame_width}x{frame_height}")

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow(window_title, annotated_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("Exiting the real-time detection loop.")
        break

cap.release()
cv2.destroyAllWindows()
