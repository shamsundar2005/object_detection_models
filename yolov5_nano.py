import cv2
import torch
from yolov5.models.common import DetectMultiBackend  
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.datasets import letterbox

model_path = "yolov5n.pt"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(model_path, device=device)

model.stride = model.stride
names = model.names
img_size = 640  

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

 
    img = letterbox(frame, img_size, stride=model.stride)[0]  
    img = img[:, :, ::-1].transpose(2, 0, 1)  
    img = torch.from_numpy(img).float() / 255.0  
    img = img.unsqueeze(0).to(device)

    pred = model(img, augment=False)
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)  

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   
    cv2.imshow("YOLOv5 Nano - IPC Camera Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
