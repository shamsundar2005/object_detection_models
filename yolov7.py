import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "yolov7.pt"  
model = torch.load(model_path, map_location=device)["model"]
model.float().eval().to(device) 

with open("data/coco.names", "r") as f: 
    class_names = f.read().splitlines()

rtsp_url = "rtsp://admin:Hhappyy11!!@192.168.1.1:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not connect to the camera.")
    exit()

img_size = 640 
transform = transforms.ToTensor()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    img = letterbox(frame, new_shape=(img_size, img_size), auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)  # Optimize array memory layout
    img_tensor = transform(img).unsqueeze(0).to(device).float() / 255.0  # Normalize to [0, 1]

    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]

    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{class_names[int(cls)]} {conf:.2f}"
                color = (0, 255, 0)  
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    cv2.imshow("YOLOv7 - RTSP Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
