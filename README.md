# Object Detection Models - README

Four object detection models: YOLOv5, YOLOv7, YOLOv8, and a Custom Fine-Tuned Model. Each model is optimized for different use cases and comes with a specific implementation to support real-time RTSP stream processing.

## 1. YOLOv5

### Overview:
YOLOv5 is a fast and lightweight object detection model, suitable for real-time applications.

### Requirements:
- Python 3.8+
- PyTorch 1.7+
- OpenCV

### Installation:
```bash
pip install torch torchvision opencv-python numpy
```

### Setup:
1. Clone the YOLOv5 repository:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained weights:
   ```bash
   python detect.py --weights yolov5s.pt --source 0
   ```

### Run with RTSP Stream:
Replace the source with your RTSP URL:
```bash
python detect.py --weights yolov5s.pt --source "rtsp://<your_rtsp_url>"
```

---

## 2. YOLOv7

### Overview:
YOLOv7 achieves state-of-the-art accuracy and is optimized for custom datasets.

### Requirements:
- Python 3.8+
- PyTorch 1.8+
- OpenCV

### Installation:
1. Clone the YOLOv7 repository:
   ```bash
   git clone https://github.com/WongKinYiu/yolov7.git
   cd yolov7
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pre-trained weights (e.g., `yolov7.pt`).

### Run with RTSP Stream:
Modify the script provided in `detect.py` to use your RTSP URL:
```bash
python detect.py --weights yolov7.pt --source "rtsp://<your_rtsp_url>"
```

---

## 3. YOLOv8

### Overview:
YOLOv8 is the latest iteration from the YOLO family, designed for better speed and precision.

### Requirements:
- Python 3.8+
- ultralytics library

### Installation:
1. Install the ultralytics package:
   ```bash
   pip install ultralytics
   ```
2. Verify installation:
   ```bash
   yolo task=detect mode=predict model=yolov8n.pt source=0
   ```

### Run with RTSP Stream:
Replace the source with your RTSP URL in the inference script:
```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("rtsp://<your_rtsp_url>")
# Add processing logic
```

---

## 4. Custom Fine-Tuned Model

### Overview:
This model is tailored to specific datasets and applications, offering maximum precision.

### Requirements:
- Python 3.8+
- PyTorch
- OpenCV

### Training:
1. Prepare your dataset in YOLO format (images and labels).
2. Use any YOLO version (e.g., YOLOv7 or YOLOv8) for fine-tuning.
3. Train the model using the appropriate training script.

### Example Command:
For YOLOv8:
```bash
yolo task=detect mode=train model=yolov8n.pt data=custom_dataset.yaml epochs=50
```

### Run with RTSP Stream:
After training, replace the model weights and RTSP URL in the inference script:
```python
model = YOLO("custom_model.pt")
cap = cv2.VideoCapture("rtsp://<your_rtsp_url>")
# Add processing logic
```

---

## Troubleshooting

- **RTSP Connection Issues**:
  - Ensure the RTSP URL is correct.
  - Check if the camera is accessible from your network.

- **Performance**:
  - Use a GPU for faster inference.
  - Reduce input size to improve speed.

- **Custom Model Accuracy**:
  - Ensure the dataset is well-labeled.
  - Perform data augmentation during training.

---

## Conclusion
Each of these models is tailored for specific use cases. For general tasks, YOLOv5 and YOLOv8 are fast and efficient. For higher precision or specific datasets, consider YOLOv7 or a custom fine-tuned model.

