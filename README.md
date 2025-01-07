# Traffic Monitoring System

## Overview
This repository contains the implementation of a Traffic Monitoring System. The primary objective of the project is **Object Detection** in traffic scenarios. The project is divided into two main folders:

1. **Simple Moving Object Recognition (Without Machine Learning)**
2. **Object Detection Using YOLO Algorithm**

Both implementations aim to detect and monitor objects (vehicles, pedestrians, etc.) in real-time traffic footage or video feeds.

---

## Folders Description

### 1. Simple-Moving-Object-Recognition
This folder contains the implementation of object detection using traditional image processing techniques like:
- Background subtraction using MOG2 (Mixture of Gaussians version 2)
- Contour detection
- Optical flow analysis

These techniques do not use machine learning algorithms but rely on classical computer vision methods.

### 2. YOLO-Object-Detection
This folder implements object detection using the **YOLO (You Only Look Once)** algorithm, a state-of-the-art real-time object detection system. This method leverages pre-trained YOLO models from the `yolo` module in Python to detect multiple classes of objects in traffic scenes.

---

## Setup and Installation
### Prerequisites
- Python 3.8 or above
- OpenCV
- NumPy
- PyTorch (for YOLO-based detection)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YaswanthBellana/TrafficMonitoringSystem.git
   cd TrafficMonitoringSystem
   ```

2. Navigate to the desired folder (e.g., `Simple-Moving-Object-Recognition(Basic)` or `YOLO-Object-Detection(usingYOLO)`).

3. Install the required libraries:
   ```bash
   pip install numpy pandas
   ```

---

## Usage
### Simple-Moving-Object-Recognition
1. Navigate to the `Simple-Moving-Object-Recognition` folder.
2. Place your traffic video files in the `data/` folder.
3. Run the main script:
   ```bash
   python Basic/main.py
   ```
4. Processed videos will be played automatically.

### YOLO-Object-Detection
1. Navigate to the `YOLO-Object-Detection` folder.
2. Ensure that the YOLO pre-trained weights and configuration files are in the `models/` folder.
3. Place your traffic video files in the `data/` folder.
4. Run the main script:
   ```bash
   python usingYolo/main.py
   ```
5. Processed videos will be played automatically.

---

## Results
- **Simple Moving Object Recognition**: Demonstrates basic moving object detection using traditional computer vision methods (e.g., MOG2 for background subtraction).
- **YOLO Object Detection**: Provides robust and accurate detection of multiple object classes (e.g., cars, buses, pedestrians) in traffic scenes using pre-trained YOLO models.

---

## Future Work
- Integrate tracking algorithms to monitor object movement.
- Enhance accuracy by fine-tuning YOLO models with custom datasets.
- Expand the system to include traffic analytics such as congestion detection and vehicle counting.

---

## Contributing
Contributions are welcome! If you would like to improve this project, feel free to open a pull request or submit an issue.

---

## Acknowledgements
- YOLO: [YOLO Official GitHub Repository](https://github.com/AlexeyAB/darknet)
- OpenCV: [OpenCV Official Website](https://opencv.org/)
