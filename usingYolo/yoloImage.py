import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO('yolov8x.pt')

image_path = 'p.webp'


results = model.predict(source=image_path, imgsz=640, conf=0.5)

sample_image = results[0].plot(line_width=2)

sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

# Display annotated image
plt.figure(figsize=(20,15))
plt.imshow(sample_image)
plt.title('Detected Objects in Sample Image by the Pre-trained YOLOv8 Model', fontsize=20)
plt.axis('off')
plt.show()
