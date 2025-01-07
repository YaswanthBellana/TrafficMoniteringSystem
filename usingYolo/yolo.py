import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import yaml
from PIL import Image
from ultralytics import YOLO
from IPython.display import Video

best_model = YOLO('yolov8x.pt')
heavy_traffic_threshold = 10

vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

x1, x2 = 325, 635 
lane_threshold = 609

text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
background_color = (0, 0, 255)

cap = cv2.VideoCapture('sample_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('traffic_density_analysis.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detection_frame = frame.copy()
    detection_frame[:x1, :] = 0
    detection_frame[x2:, :] = 0
    results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
    
    processed_frame = results[0].plot(line_width=1)
    processed_frame[:x1, :] = frame[:x1, :].copy()
    processed_frame[x2:, :] = frame[x2:, :].copy()        
    
    # Draw the quadrilaterals on the processed frame
    cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Retrieve the bounding boxes from the results
    bounding_boxes = results[0].boxes
 
    # Initialize counters for vehicles in each lane
    vehicles_in_left_lane = 0
    vehicles_in_right_lane = 0

    for box in bounding_boxes.xyxy:
        if box[0] < lane_threshold:
            vehicles_in_left_lane += 1
        else:
            vehicles_in_right_lane += 1
          
    traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
    traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

    cv2.rectangle(processed_frame, (text_position_left_lane[0]-10, text_position_left_lane[1] - 25), (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)
    cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', text_position_left_lane, font, font_scale, font_color, 2, cv2.LINE_AA)
    cv2.rectangle(processed_frame, (intensity_position_left_lane[0]-10, intensity_position_left_lane[1] - 25), (intensity_position_left_lane[0] + 460, intensity_position_left_lane[1] + 10), background_color, -1)
    cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane, font, font_scale, font_color, 2, cv2.LINE_AA)

    cv2.rectangle(processed_frame, (text_position_right_lane[0]-10, text_position_right_lane[1] - 25), (text_position_right_lane[0] + 460, text_position_right_lane[1] + 10), background_color, -1)
    cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', text_position_right_lane, font, font_scale, font_color, 2, cv2.LINE_AA)
    cv2.rectangle(processed_frame, (intensity_position_right_lane[0]-10, intensity_position_right_lane[1] - 25), (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color, -1)
    cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane, font, font_scale, font_color, 2, cv2.LINE_AA)
    
    out.write(processed_frame)

cap.release()
out.release()

