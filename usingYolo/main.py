from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("sample_video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_points = [(310, 440), (910, 440), (1203, 630), (2, 630)]

video_writer = cv2.VideoWriter("c.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True, reg_pts=region_points, classes_names=model.names, line_thickness=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(frame, persist=True, show=False)

    frame = counter.start_counting(frame, tracks)
    video_writer.write(frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
