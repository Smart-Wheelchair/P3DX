import cv2
import numpy as np
import torch
from motpy import Detection, MultiObjectTracker

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# create a multi object tracker with a specified step time of 100ms
tracker = MultiObjectTracker(dt=0.1)

# Open video file or capture device
cap = cv2.VideoCapture('/home/luke/Desktop/uid_vid_00005.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    t1 = cv2.getTickCount()
    # Perform object detection with YOLOv5
    results = model(frame)

    # Convert YOLOv5 detections to motpy detections
    detections = []
    for box in results.xyxy[0]:
        box = box.cpu().numpy()
        detections.append(Detection(box=[box[0], box[1], box[2]-box[0], box[3]-box[1]], score=box[4]))

    # Update MOT tracker with detections
    tracker.step(detections=detections)

    # Retrieve active tracks from tracker
    tracks = tracker.active_tracks()
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Draw bounding boxes and IDs on frame
    for track in tracks:
        bbox = track.box.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, str(track.id), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Get time elapsed
    t2 = cv2.getTickCount() 
    time_elapsed = (t2 - t1) / cv2.getTickFrequency()


    # Show frame
    cv2.imshow('frame', cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2)))
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
