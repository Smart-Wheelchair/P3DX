# import torch

# # Model
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# # Images
# img = "img/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# # Inference
# results = model(img)

# # Results
# results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

import cv2

# Load video
cap = cv2.VideoCapture('video.mp4')

# Get video FPS
fps = int(cap.get(cv2.CAP_PROP_FPS))

while True:
    # Read frame
    ret, frame = cap.read()
    
    if ret:
        # Get current time
        t1 = cv2.getTickCount()
        
        # Display FPS on frame
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('frame', frame)
        
        # Get time elapsed
        t2 = cv2.getTickCount()
        time_elapsed = (t2 - t1) / cv2.getTickFrequency()
        
        # Calculate delay to maintain FPS
        delay = max(1, int((1/fps - time_elapsed)*1000))
        
        # Wait for delay time or press 'q' to quit
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
