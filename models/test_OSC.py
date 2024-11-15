# app_speed_estimator.py
import cv2
import json
import logging
from ObjectSpeedCounter import ObjectSpeedCounter

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

file_path = "regions.json"
with open(file_path, "r") as f:
    data = json.load(f)
speed_region = data.get("region3", [])

cap = cv2.VideoCapture("/Users/phatvu/Downloads/Untitled.mov")
assert cap.isOpened(), "Error reading video file"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

t = 0.5  # Time in seconds to consider a vehicle's speed stable for counting

step_frame = 5  # Process every 5 frames

speed_counter = ObjectSpeedCounter(
    fps=fps / step_frame,
    t=t,
    mode='out',  # Set mode to 'in' or 'out' as needed
    show=True,
    model="best.pt",
    region=speed_region,
    device="mps",
    classes=[2, 3],  # Assuming class '2' corresponds to vehicles
)

frame_count = 0

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        if frame_count % step_frame == 0:
            processed_frame = speed_counter.estimate_speed(im0)
            cv2.imshow("Speed Estimation", processed_frame)
            print(f"Current count of vehicles with zero velocity: {speed_counter.count_zero}")
            print(f"Current count of vehicles with positive velocity: {speed_counter.count_positive}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Video frame is empty or video processing has been successfully completed.")
        break

cap.release()
cv2.destroyAllWindows()

# Output the total counts after processing
print(f"Total number of vehicles with zero velocity: {speed_counter.count_zero}")
print(f"Total number of vehicles with positive velocity: {speed_counter.count_positive}")