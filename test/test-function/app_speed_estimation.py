# app_speed_estimator.py
import cv2
import json
import speed_estimation as solutions  # Update the import as needed
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

file_path = "regions.json"
with open(file_path, "r") as f:
    data = json.load(f)
region1 = data.get("region3", [])

cap = cv2.VideoCapture("/Users/phatvu/Downloads/Untitled.mov")
assert cap.isOpened(), "Error reading video file"
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

speed_region = region1

speed_counter = solutions.SpeedEstimator(
    show=True,
    model="yolo11n.pt",
    region=speed_region,
    device="mps",
    classes= [3],
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        speed_counter.estimate_speed(im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Video frame is empty or video processing has been successfully completed.")
        break

cap.release()
cv2.destroyAllWindows()