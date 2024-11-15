import cv2
import json
from ObjectCounter import ObjectCounter
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

file_path = "regions.json"
with open(file_path, "r") as f:
    data = json.load(f)
region1 = data.get("region1", [])
region2 = data.get("region2", [])
region3 = data.get("region3", [])

cap = cv2.VideoCapture("night_rain.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = region1

# Init Object Counter
counter = ObjectCounter(
    count_mode="out",
    show=True,
    region=region_points,
    model="best_combined.pt",
    device="mps",
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # im0 = counter.count(im0)
    im0 = counter.count(im0, fps, 0)
    print(counter.total_count())

cap.release()
cv2.destroyAllWindows()