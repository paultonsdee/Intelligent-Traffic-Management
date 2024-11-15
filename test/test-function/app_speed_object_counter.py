import cv2
import json
import speed_object_counter as solutions
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

speed_region = region1

# Khởi tạo counter với velocity threshold
counter = solutions.VelocityObjectCounter(
    velocity_threshold=10.0,  # px/frame
    count_mode="in",
    show=True,
    region=speed_region,
    model="best_combined.pt",
    device="mps"
)

# Xử lý video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
        
    # Đếm với threshold 0.5 giây
    im0 = counter.count(im0, fps, 0.5)
    
    # Lấy kết quả đếm
    counts = counter.get_counts()
    print(f"Stationary: {counts['stationary']}")
    print(f"Moving: {counts['moving']}")
    print(f"Speeding: {counts['speeding']}")

cap.release()
cv2.destroyAllWindows()