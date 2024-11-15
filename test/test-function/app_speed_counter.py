# speed_counter.py
from speed_estimation2 import SpeedEstimator
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

class SpeedCounter(SpeedEstimator):
    def __init__(
        self,
        count_mode="stationary",  # "stationary", "moving", "speeding"
        speed_threshold=30,  # km/h
        pixels_per_meter=10,  # calibration factor
        **kwargs
    ):
        super().__init__(**kwargs)
        self.count_mode = count_mode
        self.speed_threshold = speed_threshold
        self.pixels_per_meter = pixels_per_meter
        self.counted_ids = []
        self.frame_counters = {}  # {track_id: frame_count}
        self.stationary_count = 0
        self.moving_count = 0
        self.speeding_count = 0
        self.frame_threshold = 0
        self.speed_buffer = {}  # {track_id: [recent_speeds]}
        self.prev_positions = {}  # {track_id: (x,y)}
        self.prev_times = {}  # {track_id: time}

    def calculate_speed(self, track_id, current_pos, fps):
        """Calculate speed in km/h"""
        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = current_pos
            self.prev_times[track_id] = time()
            self.speed_buffer[track_id] = []
            return 0

        # Calculate distance in pixels
        prev_pos = self.prev_positions[track_id]
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters using calibration factor
        distance_meters = distance_pixels / self.pixels_per_meter
        
        # Calculate time delta based on fps
        time_delta = 1.0 / fps  # seconds
        
        # Calculate speed in m/s then convert to km/h
        speed_ms = distance_meters / time_delta
        speed_kmh = speed_ms * 3.6  # convert m/s to km/h
        
        # Smooth speed using moving average
        self.speed_buffer[track_id].append(speed_kmh)
        if len(self.speed_buffer[track_id]) > 5:  # keep last 5 speeds
            self.speed_buffer[track_id].pop(0)
        
        avg_speed = np.mean(self.speed_buffer[track_id])
        
        # Update previous position and time
        self.prev_positions[track_id] = current_pos
        self.prev_times[track_id] = time()
        
        return avg_speed

    def estimate_speed(self, im0, fps=30, t=1):
        """Override parent method to include counting logic"""
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        self.annotator.draw_region(
            reg_pts=self.region,
            color=(104, 0, 123),
            thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)
            
            # Calculate centroid
            centroid = ((box[0] + box[2])/2, (box[1] + box[3])/2)
            
            # Calculate speed
            current_speed = self.calculate_speed(track_id, centroid, fps)
            self.spd[track_id] = current_speed
            
            # Count based on speed
            self.count_by_speed(track_id, current_speed, fps, t)
            
            # Display speed label
            speed_label = f"{int(current_speed)} km/h"
            self.annotator.box_label(box, label=speed_label, color=colors(track_id, True))

        self.display_output(im0)
        return im0
    
# app_speed_counter.py
import cv2
import json

# Load regions
file_path = "regions.json"
with open(file_path, "r") as f:
    data = json.load(f)
region1 = data.get("region1", [])

cap = cv2.VideoCapture("night_rain.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

counter = SpeedCounter(
    count_mode="speeding",
    speed_threshold=30,
    region=region1,
    model="best_combined.pt",
    show=True,
    device="mps"
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
        
    im0 = counter.estimate_speed(im0, fps=fps, t=2)
    print(f"Count: {counter.get_count()}")

cap.release()
cv2.destroyAllWindows()