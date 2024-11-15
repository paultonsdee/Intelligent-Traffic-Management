# speed_based_counter.py
from speed_estimation import SpeedEstimator
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from shapely.geometry import Polygon, Point
import os

class SpeedBasedCounter(SpeedEstimator):
    """
    A class to count objects based on their speed within a specified region after a certain time.
    """

    def __init__(self, speed_threshold=0, t=0, scale_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.speed_threshold = speed_threshold  # Speed threshold for counting
        self.t = t  # Time threshold in seconds
        self.frame_threshold = 0  # Will be computed based on fps and t
        self.count_zero_speed = 0
        self.count_non_zero_speed = 0
        self.count_exceeding_speed = 0
        self.object_frame_counters = {}  # {track_id: frame_count}
        self.track_history = {}  # To store positions for speed calculation
        self.scale_factor = scale_factor  # Scale to convert pixel distance to meters
        self.region_polygon = Polygon(self.region)  # Define the region as a polygon

    def estimate_speed_and_count(self, im0, fps):
        """
        Estimates speed and counts objects based on speed criteria within a specified region.

        Args:
            im0 (numpy.ndarray): The input image or frame.
            fps (float): Frames per second of the video.
        """
        self.frame_threshold = int(fps * self.t)
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        for track_id, box, cls in zip(self.track_ids, self.boxes, self.clss):
            # Calculate centroid
            centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            # Check if centroid is within the region
            if not self.region_polygon.contains(Point(centroid)):
                continue  # Skip objects outside the region

            # Calculate speed
            speed = self.calculate_speed(track_id, box, fps)
            if speed is None:
                continue  # Not enough data to calculate speed yet

            self.annotator.box_label(box, f"{self.names[cls]} {speed:.2f} km/h", color=colors(cls, True))

            if track_id not in self.object_frame_counters:
                self.object_frame_counters[track_id] = {'frames': 0, 'counted': False}

            self.object_frame_counters[track_id]['frames'] += 1

            if (
                self.object_frame_counters[track_id]['frames'] >= self.frame_threshold
                and not self.object_frame_counters[track_id]['counted']
            ):
                self.object_frame_counters[track_id]['counted'] = True
                if speed == 0:
                    self.count_zero_speed += 1
                else:
                    self.count_non_zero_speed += 1
                if speed >= self.speed_threshold:
                    self.count_exceeding_speed += 1

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )
        self.display_counts(im0)
        self.display_output(im0)

    def calculate_speed(self, track_id, box, fps):
        """
        Calculates the speed of an object based on its movement between frames.

        Args:
            track_id (int): Unique identifier for the tracked object.
            box (np.ndarray): Bounding box coordinates.
            fps (float): Frames per second of the video.

        Returns:
            float: Speed in km/h, or None if speed can't be calculated yet.
        """
        # Initialize tracking history for the object
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(box)

        # Need at least two positions to calculate speed
        if len(self.track_history[track_id]) >= 2:
            # Get last two positions
            prev_box = self.track_history[track_id][-2]
            curr_box = self.track_history[track_id][-1]

            # Calculate centroids
            prev_centroid = ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2)
            curr_centroid = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)

            # Calculate pixel distance
            pixel_distance = np.linalg.norm(np.array(curr_centroid) - np.array(prev_centroid))

            # Convert pixel distance to real-world units (meters)
            real_distance = pixel_distance * self.scale_factor

            # Calculate time between frames
            time_elapsed = 1 / fps

            # Calculate speed in m/s
            speed_m_per_s = real_distance / time_elapsed

            # Convert to km/h
            speed_km_per_h = speed_m_per_s * 3.6

            return speed_km_per_h
        else:
            return None  # Not enough data to calculate speed

    def display_counts(self, im0):
        """
        Displays counts on the image.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on.
        """
        counts_text = {
            'Zero Speed': str(self.count_zero_speed),
            'Non-zero Speed': str(self.count_non_zero_speed),
            f'Speed > {self.speed_threshold} km/h': str(self.count_exceeding_speed)
        }
        self.annotator.display_analytics(im0, counts_text, (104, 31, 17), (255, 255, 255), 10)
 
# app_speed_estimator.py
import cv2
import json
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

file_path = "regions.json"
with open(file_path, "r") as f:
    data = json.load(f)
region1 = data.get("region1", [])

cap = cv2.VideoCapture("/Users/phatvu/Downloads/street.mov")
assert cap.isOpened(), "Error reading video file"
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

speed_region = region1

speed_counter = SpeedBasedCounter(
    model="yolo11n.pt",
    region=speed_region,
    show=True,
    device="mps",
    speed_threshold=50,  # Set your speed threshold here
    t=0.5  # Time threshold in seconds
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        speed_counter.estimate_speed_and_count(im0, fps)
        os.system('clear')
        print(speed_counter.count_zero_speed)
        print(speed_counter.count_non_zero_speed)
        print(speed_counter.count_exceeding_speed)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Video frame is empty or video processing has been successfully completed.")
        break

cap.release()
cv2.destroyAllWindows()