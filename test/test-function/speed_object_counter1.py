# speed_based_counter_improved.py
from speed_estimation import SpeedEstimator
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from shapely.geometry import Polygon, Point
import os

class SpeedBasedCounter(SpeedEstimator):
    """
    A class to count objects based on their speed within a specified region after a certain time.
    """

    def __init__(self, speed_threshold=0, t=0, scale_factor=1.0, average_frames=5, **kwargs):
        """
        Initializes the SpeedBasedCounter object.

        Args:
            speed_threshold (float): Speed threshold for counting (km/h).
            t (float): Time threshold in seconds.
            scale_factor (float): Scale to convert pixel distance to meters.
            average_frames (int): Number of frames to average speed over.
            **kwargs: Additional arguments for the base SpeedEstimator class.
        """
        super().__init__(**kwargs)
        self.speed_threshold = speed_threshold  # Speed threshold for counting
        self.t = t  # Time threshold in seconds
        self.average_frames = average_frames  # Number of frames to average speed
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
        if fps == 0:
            return  # Avoid division by zero
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
                self.object_frame_counters[track_id] = {'frames': 0, 'counted': False, 'speed_accumulator': 0, 'speed_samples': 0}

            self.object_frame_counters[track_id]['frames'] += 1
            self.object_frame_counters[track_id]['speed_accumulator'] += speed
            self.object_frame_counters[track_id]['speed_samples'] += 1

            if (
                self.object_frame_counters[track_id]['frames'] >= self.frame_threshold
                and not self.object_frame_counters[track_id]['counted']
            ):
                # Calculate average speed over the accumulated frames
                avg_speed = self.object_frame_counters[track_id]['speed_accumulator'] / self.object_frame_counters[track_id]['speed_samples']
                self.object_frame_counters[track_id]['counted'] = True
                if avg_speed == 0:
                    self.count_zero_speed += 1
                else:
                    self.count_non_zero_speed += 1
                if avg_speed >= self.speed_threshold:
                    self.count_exceeding_speed += 1

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )
        self.display_counts(im0)
        self.display_output(im0)

    def calculate_speed(self, track_id, box, fps):
        """
        Calculates the speed of an object based on its movement over multiple frames.

        Args:
            track_id (int): Unique identifier for the tracked object.
            box (np.ndarray): Bounding box coordinates.
            fps (float): Frames per second of the video.

        Returns:
            float: Average speed in km/h, or None if speed can't be calculated yet.
        """
        # Initialize tracking history for the object
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(box)

        # Keep only the last 'average_frames' positions for speed calculation
        if len(self.track_history[track_id]) > self.average_frames:
            self.track_history[track_id].pop(0)

        if len(self.track_history[track_id]) >= 2:
            # Calculate total distance over the frames
            total_pixel_distance = 0
            total_time = 0
            for i in range(1, len(self.track_history[track_id])):
                prev_box, curr_box = self.track_history[track_id][i - 1], self.track_history[track_id][i]
                prev_centroid = ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2)
                curr_centroid = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)
                pixel_distance = np.linalg.norm(np.array(curr_centroid) - np.array(prev_centroid))
                total_pixel_distance += pixel_distance
                total_time += 1 / fps

            # Convert pixel distance to real-world distance (meters)
            real_distance = total_pixel_distance * self.scale_factor

            # Calculate average speed in m/s
            speed_m_per_s = real_distance / total_time

            # Convert to km/h
            speed_km_per_h = speed_m_per_s * 3.6

            # Validate speed
            if 0 <= speed_km_per_h < 300:
                return speed_km_per_h
            else:
                return None  # Discard unrealistic speeds
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

import cv2
import json
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

file_path = "regions.json"
with open(file_path, "r") as f:
    data = json.load(f)
region3 = data.get("region3", [])

cap = cv2.VideoCapture("/Users/phatvu/Downloads/street.mov")
assert cap.isOpened(), "Error reading video file"
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

speed_region = region3

# Điều chỉnh scale_factor dựa trên thông số thực tế của video
# Ví dụ: Nếu 100 pixel tương ứng với 5 mét trong thực tế, thì scale_factor = 5 / 100 = 0.05
scale_factor = 0.05  # Cần điều chỉnh chính xác

speed_counter = SpeedBasedCounter(
    model="best_combined.pt",
    region=speed_region,
    show=True,
    device="mps",
    speed_threshold=50,  # Đặt ngưỡng vận tốc của bạn ở đây (km/h)
    t=0.5,  # Ngưỡng thời gian trong giây
    scale_factor=scale_factor,  # Đặt scale_factor chính xác
    average_frames=5  # Số khung hình để trung bình vận tốc
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        speed_counter.estimate_speed_and_count(im0, fps)
        os.system('clear')
        print(f"Zero Speed: {speed_counter.count_zero_speed}")
        print(f"Non-zero Speed: {speed_counter.count_non_zero_speed}")
        print(f"Speed > {speed_counter.speed_threshold} km/h: {speed_counter.count_exceeding_speed}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Video frame is empty or video processing has been successfully completed.")
        break

cap.release()
cv2.destroyAllWindows()
