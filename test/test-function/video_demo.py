# speed_based_counter_improved.py
from speed_estimation import SpeedEstimator
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from shapely.geometry import Polygon, Point
import os
from collections import deque

class SpeedBasedCounter(SpeedEstimator):
    """
    A class to count objects based on their speed within a specified region after a certain time.
    """

    def __init__(self, speed_threshold=0, t=0, scale_factor=1.0, average_frames=5, max_history=10, **kwargs):
        """
        Initializes the SpeedBasedCounter object.

        Args:
            speed_threshold (float): Speed threshold for counting (km/h).
            t (float): Time threshold in seconds.
            scale_factor (float): Scale to convert pixel distance to meters.
            average_frames (int): Number of frames to average speed over.
            max_history (int): Maximum number of frames to keep in history for each object.
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
        self.object_frame_counters = {}  # {track_id: {'frames': int, 'counted': bool, 'speed_accumulator': float, 'speed_samples': int}}
        self.track_history = {}  # {track_id: deque of (frame_number, centroid)}
        self.scale_factor = scale_factor  # Scale to convert pixel distance to meters
        self.region_polygon = Polygon(self.region)  # Define the region as a polygon
        self.max_history = max_history  # Maximum history frames to keep

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

        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame number

        for track_id, box, cls in zip(self.track_ids, self.boxes, self.clss):
            # Calculate centroid
            centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            # Check if centroid is within the region
            if not self.region_polygon.contains(Point(centroid)):
                continue  # Skip objects outside the region

            # Update tracking history
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.max_history)
            self.track_history[track_id].append((current_frame_number, centroid))

            # Calculate speed
            speed = self.calculate_speed(track_id, fps)
            if speed is None:
                continue  # Not enough data to calculate speed yet

            # Annotate bounding box with speed
            label = f"{self.names[cls]} {speed:.2f} km/h"
            self.annotator.box_label(box, label, color=colors(cls, True))

            # Initialize frame counter for the object if not present
            if track_id not in self.object_frame_counters:
                self.object_frame_counters[track_id] = {
                    'frames': 0,
                    'counted': False,
                    'speed_accumulator': 0.0,
                    'speed_samples': 0
                }

            # Update frame counter
            self.object_frame_counters[track_id]['frames'] += 1
            self.object_frame_counters[track_id]['speed_accumulator'] += speed
            self.object_frame_counters[track_id]['speed_samples'] += 1

            # Check if object has been in the region for the required time threshold
            if (
                self.object_frame_counters[track_id]['frames'] >= self.frame_threshold
                and not self.object_frame_counters[track_id]['counted']
            ):
                # Calculate average speed
                avg_speed = self.object_frame_counters[track_id]['speed_accumulator'] / self.object_frame_counters[track_id]['speed_samples']
                self.object_frame_counters[track_id]['counted'] = True

                if avg_speed == 0:
                    self.count_zero_speed += 1
                else:
                    self.count_non_zero_speed += 1

                if avg_speed >= self.speed_threshold:
                    self.count_exceeding_speed += 1

        # Draw the counting region
        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )
        # Display counts on the frame
        self.display_counts(im0)
        # Display the annotated frame
        self.display_output(im0)

    def calculate_speed(self, track_id, fps):
        """
        Calculates the average speed of an object based on its movement over multiple frames.

        Args:
            track_id (int): Unique identifier for the tracked object.
            fps (float): Frames per second of the video.

        Returns:
            float: Average speed in km/h, or None if speed can't be calculated yet.
        """
        history = self.track_history.get(track_id, None)
        if history is None or len(history) < 2:
            return None  # Not enough data to calculate speed

        total_pixel_distance = 0.0
        total_time = 0.0

        # Calculate total pixel distance and total time over the history
        previous_frame, previous_centroid = history[0]
        for frame_number, centroid in list(history)[1:]:
            pixel_distance = np.linalg.norm(np.array(centroid) - np.array(previous_centroid))
            frame_gap = frame_number - previous_frame
            if frame_gap == 0:
                continue  # Avoid division by zero
            time_elapsed = frame_gap / fps
            total_pixel_distance += pixel_distance
            total_time += time_elapsed
            previous_frame, previous_centroid = frame_number, centroid

        if total_time == 0:
            return None  # Avoid division by zero

        # Convert pixel distance to real-world distance (meters)
        real_distance = total_pixel_distance * self.scale_factor

        # Calculate speed in m/s
        speed_m_per_s = real_distance / total_time

        # Convert to km/h
        speed_km_per_h = speed_m_per_s * 3.6

        # Validate speed
        if 0 <= speed_km_per_h < 300:
            return speed_km_per_h
        else:
            return None  # Discard unrealistic speeds

    def display_counts(self, im0):
        """
        Displays counts on the image.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on.
        """
        counts_text = {
            # 'Zero Speed': str(self.count_zero_speed),
            'Car/Bus/Truck driving in the wrong lane': str(self.count_non_zero_speed),
            # f'Speed > {self.speed_threshold} km/h': str(self.count_exceeding_speed)
        }
        # Định vị trí hiển thị các số đếm trên góc trái trên của video
        x, y = 10, 30  # Tọa độ bắt đầu
        font_scale = 0.7
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        for key, value in counts_text.items():
            text = f"{key}: {value}"
            # Sử dụng nền màu đen với độ trong suốt để tăng độ tương phản
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(im0, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (0, 0, 0), cv2.FILLED)
            cv2.putText(im0, text, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            y += text_height + 10  # Dời xuống dưới cho dòng tiếp theo

# Main Execution Script
import cv2
import json
import logging
import os

from speed_estimation import SpeedEstimator  # Ensure this imports correctly

# Thiết lập logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Đọc vùng từ file JSON
file_path = "regions.json"
with open(file_path, "r") as f:
    data = json.load(f)
region = data.get("region2", [])

# Mở video
cap = cv2.VideoCapture("night_rain.mp4")
assert cap.isOpened(), "Error reading video file"
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter("ll.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

speed_region = region

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
    average_frames=5,  # Số khung hình để trung bình vận tốc
    max_history=10,  # Số khung hình lưu trữ trong lịch sử theo dõi
    classes=[1, 2, 3],  # Chỉ theo dõi các lớp xe máy
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        speed_counter.estimate_speed_and_count(im0, fps)
        # Xóa màn hình console (chỉ hoạt động trên hệ điều hành Unix)
        if os.name == 'posix':
            os.system('clear')
        elif os.name == 'nt':
            os.system('cls')
        print(f"Zero Speed: {speed_counter.count_zero_speed}")
        print(f"Non-zero Speed: {speed_counter.count_non_zero_speed}")
        print(f"Speed > {speed_counter.speed_threshold} km/h: {speed_counter.count_exceeding_speed}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
