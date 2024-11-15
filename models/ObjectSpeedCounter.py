# speed_estimator.py
# Ultralytics YOLO 🚀, AGPL-3.0 license
# This is a modified version of the original code from Ultralytics YOLO, licensed under AGPL-3.0.
# The original source can be found at: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/speed_.py
# You must comply with the terms of AGPL-3.0 when using, modifying, or distributing this code.
# Please retain this license notice and provide the source code for any public-facing services that use this code.

from collections import defaultdict
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry import LineString, Point  # Import Point


class ObjectSpeedCounter(BaseSolution):
    """
    A class to estimate the speed of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for estimating object speeds using
    tracking data in video streams.

    Attributes:
        spd (Dict[int, float]): Dictionary storing speed data for tracked objects.
        trkd_ids (List[int]): List of tracked object IDs that have already been speed-estimated.
        trk_pp (Dict[int, Tuple[float, float]]): Dictionary storing previous positions for tracked objects.
        count_zero (int): Counter for vehicles with zero velocity.
        count_positive (int): Counter for vehicles with positive velocity.
        annotator (Annotator): Annotator object for drawing on images.
        region (List[Tuple[int, int]]): List of points defining the speed estimation region.
        frame_threshold (int): Number of consecutive frames a vehicle must maintain its speed to be counted.
        zero_speed_frames (Dict[int, int]): Dictionary tracking consecutive zero-speed frames per vehicle.
        positive_speed_frames (Dict[int, int]): Dictionary tracking consecutive positive-speed frames per vehicle.
        counted_ids (Set[int]): Set of vehicle IDs that have been counted for zero speed.
        counted_positive_ids (Set[int]): Set of vehicle IDs that have been counted for positive speed.
        mode (str): Mode can be 'in' or 'out' to focus on objects inside or outside the region.

    Methods:
        initialize_region: Initializes the speed estimation region.
        estimate_speed: Estimates the speed of objects based on tracking data.
        store_tracking_history: Stores the tracking history for an object.
        extract_tracks: Extracts tracks from the current frame.
        display_output: Displays the output with annotations.
    """

    def __init__(self, fps, t, mode='in', **kwargs):
        """
        Initializes the SpeedEstimator object with speed estimation parameters and data structures.

        Args:
            fps (float): Frames per second of the video.
            t (float): Time in seconds a vehicle must maintain its speed to be counted.
            mode (str): 'in' to focus on objects inside the region, 'out' for outside.
            **kwargs: Additional keyword arguments for the BaseSolution.
        """
        super().__init__(**kwargs)

        self.initialize_region()  # Initialize speed region

        self.spd = {}  # Dictionary for speed data
        self.trkd_ids = []  # List for already speed-estimated and tracked IDs
        self.trk_pp = {}  # Dictionary for tracks' previous point
        self.count_zero = 0  # Counter for vehicles with zero velocity
        self.count_positive = 0  # Counter for vehicles with positive velocity

        # Calculate frame threshold based on fps and t
        self.frame_threshold = int(fps * t)

        # Dictionaries to track consecutive speed frames and counted IDs
        self.zero_speed_frames = defaultdict(int)
        self.positive_speed_frames = defaultdict(int)
        self.counted_ids = set()
        self.counted_positive_ids = set()

        self.mode = mode  # Add mode attribute

    def estimate_speed(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            current_position = self.track_line[-1]
            current_point = Point(current_position)

            if self.r_s.contains(current_point):
                inside_region = True
            else:
                inside_region = False

            if (self.mode == 'in' and inside_region) or (self.mode == 'out' and not inside_region):
                # Estimate speed
                distance = np.linalg.norm(
                    np.array(self.track_line[-1]) - np.array(self.trk_pp[track_id])
                )
                speed = distance  # Need unit conversion

                if speed <= 2:
                    speed = 0
                self.spd[track_id] = speed

                if speed <= 2:
                    self.zero_speed_frames[track_id] += 1

                    if (
                        self.zero_speed_frames[track_id] >= self.frame_threshold
                        and track_id not in self.counted_ids
                    ):
                        self.count_zero += 1
                        self.counted_ids.add(track_id)
                        print(
                            f"Vehicle with track ID {track_id} has zero velocity. Total zero count: {self.count_zero}"
                        )
                else:
                    self.zero_speed_frames[track_id] = 0

                    self.positive_speed_frames[track_id] += 1

                    if (
                        self.positive_speed_frames[track_id] >= self.frame_threshold
                        and track_id not in self.counted_positive_ids
                    ):
                        self.count_positive += 1
                        self.counted_positive_ids.add(track_id)
                        print(
                            f"Vehicle with track ID {track_id} has positive velocity. Total positive count: {self.count_positive}"
                        )

                self.trk_pp[track_id] = self.track_line[-1]

                # Drawing and labeling
                if inside_region:
                    color = colors(track_id, True)
                    region_label = "(In)"
                else:
                    color = (0, 0, 255)  # Red color for outside region
                    region_label = "(Out)"

                label = f"{int(self.spd[track_id])} km/h {region_label}"
                self.annotator.box_label(box, label=label, color=color)
            else:
                # Skip this object
                continue

        self.display_output(im0)

        return im0
    

