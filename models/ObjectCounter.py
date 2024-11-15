# Ultralytics YOLO 🚀, AGPL-3.0 license
# This is a modified version of the original code from Ultralytics YOLO, licensed under AGPL-3.0.
# The original source can be found at: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/object_counter.py
# You must comply with the terms of AGPL-3.0 when using, modifying, or distributing this code.
# Please retain this license notice and provide the source code for any public-facing services that use this code.

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class ObjectCounter(BaseSolution):
    """
    A class to manage the counting of objects in a real-time video stream based on their tracks.
    """

    def __init__(self, count_mode='in', **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

        self.count_mode = count_mode
        self.in_count = 0  # Counter for objects moving inward
        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.classwise_counts = {}  # Dictionary for counts, categorized by object class
        self.region_initialized = False  # Bool variable for region initialization

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.frame_threshold = 0 # Threshold for counting objects (= fps * t)

        # Thêm dictionary để theo dõi số khung hình mỗi đối tượng đã ở trong hoặc ngoài vùng
        self.object_frame_counters = {}  # {track_id: {'in': count, 'out': count}}

    def count_objects(self, is_in_region, track_id, cls):
        """
        Counts objects based on the duration they have been in or out of the region.

        Args:
            is_in_region (bool): Whether the object is inside the region.
            track_id (int): Unique identifier for the tracked object.
            cls (int): Class index for classwise count updates.
        """
        if track_id not in self.object_frame_counters:
            self.object_frame_counters[track_id] = {'in': 0, 'out': 0}

        if is_in_region:
            self.object_frame_counters[track_id]['in'] += 1
            self.object_frame_counters[track_id]['out'] = 0  # Reset out counter
            if self.object_frame_counters[track_id]['in'] >= self.frame_threshold and track_id not in self.counted_ids:
                self.counted_ids.append(track_id)
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
        else:
            self.object_frame_counters[track_id]['out'] += 1
            self.object_frame_counters[track_id]['in'] = 0  # Reset in counter
            if self.object_frame_counters[track_id]['out'] >= self.frame_threshold and track_id not in self.counted_ids:
                self.counted_ids.append(track_id)
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1

    def store_classwise_counts(self, cls):
        """
        Initialize class-wise counts for a specific object class if not already present.

        Args:
            cls (int): Class index for classwise count updates.
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """
        Displays object counts on the input image or frame.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on.
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def count(self, im0, fps=0, t=0):
        """
        Processes input data and updates object counts based on fps and time t.
        
        Args:
            im0 (numpy.ndarray): The input image or frame to be processed.
            fps (int): Frames per second of the video.
            t (float): Time in seconds.
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.frame_threshold = int(fps * t)
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.store_tracking_history(track_id, box)
            self.store_classwise_counts(cls)

            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(cls), True), track_thickness=self.line_width
            )

            # Kiểm tra vị trí hiện tại của đối tượng
            is_in_region = len(self.region) >= 3 and self.r_s.contains(self.Point(self.track_line[-1]))
            # Hoặc kiểm tra nếu là đường thẳng và đối tượng cắt qua đường
            if len(self.region) < 3:
                is_in_region = self.LineString([self.track_history[track_id][-1], box[:2]]).intersects(self.r_s)

            self.count_objects(is_in_region, track_id, cls)

        self.display_counts(im0)
        self.display_output(im0)

        return im0
    
    def total_count(self):
        """
        Returns the total count of objects moving inward and outward.
        """
        return self.in_count + self.out_count