import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("/Users/phatvu/Downloads/5995495902553.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(200, 200), (600, 200), (600, 600), (200, 600)]

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="M9-AIoT_Developer_InnoWorks/Intelligence-Traffic-Management-System/best_combined.pt",
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)

cap.release()
cv2.destroyAllWindows()

# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class ObjectCounter(BaseSolution):
    """
    A class to manage the counting of objects outside a specified region in a video stream.
    """

    def __init__(self, **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.classwise_counts = {}  # Dictionary for counts, categorized by object class
        self.region_initialized = False  # Bool variable for region initialization

        self.show_out = self.CFG["show_out"]

    def count_objects(self, track_line, box, track_id, prev_position, cls):
        """
        Counts objects outside a polygonal or linear region based on their tracks.

        Args:
            track_line (Dict): Last 30 frame track record for the object.
            box (List[float]): Bounding box coordinates [x1, y1, x2, y2] for the specific track in the current frame.
            track_id (int): Unique identifier for the tracked object.
            prev_position (Tuple[float, float]): Last frame position coordinates (x, y) of the track.
            cls (int): Class index for classwise count updates.
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        # Check if the object is outside the region
        if len(self.region) >= 3 and not self.r_s.contains(self.Point(track_line[-1])):
            self.counted_ids.append(track_id)
            self.out_count += 1
            self.classwise_counts[self.names[cls]]["OUT"] += 1

    def store_classwise_counts(self, cls):
        """
        Initialize class-wise counts for a specific object class if not already present.
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"OUT": 0}

    def display_counts(self, im0):
        """
        Displays object counts on the input image or frame.
        """
        labels_dict = {
            str.capitalize(key): f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def count(self, im0):
        """
        Processes input data (frames or object tracks) and updates object counts.
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.store_tracking_history(track_id, box)
            self.store_classwise_counts(cls)

            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_line, box, track_id, prev_position, cls)

        self.display_counts(im0)
        self.display_output(im0)

        return im0
