# --------------------- Video Processing Thread --------------------- #
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time
from config.config_logger import logger
import wise_paas.MyWisepaas as toDatahub

class VideoThread(QThread):
    """
    Thread to handle video processing, object counting, and frame updates.
    Emits signals to update the GUI with original and predicted frames, slider values, and video completion.
    """
    change_pixmap_original = pyqtSignal(np.ndarray)
    change_pixmap_predicted = pyqtSignal(np.ndarray)
    update_slider = pyqtSignal(int, int)  # current_frame, total_frames
    video_finished = pyqtSignal()
    cars_counted = pyqtSignal(int, str)  # count, type

    def __init__(self, video_file, counter_region1, counter_region2, counter_region3_in, counter_region3_out, object_counter, violation_tracker, traffic_light_window, region3, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.counter_region1 = counter_region1
        self.counter_region2 = counter_region2
        self.counter_region3_in = counter_region3_in
        self.counter_region3_out = counter_region3_out
        self.object_counter = object_counter
        self.violation_tracker = violation_tracker
        self.traffic_light_window = traffic_light_window
        self.region3 = region3
        self._run_flag = True
        self.paused = False
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30  # Default FPS
        self.seek_requested = False
        self.seek_frame = 0
        # Car counting variables
        self.traffic_light_state = 'green'
        self.no_parking = False
        self.counting = False
        self.count_start_time = None
        self.count_duration = 300  # 5 minutes
        self.car_speed_zero_count = 0
        self.car_speed_nonzero_count = 0
        self.logger = logger.getChild(self.__class__.__name__)

    def run(self):

        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            self.logger.error("Cannot open video file.")
            self.video_finished.emit()
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # Fallback FPS
        self.t = 0.5
        self.step_frame = 5
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.update_slider.emit(self.current_frame, self.total_frames)
        self.logger.info(f"VideoThread started. Total frames: {self.total_frames}, FPS: {self.fps}")

        while self._run_flag:
            if not self.paused:
                if self.seek_requested:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                    self.current_frame = self.seek_frame
                    self.seek_requested = False
                    self.logger.info(f"Seeked to frame {self.seek_frame}")

                ret, frame = self.cap.read()
                if ret:
                    violations = 0
                    original = frame.copy()

                    # Process frame with ObjectCounters
                    detections_region1 = self.counter_region1.estimate_speed(frame)
                    motorcycle_count = self.counter_region1.count_positive
                    if motorcycle_count - self.violation_tracker.violations['motorcycles'] > 1:
                        violations += self.counter_region1.count_positive
                        self.counter_region1.count_positive = 0
                    self.violation_tracker.update_violations('motorcycles', motorcycle_count)

                    detections_region2 = self.counter_region2.estimate_speed(frame)
                    large_vehicle_count = self.counter_region2.count_positive
                    if large_vehicle_count - self.violation_tracker.violations['large_vehicles'] > 1:
                        violations += self.counter_region2.count_positive
                        self.counter_region2.count_positive = 0
                    self.violation_tracker.update_violations('large_vehicles', large_vehicle_count)

                    if self.traffic_light_window.is_no_parking_checked():
                        detections_region3_in = self.counter_region3_in.estimate_speed(frame)
                        illegal_parking_count = self.counter_region3_in.count_positive

                        if illegal_parking_count - self.violation_tracker.violations['illegal_parking'] > 1:
                            violations += self.counter_region3_in.count_positive
                            self.counter_region3_in.count_positive = 0
                        self.violation_tracker.update_violations('illegal_parking', illegal_parking_count)

                    if self.traffic_light_window.current_state == 'red':
                        detections_region3_out = self.counter_region3_out.estimate_speed(frame)
                        red_light_violations = self.counter_region3_out.count_positive
                        if red_light_violations - self.violation_tracker.violations['red_light'] > 1:
                            violations += self.counter_region3_out.count_positive
                            self.counter_region3_out.count_positive = 0
                        self.violation_tracker.update_violations('red_light', red_light_violations)

                    frame = self.object_counter.count(frame, self.fps, 0)
                    def polygon_area(vertices):
                        n = len(vertices)  # Number of vertices
                        area = 0
                        for i in range(n):
                            x1, y1 = vertices[i]
                            x2, y2 = vertices[(i + 1) % n]  # Next vertex, wrap around
                            area += x1 * y2 - y1 * x2
                        return abs(area) / 2
                    density = self.object_counter.total_count() / polygon_area(self.region3)

                    from datetime import datetime
                    timestamp = datetime.now().isoformat()

                    toDatahub.sendData("Timestamp", timestamp, "Violation", violations, "Density", density)
                    #toDatahub.__sendData("TagTimestamp", timestamp, "TagViolation", violations, "TagDensity", density)
                    # Red light violations are handled separately

                    # Emit processed frames
                    self.change_pixmap_original.emit(original)
                    self.change_pixmap_predicted.emit(frame)

                    self.current_frame += self.step_frame
                    self.update_slider.emit(self.current_frame, self.total_frames)

                    # # Time delay between frames
                    # delay = int(1000 / self.fps)
                    # self.msleep(delay)
                else:
                    break
            else:
                self.msleep(100)  # Check again after 100ms if paused

        self.cap.release()
        self.video_finished.emit()
        self.logger.info("VideoThread has finished processing.")

    def pause(self):
        """
        Pauses the video playback.
        """
        self.paused = True
        self.logger.info("Video playback paused.")

    def resume(self):
        """
        Resumes the video playback.
        """
        self.paused = False
        self.logger.info("Video playback resumed.")

    def stop(self):
        """
        Stops the thread gracefully.
        """
        self._run_flag = False
        self.wait()
        self.logger.info("VideoThread stopped.")

    def seek(self, frame_number):
        """
        Seeks to a specific frame number in the video.
        """
        self.seek_frame = frame_number
        self.seek_requested = True
        self.logger.info(f"Seek requested to frame {frame_number}.")

    def update_traffic_light(self, state, no_parking):
        """
        Updates the traffic light state and handles car counting based on the state.
        """
        self.traffic_light_state = state
        self.no_parking = no_parking
        if self.traffic_light_state == 'green' and self.no_parking:
            if not self.counting:
                self.start_counting_zero_speed()
        elif self.traffic_light_state == 'red':
            if not self.counting:
                self.start_counting_nonzero_speed()
        else:
            self.stop_counting()

    def start_counting_zero_speed(self):
        """
        Starts counting cars with zero speed for illegal parking.
        """
        self.counting = True
        self.count_start_time = time.time()
        self.car_speed_zero_count = 0
        self.logger.info("Started counting cars with speed = 0 for illegal parking.")

    def start_counting_nonzero_speed(self):
        """
        Starts counting cars with non-zero speed for red light violations.
        """
        self.counting = True
        self.count_start_time = time.time()
        self.car_speed_nonzero_count = 0
        self.logger.info("Started counting cars with speed != 0 for red light violations.")

    def stop_counting(self):
        """
        Stops the current counting process and emits the count.
        """
        if self.counting:
            self.counting = False
            if self.traffic_light_state == 'green' and self.no_parking:
                # Emit count for illegal parking
                self.cars_counted.emit(self.car_speed_zero_count, "Vận tốc = 0")
                self.logger.info(f"Counted {self.car_speed_zero_count} cars with speed = 0 for illegal parking.")
            elif self.traffic_light_state == 'red':
                # Emit count for red light violations
                self.cars_counted.emit(self.car_speed_nonzero_count, "Vận tốc != 0")
                self.logger.info(f"Counted {self.car_speed_nonzero_count} cars with speed != 0 for red light violations.")
            self.count_start_time = None