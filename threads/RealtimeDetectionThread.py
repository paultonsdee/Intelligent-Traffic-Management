# --------------------- Realtime Detection Thread --------------------- #

from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from config.Config import Config
from config.config_logger import logger
from ultralytics import solutions

class RealtimeDetectionThread(QThread):
    """
    Thread to handle realtime detection using the webcam.
    Emits signals to update the GUI with detected frames.
    """
    change_pixmap_original = pyqtSignal(np.ndarray)
    change_pixmap_predicted = pyqtSignal(np.ndarray)
    detection_error = pyqtSignal(str)

    def __init__(self, region1, region2, region3, parent=None):
        super().__init__(parent)
        self.region1 = region1
        self.region2 = region2
        self.region3 = region3
        self._run_flag = True
        self.logger = logger.getChild(self.__class__.__name__)
        # Initialize object detectors as needed
        # For example, using YOLO from Ultralytics
        try:
            self.detector = solutions.YOLO(Config.ULTRALYTICS_MODEL_PATH)
            self.logger.info("Realtime YOLO model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.detector = None

    def run(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            error_msg = "Cannot access the webcam."
            self.logger.error(error_msg)
            self.detection_error.emit(error_msg)
            return

        self.logger.info("Webcam accessed successfully for realtime detection.")

        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                original_frame = frame.copy()
                # Emit the original frame for display
                self.change_pixmap_original.emit(original_frame)

                # Perform detection if the model is loaded
                if self.detector:
                    try:
                        results = self.detector.predict(frame, conf=0.5, verbose=False)
                        # Draw detections on the frame
                        predicted_frame = results[0].plot()
                        self.change_pixmap_predicted.emit(predicted_frame)
                        # Optionally, update violation counts based on detections
                        # Implement your logic here
                    except Exception as e:
                        self.logger.error(f"Detection failed: {e}")
                        self.detection_error.emit(f"Detection failed: {e}")
                else:
                    self.change_pixmap_predicted.emit(frame)
            else:
                error_msg = "Failed to read frame from webcam."
                self.logger.error(error_msg)
                self.detection_error.emit(error_msg)
                break

            # Add a small sleep to reduce CPU usage
            self.msleep(30)

        # Release the webcam
        self.cap.release()
        self.logger.info("RealtimeDetectionThread has stopped.")

    def stop(self):
        """
        Stops the thread gracefully.
        """
        self._run_flag = False
        self.wait()
