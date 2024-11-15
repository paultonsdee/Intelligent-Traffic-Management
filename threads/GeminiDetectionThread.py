# --------------------- Gemini Detection Thread --------------------- #
import cv2
import numpy as np
from PIL import Image
from models.GeminiActions import parse_gemini_response
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from config.Config import Config
from config.config_logger import logger
from config.ConfigModels import model_gemini


class GeminiDetectionThread(QThread):
    """
    Thread to handle ambulance detection using the Gemini API.
    Emits a signal indicating whether bounding boxes were detected.
    """
    bounding_boxes_detected = pyqtSignal(bool)  # Emits True if bounding boxes are detected

    def __init__(self, video_file, region3, start_frame, fps, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.region3 = region3
        self.start_frame = start_frame
        self.fps = fps
        self._run_flag = True
        self.model = model_gemini
        self.logger = logger.getChild(self.__class__.__name__)

    def run(self):
        if not self.model:
            self.logger.error("Gemini model not initialized.")
            self.bounding_boxes_detected.emit(False)
            return

        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            self.logger.error("Failed to open video file.")
            self.bounding_boxes_detected.emit(False)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = 3
        frame_interval = int(self.fps * 3 / num_frames)
        frames_to_extract = [self.start_frame + i * frame_interval for i in range(num_frames)]
        responses = []

        for frame_num in frames_to_extract:
            if frame_num >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            roi = self.crop_to_region(frame, self.region3)
            has_bbox = self.send_to_gemini(roi)
            if has_bbox:
                responses.append(has_bbox)

        cap.release()

        bbox_count = sum(responses)
        no_bbox_count = len(responses) - bbox_count
        if bbox_count > no_bbox_count:
            # Switch traffic light to green
            self.bounding_boxes_detected.emit(True)
            self.logger.info("The ambulance has been detected.")
            # Continue detection every 2 seconds
            while self._run_flag:
                # time.sleep(2)
                next_frame = min(self.start_frame + int(self.fps * num_frames), total_frames - 1)
                cap = cv2.VideoCapture(self.video_file)
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    break
                roi = self.crop_to_region(frame, self.region3)
                has_bbox = self.send_to_gemini(roi)
                if has_bbox:
                    continue  # Continue looping
                else:
                    # No bounding box detected, reset traffic light
                    self.bounding_boxes_detected.emit(False)
                    self.logger.info("No bounding boxes detected in subsequent frames.")
                    break
        else:
            # Not enough bounding boxes detected
            self.bounding_boxes_detected.emit(False)
            self.logger.info("Bounding boxes not detected sufficiently.")

    def crop_to_region_v1(self, frame, region):
        pts = np.array(region, np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        x, y, w, h = cv2.boundingRect(pts)
        roi_cropped = roi[y:y+h, x:x+w]
        return roi_cropped
    
    def crop_to_region(self, frame, region):
        pts = np.array(region, np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        return roi

    def send_to_gemini(self, image):
        """
        Sends the cropped region to the Gemini API for ambulance detection.
        """
        if image.size == 0:
            self.logger.warning("Empty ROI image, skipping Gemini detection.")
            return False

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        width, height = pil_image.size
        try:
            response = self.model.generate_content(
                [Config.DETECT_AMBULANCE_PROMPT, pil_image],
            ).text
            bbox = parse_gemini_response(response, width, height)
            if bbox:
                self.logger.info("Ambulance detected by Gemini.")
                return True
            else:
                self.logger.info("No ambulance detected by Gemini.")
                return False
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            return False

    def stop(self):
        """
        Stops the thread gracefully.
        """
        self._run_flag = False
        self.wait()