# Import statements remain the same as in your original code
import sys
import os
import re
import json
import time
import logging
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading

import cv2
import numpy as np
import librosa
import torch
import onnx
import onnxruntime as ort
from PIL import Image

from sklearn.preprocessing import LabelEncoder

from moviepy.editor import VideoFileClip

import google.generativeai as genai

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QWidget, QHBoxLayout, QVBoxLayout, QMessageBox, QSlider, QDialog, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QUrl, QPoint
from PyQt5.QtGui import QPixmap, QImage, QResizeEvent, QPainter, QPen
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from logging.handlers import RotatingFileHandler

# --------------------- Configuration Management --------------------- #

class Config:
    """Configuration class to manage environment variables and settings."""

    # Email settings
    FROM_EMAIL = "vhp08072004@gmail.com"
    EMAIL_PASSWORD = "your_app_specific_password"  # Replace with your actual app password
    TO_EMAIL = "vhp08071974@gmail.com"

    # Gemini API
    GEMINI_API_KEY = "your_gemini_api_key"  # Replace with your actual Gemini API key

    # Model paths
    CNN_MODEL_PATH = "CNN_model.onnx"
    LSTM_MODEL_PATH = "LSTM_model.onnx"

    # Logging settings
    LOG_FILE = "system.log"
    LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
    LOG_BACKUP_COUNT = 5

# --------------------- Logging Configuration --------------------- #

# Configure rotating logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=Config.LOG_MAX_BYTES,
    backupCount=Config.LOG_BACKUP_COUNT
)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Also log to console with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --------------------- Violation Tracker --------------------- #

class ViolationTracker:
    """
    Tracks traffic violations and sends email alerts when violations exceed thresholds.
    """
    def __init__(self):
        self.violations = {
            'large_vehicles': 0,    # Placeholder for potential future use
            'motorcycles': 0,       # Placeholder for potential future use
            'illegal_parking': 0,   # Placeholder for potential future use
            'red_light': 0          # Placeholder for potential future use
        }
        self.thresholds = Config.VIOLATION_THRESHOLDS if hasattr(Config, 'VIOLATION_THRESHOLDS') else {
            'large_vehicles': 10,
            'motorcycles': 15,
            'illegal_parking': 5,
            'red_light': 7
        }
        self.last_email_time = 0
        self.email_cooldown = 60  # Cooldown to prevent spamming emails

        # Email settings
        self.from_email = Config.FROM_EMAIL
        self.password = Config.EMAIL_PASSWORD
        self.to_email = Config.TO_EMAIL

        # Configure logging for this class
        self.logger = logging.getLogger(self.__class__.__name__)

        # Lock for thread safety
        self.lock = threading.Lock()

    def get_violation_count(self, violation_type):
        with self.lock:
            return self.violations.get(violation_type, 0)

    def update_violations(self, violation_type, count):
        with self.lock:
            self.violations[violation_type] += count
            self.print_violations()
            self.check_and_send_email()

    def print_violations(self):
        # Clear terminal
        os.system('cls' if os.name == 'nt' else 'clear')
        self.logger.info("Current Violations:")
        self.logger.info("-" * 50)
        for vt, cnt in self.violations.items():
            self.logger.info(f"{vt.replace('_', ' ').title()}: {cnt}")
        self.logger.info("-" * 50)

    def send_violation_email(self, violation_details):
        try:
            message = MIMEMultipart()
            message["From"] = self.from_email
            message["To"] = self.to_email
            message["Subject"] = "Traffic Violation Alert"

            body = "Traffic violations have occurred:\n\n"
            for violation_type, count in violation_details.items():
                body += f"Type: {violation_type.replace('_', ' ').title()}\nCount: {count}\n\n"

            message.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.from_email, self.password)
            server.send_message(message)
            server.quit()

            self.logger.info(f"Sent email for violations: {', '.join(violation_details.keys())}")

            # Reset violation counts after email sent
            for violation_type in violation_details.keys():
                self.violations[violation_type] = 0

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")

    def check_and_send_email(self):
        current_time = time.time()
        if current_time - self.last_email_time >= self.email_cooldown:
            violations_to_alert = {
                vt: cnt for vt, cnt in self.violations.items()
                if cnt >= self.thresholds.get(vt, 10)
            }
            if violations_to_alert:
                self.send_violation_email(violations_to_alert)
                self.last_email_time = current_time

# --------------------- Gemini API Configuration --------------------- #

def configure_gemini():
    """
    Configures the Gemini API using the provided API key.
    Returns the configured Gemini model.
    """
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")
        logger.info("Gemini API configured successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        return None

model_gemini = configure_gemini()

def parse_gemini_response(response_text):
    """
    Parses the Gemini API response to extract ambulance bounding box data.
    """
    try:
        # Extract JSON code block using regex
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return None
        json_str = json_match.group(0)
        data = json.loads(json_str)
        if "ambulance" in data:
            return data["ambulance"]
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
    return None

# --------------------- PyTorch Configuration --------------------- #

def configure_pytorch():
    """
    Configures PyTorch to use CUDA or CPU.
    Returns the device being used.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cuda_name = torch.cuda.get_device_name(device)
        logger.info(f"Using device: CUDA ({cuda_name})")
    else:
        device = torch.device('cpu')
        logger.info("Using device: CPU")
    return device

device_pt = configure_pytorch()

# --------------------- Feature Extraction and Prediction Functions --------------------- #

def features_extractor(audio, sample_rate):
    """
    Extracts MFCC features from audio data.
    """
    try:
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def load_onnx_model(model_path):
    """
    Loads an ONNX model and returns an ONNX Runtime session.
    """
    try:
        session = ort.InferenceSession(model_path)
        logger.info(f"Loaded ONNX model from {model_path}")
        return session
    except Exception as e:
        logger.error(f"Failed to load ONNX model from {model_path}: {e}")
        return None

# Load models
cnn_session = load_onnx_model(Config.CNN_MODEL_PATH)
lstm_session = load_onnx_model(Config.LSTM_MODEL_PATH)

def predict_cnn(features, session):
    """
    Predicts the class using a CNN ONNX model.
    """
    try:
        # Prepare input
        input_name = session.get_inputs()[0].name
        input_data = features.astype(np.float32).reshape(1, -1, 1)
        predictions = session.run(None, {input_name: input_data})
        predicted_class = np.argmax(predictions[0], axis=1)
        return predicted_class[0], predictions[0][0]
    except Exception as e:
        logger.error(f"CNN prediction failed: {e}")
        return None, None

def predict_lstm(features, session):
    """
    Predicts the class using an LSTM ONNX model.
    """
    try:
        # Prepare input
        input_name = session.get_inputs()[0].name
        input_data = features.astype(np.float32).reshape(1, -1, 80)
        predictions = session.run(None, {input_name: input_data})
        predicted_class = np.argmax(predictions[0], axis=1)
        return predicted_class[0], predictions[0][0]
    except Exception as e:
        logger.error(f"LSTM prediction failed: {e}")
        return None, None

def ensemble_predictions(cnn_probs, lstm_probs):
    """
    Averages the probabilities from CNN and LSTM models for ensemble prediction.
    """
    try:
        ensemble_probs = (cnn_probs + lstm_probs) / 2  # Average probabilities
        return ensemble_probs
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        return None

# --------------------- Gemini Detection Thread --------------------- #

class GeminiDetectionThread(QThread):
    """
    Thread to handle ambulance detection using the Gemini API.
    Emits a signal indicating whether bounding boxes were detected.
    """
    bounding_boxes_detected = pyqtSignal(bool, list)  # Emits True if bounding boxes are detected, along with bounding box data

    def __init__(self, video_file, region3, start_frame, fps, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.region3 = region3
        self.start_frame = start_frame
        self.fps = fps
        self._run_flag = True
        self.model = model_gemini
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        if not self.model:
            self.logger.error("Gemini model not initialized.")
            self.bounding_boxes_detected.emit(False, [])
            return

        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            self.logger.error("Failed to open video file.")
            self.bounding_boxes_detected.emit(False, [])
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(self.fps * 3)  # Process every 3 seconds
        frames_to_extract = [self.start_frame + i * frame_interval for i in range(5)]
        responses = []

        bounding_boxes = []

        for frame_num in frames_to_extract:
            if frame_num >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            roi = self.crop_to_region(frame, self.region3)
            bbox = self.send_to_gemini(roi)
            if bbox:
                responses.append(True)
                bounding_boxes.append(bbox)
            else:
                responses.append(False)

        cap.release()

        bbox_count = len(bounding_boxes)
        no_bbox_count = len(responses) - bbox_count
        if bbox_count > no_bbox_count:
            # Emit signal with bounding boxes
            self.bounding_boxes_detected.emit(True, bounding_boxes)
            self.logger.info("Bounding boxes detected more than non-detections.")
            # Continue detection every 2 seconds
            while self._run_flag:
                time.sleep(2)
                next_frame = min(self.start_frame + int(self.fps * 3), total_frames - 1)
                cap = cv2.VideoCapture(self.video_file)
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    break
                roi = self.crop_to_region(frame, self.region3)
                bbox = self.send_to_gemini(roi)
                if bbox:
                    bounding_boxes.append(bbox)
                    self.bounding_boxes_detected.emit(True, [bbox])
                else:
                    # No bounding box detected, emit False
                    self.bounding_boxes_detected.emit(False, [])
                    self.logger.info("No bounding boxes detected in subsequent frames.")
                    break
        else:
            # Not enough bounding boxes detected
            self.bounding_boxes_detected.emit(False, [])
            self.logger.info("Bounding boxes not detected sufficiently.")

    def crop_to_region(self, frame, region):
        pts = np.array(region, np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        x, y, w, h = cv2.boundingRect(pts)
        roi_cropped = roi[y:y+h, x:x+w]
        return roi_cropped

    def send_to_gemini(self, image):
        """
        Sends the cropped region to the Gemini API for ambulance detection.
        Returns the bounding box if detected, else None.
        """
        if image.size == 0:
            self.logger.warning("Empty ROI image, skipping Gemini detection.")
            return None

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            response = self.model.generate_content(
                [PROMPT, pil_image],
            ).text
            bbox = parse_gemini_response(response)
            if bbox:
                self.logger.info("Bounding box detected by Gemini.")
                return bbox
            else:
                self.logger.info("No bounding box detected by Gemini.")
                return None
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            return None

    def stop(self):
        """
        Stops the thread gracefully.
        """
        self._run_flag = False
        self.wait()

# --------------------- Prompt Definition --------------------- #

PROMPT = (
    "What is the position of the ambulance present in the image? "
    "Output ambulance in JSON format with both object names and positions as a JSON object: "
    '{"ambulance": [x_center, y_center, width, height]}. '
    "Put the answer in a JSON code block, and all numbers in pixel units. Please detect carefully because there may not be an ambulance in the photo. "
    "The bounding box must be wide enough to enclose the entire ambulance (note that in some images, the ambulance may have glare from its lights in dark conditions)."
)

# --------------------- Region Selection Window --------------------- #

class RegionSelectionWindow(QDialog):
    """
    A dialog window for selecting regions on an image.
    """
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Regions")
        self.setGeometry(100, 100, 1280, 720)
        self.image = image
        self.parent = parent  # Reference to MainWindow
        self.setup_ui()
        self.initialize_properties()
        self.selected_region = []  # Current selected points
        self.region1 = []
        self.region2 = []
        self.region3 = []
        self.logger = logging.getLogger(self.__class__.__name__)

        # Apply modern stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                color: #000000;
            }
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #AAAAAA;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #CCCCCC;
            }
        """)

    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        # Canvas to display image
        self.canvas = QLabel(self)
        self.canvas.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.layout.addWidget(self.canvas)
        # Convert image to QImage and display
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qt_image)
        self.canvas.setPixmap(self.pixmap)
        # Buttons
        button_layout = QHBoxLayout()
        self.btn_save_region1 = QPushButton("Save to Region 1")
        self.btn_save_region1.clicked.connect(self.save_region1)
        button_layout.addWidget(self.btn_save_region1)

        self.btn_save_region2 = QPushButton("Save to Region 2")
        self.btn_save_region2.clicked.connect(self.save_region2)
        button_layout.addWidget(self.btn_save_region2)

        self.btn_save_region3 = QPushButton("Save to Region 3")
        self.btn_save_region3.clicked.connect(self.save_region3)
        button_layout.addWidget(self.btn_save_region3)

        self.btn_load_regions = QPushButton("Load Regions")
        self.btn_load_regions.clicked.connect(self.load_regions)
        button_layout.addWidget(self.btn_load_regions)

        self.btn_reset = QPushButton("Reset Selection")
        self.btn_reset.clicked.connect(self.reset_selection)
        button_layout.addWidget(self.btn_reset)

        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(self.btn_ok)

        self.layout.addLayout(button_layout)

        # Mouse event
        self.canvas.mousePressEvent = self.on_canvas_click

    def initialize_properties(self):
        self.current_polygon = []
        self.polygons = []
        self.max_points = 4  # Requires 4 points per region

    def on_canvas_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if len(self.current_polygon) < self.max_points:
            self.current_polygon.append((x, y))
            painter = QPainter(self.canvas.pixmap())
            painter.setPen(QPen(Qt.red, 5))
            painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.end()
            self.canvas.update()

            if len(self.current_polygon) == self.max_points:
                self.draw_polygon(self.current_polygon)
                QMessageBox.information(self, "Notification", f"Selected {self.max_points} points for the current region.")
                self.logger.info(f"Region selected: {self.current_polygon}")

    def draw_polygon(self, polygon):
        painter = QPainter(self.canvas.pixmap())
        painter.setPen(QPen(Qt.blue, 2))
        for i in range(len(polygon)):
            p1 = QPoint(polygon[i][0], polygon[i][1])
            p2 = QPoint(polygon[(i + 1) % len(polygon)][0], polygon[(i + 1) % len(polygon)][1])
            painter.drawLine(p1, p2)
        painter.end()
        self.canvas.update()

    def save_region1(self):
        self.save_region(self.region1, "Region 1")

    def save_region2(self):
        self.save_region(self.region2, "Region 2")

    def save_region3(self):
        self.save_region(self.region3, "Region 3")

    def save_region(self, region, region_name):
        if len(self.current_polygon) != self.max_points:
            QMessageBox.warning(self, "Warning", f"Region requires exactly {self.max_points} points.")
            return
        region.extend(self.current_polygon)
        self.save_regions_to_file()
        QMessageBox.information(self, "Saved", f"{region_name} saved.")
        self.logger.info(f"{region_name} saved with points: {self.current_polygon}")
        self.reset_selection()

    def save_regions_to_file(self):
        data = {
            "region1": self.region1,
            "region2": self.region2,
            "region3": self.region3
        }
        try:
            with open("regions.json", "w") as f:
                json.dump(data, f, indent=4)
            self.logger.info("Regions saved to regions.json")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot save regions: {e}")
            self.logger.error(f"Failed to save regions: {e}")

    def load_regions(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Bounding Boxes",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                self.region1 = data.get("region1", [])
                self.region2 = data.get("region2", [])
                self.region3 = data.get("region3", [])
                if not (self.region1 and self.region2 and self.region3):
                    QMessageBox.warning(self, "Error", "File does not contain all regions.")
                    self.logger.warning("Loaded regions are incomplete.")
                    return
                else:
                    self.draw_loaded_regions()
                    QMessageBox.information(self, "Loaded", "Bounding boxes loaded from file.")
                    self.logger.info("Bounding boxes loaded from file.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Cannot load bounding boxes: {e}")
                self.logger.error(f"Failed to load bounding boxes: {e}")

    def draw_loaded_regions(self):
        self.canvas.setPixmap(self.pixmap.copy())
        for region in [self.region1, self.region2, self.region3]:
            if region:
                self.draw_polygon(region)

    def reset_selection(self):
        self.current_polygon.clear()
        self.canvas.setPixmap(self.pixmap.copy())
        self.canvas.update()
        self.logger.info("Selection reset.")

    def accept(self):
        if not self.region1 or not self.region2 or not self.region3:
            QMessageBox.warning(self, "Error", "Not all regions have been selected.")
            return
        super().accept()

# --------------------- Traffic Light Window --------------------- #

class TrafficLightWindow(QWidget):
    """
    A window that simulates a traffic light with countdown and a "No Parking" option.
    Emits signals when the state changes.
    """
    state_changed = pyqtSignal(str)  # Emits 'green', 'yellow', 'red'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traffic Light")
        self.setGeometry(50, 50, 100, 400)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_ui()
        self.setup_timer()
        self.current_state = 'green'
        self.set_state('green')  # Initialize with green
        self.logger.info("TrafficLightWindow initialized.")

        # Apply modern stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
            }
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #AAAAAA;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #CCCCCC;
            }
            QLabel {
                color: #000000;
            }
            QCheckBox {
                color: #000000;
            }
        """)

    def setup_ui(self):
        layout = QVBoxLayout()

        # Green light button
        self.btn_green = QPushButton()
        self.btn_green.setFixedSize(50, 50)
        self.btn_green.setStyleSheet("background-color: green; border-radius: 25px;")
        self.btn_green.setEnabled(False)  # Disable manual control
        layout.addWidget(self.btn_green, alignment=Qt.AlignCenter)

        # Yellow light button
        self.btn_yellow = QPushButton()
        self.btn_yellow.setFixedSize(50, 50)
        self.btn_yellow.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_yellow.setEnabled(False)  # Disable manual control
        layout.addWidget(self.btn_yellow, alignment=Qt.AlignCenter)

        # Red light button
        self.btn_red = QPushButton()
        self.btn_red.setFixedSize(50, 50)
        self.btn_red.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_red.setEnabled(False)  # Disable manual control
        layout.addWidget(self.btn_red, alignment=Qt.AlignCenter)

        # Countdown label
        self.lbl_countdown = QLabel("30")
        self.lbl_countdown.setAlignment(Qt.AlignCenter)
        self.lbl_countdown.setStyleSheet("font-size: 24px; color: #000000;")
        layout.addWidget(self.lbl_countdown, alignment=Qt.AlignCenter)

        # "No Parking" checkbox
        self.checkbox_no_parking = QCheckBox("No Parking")
        layout.addWidget(self.checkbox_no_parking, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_countdown)
        self.time_left = 30  # Initial time for green light
        self.timer.start(1000)  # Update every second

    def update_countdown(self):
        self.time_left -= 1
        self.lbl_countdown.setText(str(self.time_left))
        if self.time_left <= 0:
            self.transition_state()

    def transition_state(self):
        if self.current_state == 'green':
            self.set_state('yellow')
            self.time_left = 5
        elif self.current_state == 'yellow':
            self.set_state('red')
            self.time_left = 35
        elif self.current_state == 'red':
            self.set_state('green')
            self.time_left = 30

    def set_state(self, state):
        self.current_state = state
        
        # Reset light colors
        self.btn_green.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_yellow.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_red.setStyleSheet("background-color: grey; border-radius: 25px;")
        # Set current light color
        if state == 'green':
            self.btn_green.setStyleSheet("background-color: green; border-radius: 25px;")
        elif state == 'yellow':
            self.btn_yellow.setStyleSheet("background-color: yellow; border-radius: 25px;")
        elif state == 'red':
            self.btn_red.setStyleSheet("background-color: red; border-radius: 25px;")
        self.state_changed.emit(state)
        self.logger.info(f"Traffic light state changed to {state}")

    def is_no_parking_checked(self):
        return self.checkbox_no_parking.isChecked()

# --------------------- Audio Processing Thread --------------------- #

class AudioThread(QThread):
    """
    Thread to handle audio processing and predictions.
    Emits detected audio class and handles load errors.
    """
    detected_audio = pyqtSignal(str, float, float)  # predicted_class, start_time, end_time
    load_error = pyqtSignal(str)

    def __init__(self, video_file, violation_tracker, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.violation_tracker = violation_tracker
        self._run_flag = True
        self.segment_duration = 3.0  # Duration of each audio segment in seconds
        self.current_time = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        # Initialize LabelEncoder
        self.le = LabelEncoder()
        self.le.fit(['ambulance', 'firetruck', 'traffic'])

        # Load video and extract audio
        try:
            video = VideoFileClip(self.video_file)
            self.audio = video.audio
            self.sample_rate = 22050  # Default librosa sample rate
            self.duration = video.duration  # in seconds
            self.logger.info(f"Video duration: {self.duration} seconds")
        except Exception as e:
            error_msg = f"Cannot load video file! Error: {e}"
            self.logger.error(error_msg)
            self.load_error.emit(error_msg)
            return

        # Process audio segments every 3 seconds
        while self._run_flag and self.current_time < self.duration:
            start = self.current_time
            end = min(start + self.segment_duration, self.duration)
            self.logger.info(f"Processing audio segment: {start} - {end} seconds")

            try:
                # Create a temporary audio file for the segment
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                    audio_segment = self.audio.subclip(start, end)
                    audio_segment.write_audiofile(tmp_audio_file.name, fps=self.sample_rate, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
                    tmp_audio_file_path = tmp_audio_file.name

                # Extract features from the audio segment
                y, sr = librosa.load(tmp_audio_file_path, sr=self.sample_rate)
                features = features_extractor(y, sr)

            except Exception as e:
                error_msg = f"Error processing audio segment {start}-{end}: {e}"
                self.logger.error(error_msg)
                self.load_error.emit(error_msg)
                self.current_time += self.segment_duration
                continue
            finally:
                # Ensure the temporary file is removed
                if os.path.exists(tmp_audio_file_path):
                    os.remove(tmp_audio_file_path)

            if features is None:
                error_msg = f"Feature extraction returned None for segment {start}-{end}."
                self.logger.error(error_msg)
                self.load_error.emit(error_msg)
                self.current_time += self.segment_duration
                continue

            # Initialize prediction variables
            cnn_predictions = None
            lstm_predictions = None

            # CNN prediction
            if cnn_session is not None:
                try:
                    cnn_pred_class, cnn_probs = predict_cnn(features, cnn_session)
                    cnn_predictions = cnn_probs
                except Exception as e:
                    error_msg = f"Error in CNN prediction for segment {start}-{end}: {e}"
                    self.logger.error(error_msg)
                    self.load_error.emit(error_msg)

            # LSTM prediction
            if lstm_session is not None:
                try:
                    lstm_pred_class, lstm_probs = predict_lstm(features, lstm_session)
                    lstm_predictions = lstm_probs
                except Exception as e:
                    error_msg = f"Error in LSTM prediction for segment {start}-{end}: {e}"
                    self.logger.error(error_msg)
                    self.load_error.emit(error_msg)

            # Ensemble predictions
            if cnn_predictions is not None and lstm_predictions is not None:
                ensemble_probs = ensemble_predictions(cnn_predictions, lstm_predictions)
                if ensemble_probs is not None:
                    ensemble_class_id = np.argmax(ensemble_probs)
                    ensemble_predicted_class = self.le.inverse_transform([ensemble_class_id])[0]

                    self.logger.info(f"Segment Prediction (Ensemble): {ensemble_predicted_class}")
                    self.detected_audio.emit(ensemble_predicted_class, start, end)
            elif cnn_predictions is not None:
                class_id = np.argmax(cnn_predictions)
                predicted_class = self.le.inverse_transform([class_id])[0]
                self.logger.info(f"Segment Prediction (CNN): {predicted_class}")
                self.detected_audio.emit(predicted_class, start, end)
            elif lstm_predictions is not None:
                class_id = np.argmax(lstm_predictions)
                predicted_class = self.le.inverse_transform([class_id])[0]
                self.logger.info(f"Segment Prediction (LSTM): {predicted_class}")
                self.detected_audio.emit(predicted_class, start, end)
            else:
                self.logger.info(f"Segment {start}-{end}: No predictions available.")

            self.current_time += self.segment_duration
            time.sleep(self.segment_duration)  # Sync with video playback time

        # Release resources
        video.close()
        self.logger.info("AudioThread has finished processing.")

    def stop(self):
        """
        Stops the thread gracefully.
        """
        self._run_flag = False
        self.wait()

# --------------------- Video Processing Thread --------------------- #

class VideoThread(QThread):
    """
    Thread to handle video processing and frame updates.
    Emits signals to update the GUI with original and predicted frames, slider values, and video completion.
    """
    change_pixmap_original = pyqtSignal(np.ndarray)
    change_pixmap_predicted = pyqtSignal(np.ndarray)
    update_slider = pyqtSignal(int, int)  # current_frame, total_frames
    video_finished = pyqtSignal()

    def __init__(self, video_file, region3, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.region3 = region3
        self._run_flag = True
        self.paused = False
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30  # Default FPS
        self.seek_requested = False
        self.seek_frame = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bounding_boxes = []  # List of bounding boxes from Gemini

    def run(self):
        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            self.logger.error("Cannot open video file.")
            self.video_finished.emit()
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # Fallback FPS
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
                    original = frame.copy()

                    # If there are bounding boxes from Gemini, draw them
                    if self.bounding_boxes:
                        for bbox in self.bounding_boxes:
                            x_center, y_center, width, height = bbox
                            x1 = int(x_center - width / 2)
                            y1 = int(y_center - height / 2)
                            x2 = int(x_center + width / 2)
                            y2 = int(y_center + height / 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "Ambulance", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        # Clear bounding boxes after drawing
                        self.bounding_boxes = []

                    # Emit processed frames
                    self.change_pixmap_original.emit(original)
                    self.change_pixmap_predicted.emit(frame)

                    self.current_frame += 1
                    self.update_slider.emit(self.current_frame, self.total_frames)

                    # Time delay between frames
                    delay = int(1000 / self.fps)
                    self.msleep(delay)
                else:
                    break
            else:
                self.msleep(100)  # Check again after 100ms if paused

        self.cap.release()
        self.video_finished.emit()
        self.logger.info("VideoThread has finished processing.")

    def add_bounding_boxes(self, bboxes):
        """
        Adds bounding boxes detected by Gemini to be drawn on the video frames.
        """
        self.bounding_boxes.extend(bboxes)
        self.logger.info(f"Added {len(bboxes)} bounding boxes from Gemini.")

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

# --------------------- Main Application Window --------------------- #

class MainWindow(QMainWindow):
    """
    The main application window that handles file selection, video playback, and integrates all components.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Traffic Management Application")
        self.resize(1600, 900)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize trackers
        self.violation_tracker = ViolationTracker()
        self.initUI()
        self.video_thread = None
        self.audio_thread = None
        self.gemini_thread = None
        self.region1 = []
        self.region2 = []
        self.region3 = []

        # Initialize traffic light window
        self.traffic_light_window = TrafficLightWindow()
        self.traffic_light_window.show()
        self.traffic_light_window.state_changed.connect(self.handle_traffic_light_state_changed)
        self.traffic_light_window.checkbox_no_parking.stateChanged.connect(self.handle_no_parking_changed)

        # Apply modern stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
                color: #000000;
            }
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #AAAAAA;
                padding: 8px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #CCCCCC;
            }
            QLabel {
                color: #000000;
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #E0E0E0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #555555;
                border: 1px solid #5c5c5c;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)


    def initUI(self):
        """
        Sets up the user interface components.
        """
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        video_layout = QHBoxLayout()
        control_layout = QHBoxLayout()
        slider_layout = QHBoxLayout()

        # Buttons
        self.btn_select = QPushButton("Select File")
        self.btn_select.clicked.connect(self.select_file)

        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.play_video)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_video)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_video)

        self.btn_prev = QPushButton("⏪ Previous")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self.prev_frame)

        self.btn_next = QPushButton("Next ⏩")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.next_frame)

        # Add buttons to button layout
        button_layout.addWidget(self.btn_select)
        button_layout.addStretch()

        # Video labels
        self.label_original = QLabel("Original")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("border: 2px solid #AAAAAA; border-radius: 10px;")
        self.label_original.setMinimumSize(600, 400)

        self.label_predicted = QLabel("Predicted")
        self.label_predicted.setAlignment(Qt.AlignCenter)
        self.label_predicted.setStyleSheet("border: 2px solid #AAAAAA; border-radius: 10px;")
        self.label_predicted.setMinimumSize(600, 400)

        # Add video labels to video layout
        video_layout.addWidget(self.label_original)
        video_layout.addWidget(self.label_predicted)

        # Control buttons
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_next)
        control_layout.addStretch()

        # Slider and time labels
        self.label_start_time = QLabel("00:00")
        self.label_end_time = QLabel("00:00")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider_moving = False

        slider_layout.addWidget(self.label_start_time)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.label_end_time)

        # Add all layouts to main layout
        main_layout.addLayout(button_layout)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(slider_layout)

        central_widget.setLayout(main_layout)

    def select_file(self):
        """
        Opens a file dialog for the user to select an image or video file.
        """
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video or Image",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)",
            options=options
        )
        if filepath:
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                self.display_image(filepath)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                self.handle_video_selection(filepath)
            else:
                QMessageBox.warning(self, "Warning", "Unsupported file format.")
                self.logger.warning(f"Unsupported file format selected: {ext}")

    def handle_video_selection(self, filepath):
        """
        Handles the selection and processing of a video file, including region selection.
        """
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot open video.")
            self.logger.error("Cannot open video file.")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(self, "Error", "Cannot read the first frame of the video.")
            self.logger.error("Cannot read the first frame of the video.")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        region_window = RegionSelectionWindow(frame)
        result = region_window.exec_()
        if result == QDialog.Accepted:
            # Retrieve regions from region_window
            self.region1 = region_window.region1
            self.region2 = region_window.region2
            self.region3 = region_window.region3
            if not (self.region1 and self.region2 and self.region3):
                QMessageBox.warning(self, "Error", "Not all regions have been selected.")
                self.logger.warning("Incomplete region selection.")
                return
            else:
                # Proceed to display video
                self.display_video(filepath)
        else:
            # User canceled region selection
            self.logger.info("Region selection canceled by user.")

    def display_image(self, filepath):
        """
        Displays the selected image and its predictions.
        """
        # Stop any running video threads
        self.stop_video()

        image = cv2.imread(filepath)
        if image is None:
            QMessageBox.warning(self, "Error", "Cannot read image.")
            self.logger.error(f"Cannot read image file: {filepath}")
            return

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        qt_image = self.convert_cv_qt(image_rgb)

        # Display on QLabel
        self.label_original.setPixmap(qt_image)
        self.label_predicted.clear()
        self.label_predicted.setText("Predicted")

        # Disable video control buttons
        self.set_video_controls_enabled(False)
        self.slider.setValue(0)
        self.label_start_time.setText("00:00")
        self.label_end_time.setText("00:00")

        self.logger.info(f"Displayed image: {filepath}")

    def display_video(self, filepath):
        """
        Initializes and starts video and audio threads, and sets up object counters.
        """
        # Stop any running video threads
        self.stop_video()

        self.video_file = filepath

        # Initialize AudioThread
        self.audio_thread = AudioThread(filepath, self.violation_tracker)
        self.audio_thread.detected_audio.connect(self.print_audio_detection)
        self.audio_thread.load_error.connect(self.handle_load_error)
        self.audio_thread.start()

        # Initialize QMediaPlayer for audio playback
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filepath)))
        self.media_player.play()

        # Initialize and start VideoThread
        self.video_thread = VideoThread(
            filepath, 
            self.region3
        )
        self.video_thread.change_pixmap_original.connect(self.update_original_frame)
        self.video_thread.change_pixmap_predicted.connect(self.update_predicted_frame)
        self.video_thread.update_slider.connect(self.update_slider_value)
        self.video_thread.video_finished.connect(self.video_finished)
        self.video_thread.start()

        # Enable video control buttons
        self.set_video_controls_enabled(True)
        self.slider.setMaximum(self.video_thread.total_frames)

        # Update time labels
        total_seconds = int(self.video_thread.total_frames / self.video_thread.fps)
        self.label_end_time.setText(self.format_time(total_seconds))

        self.logger.info(f"Started video playback: {filepath}")

    def set_video_controls_enabled(self, enabled):
        """
        Enables or disables video control buttons.
        """
        self.btn_play.setEnabled(enabled)
        self.btn_pause.setEnabled(enabled)
        self.btn_stop.setEnabled(enabled)
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def convert_cv_qt(self, cv_img):
        """
        Converts an OpenCV image to QPixmap for display in QLabel.
        """
        try:
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            qt_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            # Scale pixmap to fit the label while keeping aspect ratio
            return pixmap.scaled(
                self.label_original.width(),
                self.label_original.height(),
                Qt.KeepAspectRatio
            )
        except Exception as e:
            self.logger.error(f"Failed to convert image for display: {e}")
            return QPixmap()

    def play_video(self):
        """
        Resumes video playback.
        """
        if self.video_thread:
            self.video_thread.resume()
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.play()
        if self.audio_thread and not self.audio_thread.isRunning():
            self.audio_thread.start()
        self.logger.info("Video playback resumed.")

    def pause_video(self):
        """
        Pauses video playback.
        """
        if self.video_thread:
            self.video_thread.pause()
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.pause()
        # Note: AudioThread does not have a pause method; implement if necessary
        self.logger.info("Video playback paused.")

    def stop_video(self):
        """
        Stops video playback and associated threads.
        """
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
            self.logger.info("VideoThread stopped.")
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread = None
            self.logger.info("AudioThread stopped.")
        # Stop media player
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.stop()
            self.media_player = None
            self.logger.info("Media player stopped.")
        # Clear QLabels
        self.label_original.clear()
        self.label_predicted.clear()
        self.label_original.setText("Original")
        self.label_predicted.setText("Predicted")
        # Disable video control buttons
        self.set_video_controls_enabled(False)
        self.slider.setValue(0)
        self.label_start_time.setText("00:00")
        self.label_end_time.setText("00:00")
        self.logger.info("Video playback stopped and UI reset.")

    def video_finished(self):
        """
        Handles the event when video playback finishes.
        """
        QMessageBox.information(self, "Notification", "Video has ended.")
        self.stop_video()
        self.logger.info("Video playback finished.")

    def prev_frame(self):
        """
        Seeks to the previous frame (rewinds 1 second).
        """
        if self.video_thread:
            target_frame = max(self.video_thread.current_frame - int(self.video_thread.fps), 0)  # Rewind 1 second
            self.video_thread.seek(target_frame)
            if self.audio_thread:
                self.audio_thread.current_time = max(self.audio_thread.current_time - 1, 0)
            self.logger.info(f"Seeked to frame {target_frame}.")

    def next_frame(self):
        """
        Seeks to the next frame (forwards 1 second).
        """
        if self.video_thread:
            target_frame = min(self.video_thread.current_frame + int(self.video_thread.fps), self.video_thread.total_frames - 1)  # Forward 1 second
            self.video_thread.seek(target_frame)
            if self.audio_thread:
                self.audio_thread.current_time = min(self.audio_thread.current_time + 1, self.audio_thread.duration)
            self.logger.info(f"Seeked to frame {target_frame}.")

    def update_slider_value(self, current_frame, total_frames):
        """
        Updates the slider position and start time label based on the current frame.
        """
        if not self.slider_moving:
            self.slider.setMaximum(total_frames)
            self.slider.setValue(current_frame)
            current_seconds = int(current_frame / self.video_thread.fps)
            self.label_start_time.setText(self.format_time(current_seconds))

    def slider_pressed(self):
        """
        Sets the slider moving flag when the user starts moving the slider.
        """
        self.slider_moving = True
        self.logger.info("Slider pressed.")

    def slider_released(self):
        """
        Seeks to the slider position when the user releases the slider.
        """
        if self.video_thread:
            target_frame = self.slider.value()
            self.video_thread.seek(target_frame)
            if self.audio_thread:
                self.audio_thread.current_time = target_frame / self.video_thread.fps
            self.logger.info(f"Slider released. Seeking to frame {target_frame}.")
        self.slider_moving = False

    def format_time(self, seconds):
        """
        Formats seconds into MM:SS format.
        """
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"

    def print_audio_detection(self, predicted_class, start_time, end_time):
        """
        Handles the detected audio class by emitting signals or triggering actions.
        """
        time_str = f"{self.format_time(int(start_time))} - {self.format_time(int(end_time))}"
        self.logger.info(f"[{time_str}] {predicted_class}")
        if predicted_class.lower() == 'ambulance':
            # Start Gemini detection process if not already running
            if not self.gemini_thread or not self.gemini_thread.isRunning():
                self.gemini_thread = GeminiDetectionThread(
                    self.video_file,
                    self.region3,
                    self.video_thread.current_frame,
                    self.video_thread.fps
                )
                self.gemini_thread.bounding_boxes_detected.connect(self.handle_gemini_detection)
                self.gemini_thread.start()
                self.logger.info("GeminiDetectionThread started.")

    def handle_load_error(self, error_message):
        """
        Handles errors during model or audio loading by displaying warnings.
        """
        QMessageBox.warning(self, "Model/Audio Load Error", error_message)
        self.logger.error(f"Load error: {error_message}")

    def handle_traffic_light_state_changed(self, state):
        """
        Updates the traffic light state and handles any related logic.
        """
        # Placeholder for handling traffic light state changes if needed
        self.logger.info(f"Traffic light state changed to {state}")

    def handle_no_parking_changed(self, state):
        """
        Updates based on the "No Parking" checkbox state.
        """
        # Placeholder for handling "No Parking" checkbox changes if needed
        no_parking = self.traffic_light_window.is_no_parking_checked()
        self.logger.info(f"No Parking checkbox changed to {no_parking}")

    def handle_gemini_detection(self, detected, bounding_boxes):
        """
        Handles the result from Gemini detection by updating the video thread to draw bounding boxes.
        """
        if detected and bounding_boxes:
            # Add bounding boxes to VideoThread
            if self.video_thread:
                self.video_thread.add_bounding_boxes(bounding_boxes)
            self.logger.info("Gemini detected bounding boxes and added to VideoThread.")
        else:
            self.logger.info("Gemini did not detect any bounding boxes.")

    def update_original_frame(self, frame):
        """
        Updates the original frame label with the latest frame.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = self.convert_cv_qt(rgb_image)
        self.label_original.setPixmap(qt_image)

    def update_predicted_frame(self, frame):
        """
        Updates the predicted frame label with the latest frame.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = self.convert_cv_qt(rgb_image)
        self.label_predicted.setPixmap(qt_image)

    def closeEvent(self, event):
        """
        Handles the application closing event by stopping all threads and closing windows.
        """
        # Ensure threads are stopped gracefully
        self.stop_video()
        if self.gemini_thread and self.gemini_thread.isRunning():
            self.gemini_thread.stop()
            self.logger.info("GeminiDetectionThread stopped.")
        # Close traffic light window
        if self.traffic_light_window:
            self.traffic_light_window.close()
            self.logger.info("TrafficLightWindow closed.")
        event.accept()
        self.logger.info("Application closed.")

# --------------------- Main Function --------------------- #

def main():
    """
    The main function to start the application.
    """
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()
