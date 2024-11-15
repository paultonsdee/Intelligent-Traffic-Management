# --------------------- Main Application Window --------------------- #

from PyQt5.QtCore import pyqtSlot, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QMessageBox, QDialog
)
import cv2
import os
import numpy as np
from ultralytics import solutions
from tracker.ViolationTracker import ViolationTracker
from config.Config import Config
from gui.RegionSelectionWindow import RegionSelectionWindow
from threads.VideoThread import VideoThread
from threads.AudioThread import AudioThread
from threads.GeminiDetectionThread import GeminiDetectionThread
from threads.RealtimeDetectionThread import RealtimeDetectionThread
from gui.TrafficLightWindow import TrafficLightWindow
from models.ObjectSpeedCounter import ObjectSpeedCounter
from models.ObjectCounter import ObjectCounter
from config.config_logger import logger
from config.ConfigModels import device_pt


class MainWindow(QMainWindow):
    """
    The main application window that handles file selection, video playback, and integrates all components.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Traffic Management Application")
        self.resize(1600, 900)
        self.logger = logger.getChild(self.__class__.__name__)

        # Initialize models and trackers
        self.speed_obj = solutions.SpeedEstimator(
            model=Config.SPEED_ESTIMATOR_MODEL_PATH,
            device=device_pt.type,
            show=False,
            verbose=False
        )
        self.violation_tracker = ViolationTracker()
        self.initUI()
        self.video_thread = None
        self.audio_thread = None
        self.gemini_thread = None
        self.region1 = []
        self.region2 = []
        self.region3 = []
        self.realtime_thread = None  # Initialize the realtime detection thread

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

        self.btn_prev = QPushButton("⏪ Prev")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self.prev_frame)

        self.btn_next = QPushButton("Next ⏩")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.next_frame)

        # Add buttons to button layout
        button_layout.addWidget(self.btn_select)
        button_layout.addStretch()

        # Add the new "Realtime Detection" button
        self.btn_realtime = QPushButton("Realtime Detection")
        self.btn_realtime.clicked.connect(self.start_realtime_detection)
        button_layout.addWidget(self.btn_realtime)

        # Add the "Stop Realtime Detection" button
        self.btn_stop_realtime = QPushButton("Stop Realtime Detection")
        self.btn_stop_realtime.clicked.connect(self.stop_realtime_detection)
        self.btn_stop_realtime.setEnabled(False)
        button_layout.addWidget(self.btn_stop_realtime)

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
        control_layout.addStretch()
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
            "assets/videos_demo",
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

        predicted_image = self.speed_obj.estimate_speed(image.copy())

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predicted_image_rgb = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        qt_image = self.convert_cv_qt(image_rgb)
        qt_predicted = self.convert_cv_qt(predicted_image_rgb)

        # Display on QLabel
        self.label_original.setPixmap(qt_image)
        self.label_predicted.setPixmap(qt_predicted)

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
        self.audio_thread = AudioThread(filepath, self.speed_obj, self.violation_tracker)
        self.audio_thread.detected_audio.connect(self.print_audio_detection)
        self.audio_thread.load_error.connect(self.handle_load_error)
        self.audio_thread.start()

        # Initialize QMediaPlayer for audio playback
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filepath)))
        self.media_player.play()


        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        t = 0.5
        step_frame = 5
        cap.release()
        # Initialize ObjectCounters and SpeedEstimator with selected regions
        try:
            self.counter_region1 = ObjectSpeedCounter(
                fps=fps / step_frame,
                t=20,
                mode='in',  # Set mode to 'in' or 'out' as needed
                show=False,
                model=Config.SPEED_ESTIMATOR_MODEL_PATH,
                region=self.region1,
                device="mps",
                classes=[0],  # Adjust classes as per your model
            )
            self.counter_region2 = ObjectSpeedCounter(
                fps=fps / step_frame,
                t=20,
                mode='in',  # Set mode to 'in' or 'out' as needed
                show=False,
                model=Config.SPEED_ESTIMATOR_MODEL_PATH,
                region=self.region2,
                device="mps",
                classes=[1, 2, 3],  # Adjust classes as per your model
            )
            self.counter_region3_in = ObjectSpeedCounter(
                fps=fps / step_frame,
                t=300,
                mode='in',  # Set mode to 'in' or 'out' as needed
                show=False,
                model=Config.SPEED_ESTIMATOR_MODEL_PATH,
                region=self.region3,
                device="mps",
                classes=[0, 1, 2, 3],  # Adjust classes as per your model
            )
            self.counter_region3_out = ObjectSpeedCounter(
                fps=fps / step_frame,
                t=t,
                mode='out',  # Set mode to 'in' or 'out' as needed
                show=False,
                model=Config.SPEED_ESTIMATOR_MODEL_PATH,
                region=self.region3,
                device="mps",
                classes=[0, 1, 2, 3],  # Adjust classes as per your model
            )
            self.object_counter = ObjectCounter(
                count_mode="in",
                show=False,
                region=self.region3,
                model=Config.SPEED_ESTIMATOR_MODEL_PATH,
                device="mps",
            )
            self.logger.info("Initialized ObjectCounters and SpeedEstimator.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to initialize object counters: {e}")
            self.logger.error(f"Failed to initialize object counters: {e}")
            return

        # Initialize and start VideoThread
        self.video_thread = VideoThread(
            filepath, 
            self.counter_region1, 
            self.counter_region2, 
            self.counter_region3_in,
            self.counter_region3_out,
            self.object_counter, 
            self.violation_tracker,
            self.traffic_light_window,
            self.region3,
        )
        self.video_thread.change_pixmap_original.connect(self.update_original_frame)
        self.video_thread.change_pixmap_predicted.connect(self.update_predicted_frame)
        self.video_thread.update_slider.connect(self.update_slider_value)
        self.video_thread.video_finished.connect(self.video_finished)
        self.video_thread.cars_counted.connect(self.handle_cars_counted)
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

    def handle_cars_counted(self, count, count_type):
        """
        Updates violations based on car counts and their types.
        """
        # Update violation display
        if count_type == "Vận tốc = 0":
            self.violation_tracker.update_violations('illegal_parking', count)
        elif count_type == "Vận tốc != 0":
            self.violation_tracker.update_violations('red_light', count)
        else:
            self.logger.warning(f"Unknown count type: {count_type}")

    def handle_traffic_light_state_changed(self, state):
        """
        Updates the traffic light state and handles counting logic accordingly.
        """
        # Update traffic light state and handle counting logic
        no_parking = self.traffic_light_window.is_no_parking_checked()
        if self.video_thread:
            self.video_thread.update_traffic_light(state, no_parking)
        self.logger.info(f"Traffic light state changed to {state} with no_parking={no_parking}")

    def handle_no_parking_changed(self, state):
        """
        Updates the traffic light based on the "No Parking" checkbox state.
        """
        # Update based on checkbox state
        current_state = self.traffic_light_window.current_state
        no_parking = self.traffic_light_window.is_no_parking_checked()
        if self.video_thread:
            self.video_thread.update_traffic_light(current_state, no_parking)
        self.logger.info(f"No Parking checkbox changed to {no_parking}")

    def handle_gemini_detection(self, detected):
        """
        Handles the result from Gemini detection by updating the traffic light state.
        """
        if detected:
            # Switch traffic light to green
            self.traffic_light_window.set_state('green')
            self.logger.info("Gemini detected bounding boxes. Traffic light set to green.")
        else:
            # Reset traffic light to red
            self.traffic_light_window.set_state('red')
            self.logger.info("Gemini did not detect bounding boxes. Traffic light set to red.")

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
        # Stop realtime detection if running
        self.stop_realtime_detection(dialog=False)
        
        # Stop video playback if running
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

    @pyqtSlot(str, float, float)
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

    def start_realtime_detection(self):
        """
        Starts the realtime detection by accessing the webcam, allowing region selection, and initiating detection.
        """
        # Access the webcam and capture the first frame for region selection
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot access the webcam.")
            self.logger.error("Cannot access the webcam for realtime detection.")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(self, "Error", "Cannot read from the webcam.")
            self.logger.error("Cannot read from the webcam for region selection.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        region_window = RegionSelectionWindow(frame_rgb)
        result = region_window.exec_()
        if result == QDialog.Accepted:
            # Retrieve regions from region_window
            self.region1 = region_window.region1
            self.region2 = region_window.region2
            self.region3 = region_window.region3
            if not (self.region1 and self.region2 and self.region3):
                QMessageBox.warning(self, "Error", "Not all regions have been selected.")
                self.logger.warning("Incomplete region selection for realtime detection.")
                return
            else:
                # Initialize and start the realtime detection thread
                self.realtime_thread = RealtimeDetectionThread(
                    self.region1, 
                    self.region2, 
                    self.region3
                )
                self.realtime_thread.change_pixmap_original.connect(self.update_original_frame_realtime)
                self.realtime_thread.change_pixmap_predicted.connect(self.update_predicted_frame_realtime)
                self.realtime_thread.detection_error.connect(self.handle_realtime_detection_error)
                self.realtime_thread.start()

                # Enable the stop button and disable other controls
                self.btn_stop_realtime.setEnabled(True)
                self.btn_select.setEnabled(False)
                self.btn_realtime.setEnabled(False)

                QMessageBox.information(self, "Realtime Detection", "Realtime detection started.")
                self.logger.info("Realtime detection started.")
        else:
            self.logger.info("Realtime detection region selection canceled by user.")

    @pyqtSlot(np.ndarray)
    def update_original_frame_realtime(self, frame):
        """
        Updates the original frame label with the latest frame from the camera.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = self.convert_cv_qt_realtime(rgb_image, self.label_original)
        self.label_original.setPixmap(qt_image)

    @pyqtSlot(np.ndarray)
    def update_predicted_frame_realtime(self, frame):
        """
        Updates the predicted frame label with the latest detected frame from the camera.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = self.convert_cv_qt_realtime(rgb_image, self.label_predicted)
        self.label_predicted.setPixmap(qt_image)

    def convert_cv_qt_realtime(self, cv_img, label):
        """
        Converts an OpenCV image to QPixmap for display in QLabel, tailored for realtime detection.
        """
        try:
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            qt_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            # Scale pixmap to fit the label while keeping aspect ratio
            return pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio
            )
        except Exception as e:
            self.logger.error(f"Failed to convert image for realtime display: {e}")
            return QPixmap()

    def handle_realtime_detection_error(self, error_message):
        """
        Handles errors during realtime detection by displaying warnings and stopping the detection.
        """
        QMessageBox.warning(self, "Realtime Detection Error", error_message)
        self.logger.error(f"Realtime detection error: {error_message}")
        self.stop_realtime_detection()

    def stop_realtime_detection(self, dialog=True):
        """
        Stops the realtime detection thread and re-enables controls.
        """
        if self.realtime_thread and self.realtime_thread.isRunning():
            self.realtime_thread.stop()
            self.realtime_thread = None
            self.logger.info("RealtimeDetectionThread stopped.")
        
        # Re-enable other controls
        self.btn_stop_realtime.setEnabled(False)
        self.btn_select.setEnabled(True)
        self.btn_realtime.setEnabled(True)

        # Clear the display labels
        self.label_original.clear()
        self.label_predicted.clear()
        self.label_original.setText("Original")
        self.label_predicted.setText("Predicted")

        if dialog:
            QMessageBox.information(self, "Realtime Detection", "Realtime detection stopped.")
        self.logger.info("Realtime detection stopped.")
