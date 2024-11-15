import sys
import os
import cv2
import numpy as np
import tempfile
import librosa
import time
import tensorflow as tf
from ultralytics import solutions
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from moviepy.editor import VideoFileClip
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QWidget, QHBoxLayout, QVBoxLayout, QMessageBox, QSlider, QDialog, QCheckBox
)
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QResizeEvent
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

class RegionSelectionWindow(QDialog):
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

    def setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        # Canvas to display image
        self.canvas = QtWidgets.QLabel(self)
        self.canvas.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.layout.addWidget(self.canvas)
        # Convert image to QImage and display
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qt_image)
        self.canvas.setPixmap(self.pixmap)
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_save_region1 = QtWidgets.QPushButton("Save to Region 1")
        self.btn_save_region1.clicked.connect(self.save_region1)
        button_layout.addWidget(self.btn_save_region1)

        self.btn_save_region2 = QtWidgets.QPushButton("Save to Region 2")
        self.btn_save_region2.clicked.connect(self.save_region2)
        button_layout.addWidget(self.btn_save_region2)

        self.btn_save_region3 = QtWidgets.QPushButton("Save to Region 3")
        self.btn_save_region3.clicked.connect(self.save_region3)
        button_layout.addWidget(self.btn_save_region3)

        self.btn_reset = QtWidgets.QPushButton("Reset Selection")
        self.btn_reset.clicked.connect(self.reset_selection)
        button_layout.addWidget(self.btn_reset)

        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(self.btn_ok)

        self.layout.addLayout(button_layout)

        # Mouse event
        self.canvas.mousePressEvent = self.on_canvas_click

    def initialize_properties(self):
        self.current_polygon = []
        self.polygons = []

    def on_canvas_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.current_polygon.append((x, y))
        painter = QtGui.QPainter(self.canvas.pixmap())
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 5))
        painter.drawEllipse(x - 3, y - 3, 6, 6)
        painter.end()
        self.canvas.update()
        if len(self.current_polygon) >= 2:
            self.draw_polygon(self.current_polygon)

    def draw_polygon(self, polygon):
        painter = QtGui.QPainter(self.canvas.pixmap())
        painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
        painter.drawPixmap(0, 0, self.pixmap)
        for i in range(len(polygon) - 1):
            p1 = QtCore.QPoint(polygon[i][0], polygon[i][1])
            p2 = QtCore.QPoint(polygon[i + 1][0], polygon[i + 1][1])
            painter.drawLine(p1, p2)
        painter.end()
        self.canvas.update()

    def save_region1(self):
        if len(self.current_polygon) < 3:
            QMessageBox.warning(self, "Cảnh Báo", "Vùng cần ít nhất 3 điểm để tạo đa giác.")
            return
        self.region1 = self.current_polygon.copy()
        QMessageBox.information(self, "Saved", "Region 1 saved")
        self.reset_selection()

    def save_region2(self):
        if len(self.current_polygon) < 3:
            QMessageBox.warning(self, "Cảnh Báo", "Vùng cần ít nhất 3 điểm để tạo đa giác.")
            return
        self.region2 = self.current_polygon.copy()
        QMessageBox.information(self, "Saved", "Region 2 saved")
        self.reset_selection()

    def save_region3(self):
        if len(self.current_polygon) < 3:
            QMessageBox.warning(self, "Cảnh Báo", "Vùng cần ít nhất 3 điểm để tạo đa giác.")
            return
        self.region3 = self.current_polygon.copy()
        QMessageBox.information(self, "Saved", "Region 3 saved")
        self.reset_selection()

    def reset_selection(self):
        self.current_polygon.clear()
        # Redraw the image
        self.canvas.setPixmap(self.pixmap.copy())
        self.canvas.update()

    def accept(self):
        # Check that all regions are set
        if not self.region1 or not self.region2 or not self.region3:
            QMessageBox.warning(self, "Lỗi", "Bạn chưa chọn đủ các vùng.")
            return
        super().accept()

# Cấu hình TensorFlow để sử dụng GPU hoặc MPS nếu có
def configure_tensorflow():
    physical_devices = tf.config.list_physical_devices()
    print("Available physical devices:")
    for device in physical_devices:
        print(device)
    
    mps_devices = tf.config.list_physical_devices('MPS')
    gpus = tf.config.list_physical_devices('GPU')
    
    if mps_devices:
        try:
            for device in mps_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"TensorFlow is using MPS: {[device.name for device in mps_devices]}")
            return 'MPS'
        except Exception as e:
            print(f"Error configuring TensorFlow for MPS: {e}")
    
    elif gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow is using GPU: {[gpu.name for gpu in gpus]}")
            return 'GPU'
        except Exception as e:
            print(f"Error configuring TensorFlow for GPU: {e}")
    
    else:
        print("TensorFlow is using CPU")
        return 'CPU'

device_tf = configure_tensorflow()

# Hàm trích xuất đặc trưng từ file âm thanh
def features_extractor(audio, sample_rate):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Hàm dự đoán sử dụng model CNN
def predict_cnn(features, model):
    # Reshape features để phù hợp với input của model
    features = features.reshape(1, -1, 1)
    
    # Dự đoán
    predictions = model.predict(features)
    
    # Lấy class có xác suất cao nhất
    predicted_class = np.argmax(predictions, axis=1)
    
    return predicted_class[0], predictions[0]

# Hàm dự đoán sử dụng model LSTM
def predict_lstm(features, model):
    # Reshape features để phù hợp với input của model
    features = features.reshape(1, -1, 80)
    
    # Dự đoán
    predictions = model.predict(features)
    
    # Lấy class có xác suất cao nhất
    predicted_class = np.argmax(predictions, axis=1)
    
    return predicted_class[0], predictions[0]

def ensemble_predictions(cnn_probs, lstm_probs):
    ensemble_probs = (cnn_probs + lstm_probs) / 2  # Trung bình cộng xác suất
    return ensemble_probs

class AudioThread(QThread):
    detected_audio = pyqtSignal(str, float, float)  # predicted_class, start_time, end_time
    load_error = pyqtSignal(str)

    def __init__(self, video_file, speed_obj, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.speed_obj = speed_obj
        self._run_flag = True
        self.segment_duration = 3.0  # Độ dài mỗi đoạn (giây)
        self.current_time = 0.0

    def run(self):
        # Khởi tạo LabelEncoder với các class
        self.le = LabelEncoder()
        self.le.fit(['ambulance', 'firetruck', 'traffic'])

        # Load models
        try:
            with tf.device('/MPS:0') if device_tf == 'MPS' else (
                tf.device('/GPU:0') if device_tf == 'GPU' else tf.device('/CPU:0')):
                self.cnn_model = load_model('/Users/phatvu/Downloads/CNN_Model')  # Thay đổi đường dẫn nếu cần
            print("Loading CNN model successfully!")
        except Exception as e:
            error_msg = f"Cannot load CNN model! Error: {e}"
            print(error_msg)
            self.load_error.emit(error_msg)
            self.cnn_model = None

        try:
            with tf.device('/MPS:0') if device_tf == 'MPS' else (
                tf.device('/GPU:0') if device_tf == 'GPU' else tf.device('/CPU:0')):
                self.lstm_model = load_model('/Users/phatvu/Downloads/LSTM')  # Thay đổi đường dẫn nếu cần
            print("Loading LSTM model successfully!")
        except Exception as e:
            error_msg = f"Cannot load LSTM model! Error: {e}"
            print(error_msg)
            self.load_error.emit(error_msg)
            self.lstm_model = None

        # Tải video và trích xuất âm thanh
        try:
            video = VideoFileClip(self.video_file)
            self.audio = video.audio
            self.sample_rate = 22050  # Mặc định của librosa
            self.duration = video.duration  # đơn vị: giây
            print(f"Video duration: {self.duration} seconds")
        except Exception as e:
            error_msg = f"Cannot load video file! Error: {e}"
            print(error_msg)
            self.load_error.emit(error_msg)
            return

        # Process audio segments every 3 seconds
        while self._run_flag and self.current_time < self.duration:
            start = self.current_time
            end = min(start + self.segment_duration, self.duration)
            print(f"\nProcessing audio segment: {start} - {end} seconds")

            try:
                # Tạo tệp âm thanh tạm thời cho đoạn này
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                    # Ghi âm thanh đoạn vào tệp tạm
                    audio_segment = self.audio.subclip(start, end)
                    audio_segment.write_audiofile(tmp_audio_file.name, fps=self.sample_rate, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
                    tmp_audio_file_path = tmp_audio_file.name

                # Trích xuất đặc trưng từ đoạn âm thanh
                y, sr = librosa.load(tmp_audio_file_path, sr=self.sample_rate)
                features = features_extractor(y, sr)
                
                # Xóa tệp âm thanh tạm sau khi đã trích xuất đặc trưng
                os.remove(tmp_audio_file_path)

            except Exception as e:
                error_msg = f"Error processing audio segment {start}-{end}: {e}"
                print(error_msg)
                self.load_error.emit(error_msg)
                self.current_time += self.segment_duration
                continue

            # Khởi tạo biến lưu kết quả dự đoán
            cnn_predictions = None
            lstm_predictions = None

            # Thực hiện dự đoán với CNN model
            if self.cnn_model is not None:
                try:
                    with tf.device('/MPS:0') if device_tf == 'MPS' else (
                        tf.device('/GPU:0') if device_tf == 'GPU' else tf.device('/CPU:0')):
                        _, probabilities = predict_cnn(features, self.cnn_model)
                    cnn_predictions = probabilities
                except Exception as e:
                    error_msg = f"Error in CNN prediction for segment {start}-{end}: {e}"
                    print(error_msg)
                    self.load_error.emit(error_msg)

            # Thực hiện dự đoán với LSTM model
            if self.lstm_model is not None:
                try:
                    with tf.device('/MPS:0') if device_tf == 'MPS' else (
                        tf.device('/GPU:0') if device_tf == 'GPU' else tf.device('/CPU:0')):
                        _, probabilities = predict_lstm(features, self.lstm_model)
                    lstm_predictions = probabilities
                except Exception as e:
                    error_msg = f"Error in LSTM prediction for segment {start}-{end}: {e}"
                    print(error_msg)
                    self.load_error.emit(error_msg)

            # Tổng hợp kết quả nếu cả hai model đều dự đoán
            if cnn_predictions is not None and lstm_predictions is not None:
                ensemble_probs = ensemble_predictions(cnn_predictions, lstm_predictions)
                ensemble_class_id = np.argmax(ensemble_probs)
                ensemble_predicted_class = self.le.inverse_transform([ensemble_class_id])[0]
                
                print(f"Segment Prediction (Ensemble): {ensemble_predicted_class}")
                self.detected_audio.emit(ensemble_predicted_class, start, end)
            
            elif cnn_predictions is not None:
                class_id = np.argmax(cnn_predictions)
                predicted_class = self.le.inverse_transform([class_id])[0]
                print(f"Segment Prediction (CNN): {predicted_class}")
                self.detected_audio.emit(predicted_class, start, end)
            
            elif lstm_predictions is not None:
                class_id = np.argmax(lstm_predictions)
                predicted_class = self.le.inverse_transform([class_id])[0]
                print(f"Segment Prediction (LSTM): {predicted_class}")
                self.detected_audio.emit(predicted_class, start, end)
            
            else:
                print(f"Segment {start}-{end}: No predictions available.")
            
            self.current_time += self.segment_duration
            time.sleep(self.segment_duration)  # Sleep để đồng bộ với thời gian phát video

        # Giải phóng tài nguyên
        video.close()

    def stop(self):
        self._run_flag = False
        self.wait()

class TrafficLightWindow(QWidget):
    state_changed = pyqtSignal(str)  # Emits 'green', 'yellow', 'red'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Đèn Giao Thông")
        self.setGeometry(50, 50, 100, 300)
        self.setup_ui()
        self.current_state = 'green'
        self.set_state('green')  # Initialize with green

    def setup_ui(self):
        layout = QVBoxLayout()

        # Nút đèn xanh
        self.btn_green = QPushButton()
        self.btn_green.setFixedSize(50, 50)
        self.btn_green.setStyleSheet("background-color: green; border-radius: 25px;")
        self.btn_green.setEnabled(True)
        self.btn_green.clicked.connect(lambda: self.set_state('green'))

        # Nút đèn vàng
        self.btn_yellow = QPushButton()
        self.btn_yellow.setFixedSize(50, 50)
        self.btn_yellow.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_yellow.setEnabled(True)
        self.btn_yellow.clicked.connect(lambda: self.set_state('yellow'))

        # Nút đèn đỏ
        self.btn_red = QPushButton()
        self.btn_red.setFixedSize(50, 50)
        self.btn_red.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_red.setEnabled(True)
        self.btn_red.clicked.connect(lambda: self.set_state('red'))

        # Hộp kiểm "No Parking"
        self.checkbox_no_parking = QCheckBox("No Parking")

        # Thêm các widget vào layout
        layout.addWidget(self.btn_green, alignment=Qt.AlignCenter)
        layout.addWidget(self.btn_yellow, alignment=Qt.AlignCenter)
        layout.addWidget(self.btn_red, alignment=Qt.AlignCenter)
        layout.addWidget(self.checkbox_no_parking, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def set_state(self, state):
        self.current_state = state
        # Reset màu đèn
        self.btn_green.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_yellow.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_red.setStyleSheet("background-color: grey; border-radius: 25px;")
        # Set màu đèn hiện tại
        if state == 'green':
            self.btn_green.setStyleSheet("background-color: green; border-radius: 25px;")
        elif state == 'yellow':
            self.btn_yellow.setStyleSheet("background-color: yellow; border-radius: 25px;")
        elif state == 'red':
            self.btn_red.setStyleSheet("background-color: red; border-radius: 25px;")
        self.state_changed.emit(state)

    def is_no_parking_checked(self):
        return self.checkbox_no_parking.isChecked()

class VideoThread(QThread):
    change_pixmap_original = pyqtSignal(np.ndarray)
    change_pixmap_predicted = pyqtSignal(np.ndarray)
    update_slider = pyqtSignal(int, int)  # current_frame, total_frames
    video_finished = pyqtSignal()
    cars_counted = pyqtSignal(int, str)  # số xe, loại đếm

    def __init__(self, video_file, speed_estimator, counter_region1, counter_region2, region3, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.speed_estimator = speed_estimator
        self.counter_region1 = counter_region1
        self.counter_region2 = counter_region2
        self.region3 = region3  # Thêm region3 để đếm xe
        self._run_flag = True
        self.paused = False
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30  # Default FPS
        self.seek_requested = False
        self.seek_frame = 0
        # Biến quản lý đếm xe
        self.traffic_light_state = 'green'
        self.no_parking = False
        self.counting = False
        self.count_start_time = None
        self.count_duration = 300  # 5 phút = 300 giây
        self.car_speed_zero_count = 0
        self.car_speed_nonzero_count = 0

    def update_traffic_light(self, state, no_parking):
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
        self.counting = True
        self.count_start_time = time.time()
        self.car_speed_zero_count = 0
        print("Bắt đầu đếm xe với vận tốc = 0 trong 5 phút.")

    def start_counting_nonzero_speed(self):
        self.counting = True
        self.count_start_time = time.time()
        self.car_speed_nonzero_count = 0
        print("Bắt đầu đếm xe với vận tốc != 0.")

    def stop_counting(self):
        if self.counting:
            self.counting = False
            if self.traffic_light_state == 'green' and self.no_parking:
                # Gửi kết quả đếm xe vận tốc = 0
                self.cars_counted.emit(self.car_speed_zero_count, "Vận tốc = 0")
                print(f"Đã đếm {self.car_speed_zero_count} xe với vận tốc = 0 trong 5 phút.")
            elif self.traffic_light_state == 'red':
                # Gửi kết quả đếm xe vận tốc != 0
                self.cars_counted.emit(self.car_speed_nonzero_count, "Vận tốc != 0")
                print(f"Đã đếm {self.car_speed_nonzero_count} xe với vận tốc != 0.")
            self.count_start_time = None

    def run(self):
        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            self.video_finished.emit()
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # Fallback FPS
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.update_slider.emit(self.current_frame, self.total_frames)

        while self._run_flag:
            if not self.paused:
                if self.seek_requested:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                    self.current_frame = self.seek_frame
                    self.seek_requested = False

                ret, frame = self.cap.read()
                if ret:
                    original = frame.copy()

                    # Process frame with counter_region1
                    frame = self.counter_region1.count(frame)

                    # Process frame with counter_region2
                    frame = self.counter_region2.count(frame)

                    # Process frame with speed_estimator
                    frame = self.speed_estimator.estimate_speed(frame)

                    # Emit các frame đã xử lý
                    self.change_pixmap_original.emit(original)
                    self.change_pixmap_predicted.emit(frame)

                    self.current_frame += 1
                    self.update_slider.emit(self.current_frame, self.total_frames)

                    # Kiểm tra và thực hiện đếm xe
                    if self.counting:
                        current_time = time.time()
                        elapsed = current_time - self.count_start_time
                        if self.traffic_light_state == 'green' and self.no_parking:
                            if elapsed <= self.count_duration:
                                # Giả sử bạn có phương thức để xác định vận tốc xe trong region3
                                # Ví dụ: self.speed_estimator.get_car_speeds(frame, self.region3)
                                # Và đếm số xe có vận tốc = 0
                                # Vì get_car_speeds không tồn tại, chúng ta sẽ giả định cách lấy tốc độ
                                car_speeds = self.get_car_speeds(frame)
                                self.car_speed_zero_count += sum(1 for speed in car_speeds if speed == 0)
                            else:
                                self.stop_counting()
                        elif self.traffic_light_state == 'red':
                            if elapsed <= self.count_duration:
                                # Đếm số xe có vận tốc != 0
                                car_speeds = self.get_car_speeds(frame)
                                self.car_speed_nonzero_count += sum(1 for speed in car_speeds if speed != 0)
                            else:
                                self.stop_counting()

                    # Thời gian chờ giữa các khung hình
                    delay = int(1000 / self.fps)
                    self.msleep(delay)
                else:
                    break
            else:
                self.msleep(100)  # Kiểm tra lại sau 100ms nếu đang tạm dừng

        self.cap.release()
        self.video_finished.emit()

    def get_car_speeds(self, frame):
        """
        Giả định phương thức này sẽ trả về danh sách tốc độ của các xe trong frame.
        Bạn cần triển khai phương thức này dựa trên cách SpeedEstimator hoạt động.
        Dưới đây là một ví dụ giả định.
        """
        # Ví dụ: Sử dụng SpeedEstimator để lấy tốc độ
        # Điều này phụ thuộc vào cách SpeedEstimator cung cấp thông tin về tốc độ
        # Nếu SpeedEstimator chỉ trả về frame đã xử lý, bạn cần tự theo dõi vị trí và tính toán tốc độ
        # Dưới đây là một ví dụ đơn giản không thực tế
        # Bạn cần thay thế bằng logic thực tế của bạn
        car_speeds = []  # Danh sách tốc độ của các xe
        # Ví dụ:
        # detections = self.speed_estimator.detect(frame)
        # for detection in detections:
        #     speed = self.speed_estimator.get_speed(detection)
        #     car_speeds.append(speed)
        return car_speeds  # Trả về danh sách tốc độ

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self._run_flag = False
        self.wait()

    def seek(self, frame_number):
        if self.cap:
            frame_number = max(0, min(frame_number, self.total_frames - 1))
            self.seek_frame = frame_number
            self.seek_requested = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ứng Dụng Quản Lý Giao Thông Thông Minh")
        self.resize(1600, 900)
        self.speed_obj = solutions.SpeedEstimator(
            model="/Users/phatvu/Library/CloudStorage/GoogleDrive-phatvu.coder@gmail.com/My Drive/Dev-Drive/Competitions/Competitions-2024/M9-AIoT_Developer_InnoWorks/Intelligence-Traffic-Management-System/best_combined.pt",
            device='mps' if device_tf == 'MPS' else ('cuda' if device_tf == 'GPU' else 'cpu'),
            show=False
        )
        self.initUI()
        self.video_thread = None
        self.audio_thread = None
        self.region1 = []
        self.region2 = []
        self.region3 = []
        # Khởi tạo cửa sổ đèn giao thông
        self.traffic_light_window = TrafficLightWindow()
        self.traffic_light_window.show()
        # Kết nối tín hiệu từ TrafficLightWindow
        self.traffic_light_window.state_changed.connect(self.handle_traffic_light_state_changed)
        self.traffic_light_window.checkbox_no_parking.stateChanged.connect(self.handle_no_parking_changed)

    def initUI(self):
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
        self.btn_select = QPushButton("Chọn Tệp")
        self.btn_select.clicked.connect(self.select_file)

        self.btn_play = QPushButton("Phát")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.play_video)

        self.btn_pause = QPushButton("Tạm Dừng")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_video)

        self.btn_stop = QPushButton("Dừng")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_video)

        self.btn_prev = QPushButton("⏪ Tua Lùi")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self.prev_frame)

        self.btn_next = QPushButton("Tua Tới ⏩")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.next_frame)

        # Add buttons to button layout
        button_layout.addWidget(self.btn_select)
        button_layout.addStretch()

        # Video labels
        self.label_original = QLabel("Original")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("border: 1px solid black;")
        self.label_original.setMinimumSize(600, 400)

        self.label_predicted = QLabel("Predicted")
        self.label_predicted.setAlignment(Qt.AlignCenter)
        self.label_predicted.setStyleSheet("border: 1px solid black;")
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

        # Slider và nhãn thời gian
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
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn Video hoặc Hình Ảnh",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)",
            options=options
        )
        if filepath:
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                self.display_image(filepath)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                # Trước khi hiển thị video, mở cửa sổ chọn vùng
                cap = cv2.VideoCapture(filepath)
                if not cap.isOpened():
                    QMessageBox.warning(self, "Lỗi", "Không thể mở video.")
                    return
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    QMessageBox.warning(self, "Lỗi", "Không thể đọc frame đầu tiên của video.")
                    return
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                region_window = RegionSelectionWindow(frame)
                result = region_window.exec_()
                if result == QDialog.Accepted:
                    # Lấy các vùng từ region_window
                    self.region1 = region_window.region1
                    self.region2 = region_window.region2
                    self.region3 = region_window.region3
                    if not self.region1 or not self.region2 or not self.region3:
                        QMessageBox.warning(self, "Lỗi", "Bạn chưa chọn đủ các vùng.")
                        return
                    else:
                        # Tiến hành hiển thị video
                        self.display_video(filepath)
                else:
                    # Người dùng hủy chọn vùng
                    return
            else:
                QMessageBox.warning(self, "Cảnh Báo", "Định dạng tệp không được hỗ trợ.")

    def display_image(self, filepath):
        # Dừng video nếu đang chạy
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()

        # Nếu có QMediaPlayer đang chạy, dừng nó
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.stop()

        image = cv2.imread(filepath)
        if image is None:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc hình ảnh.")
            return

        predicted_image = self.speed_obj.estimate_speed(image.copy())

        # Chuyển đổi BGR sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)

        # Chuyển đổi thành QImage
        qt_image = self.convert_cv_qt(image)
        qt_predicted = self.convert_cv_qt(predicted_image)

        # Hiển thị trên QLabel
        self.label_original.setPixmap(qt_image)
        self.label_predicted.setPixmap(qt_predicted)

        # Vô hiệu hóa các nút điều khiển video
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.slider.setEnabled(False)
        self.slider.setValue(0)
        self.label_start_time.setText("00:00")
        self.label_end_time.setText("00:00")

    def display_video(self, filepath):
        # Dừng video nếu đang chạy
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()

        # Nếu có QMediaPlayer đang chạy, dừng nó
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.stop()

        # Tạo và khởi chạy AudioThread
        self.audio_thread = AudioThread(filepath, self.speed_obj)
        self.audio_thread.detected_audio.connect(self.print_audio_detection)
        self.audio_thread.load_error.connect(self.handle_load_error)
        self.audio_thread.start()

        # Tạo QMediaPlayer để phát âm thanh
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filepath)))
        self.media_player.play()

        # Tạo các đối tượng ObjectCounter và SpeedEstimator với các vùng đã chọn
        self.counter_region1 = solutions.ObjectCounter(
            model="/Users/phatvu/Library/CloudStorage/GoogleDrive-phatvu.coder@gmail.com/My Drive/Dev-Drive/Competitions/Competitions-2024/M9-AIoT_Developer_InnoWorks/Intelligence-Traffic-Management-System/best_combined.pt",
            region=self.region1,
            classes=[1, 2, 3],
            show=False
        )
        self.counter_region2 = solutions.ObjectCounter(
            model="/Users/phatvu/Library/CloudStorage/GoogleDrive-phatvu.coder@gmail.com/My Drive/Dev-Drive/Competitions/Competitions-2024/M9-AIoT_Developer_InnoWorks/Intelligence-Traffic-Management-System/best_combined.pt",
            region=self.region2,
            classes=[0],
            show=False
        )
        self.speed_estimator = solutions.SpeedEstimator(
            model="/Users/phatvu/Library/CloudStorage/GoogleDrive-phatvu.coder@gmail.com/My Drive/Dev-Drive/Competitions/Competitions-2024/M9-AIoT_Developer_InnoWorks/Intelligence-Traffic-Management-System/best_combined.pt",
            region=self.region3,
            classes=[0, 1, 2, 3],
            show=False
        )

        # Tạo và khởi chạy VideoThread
        self.video_thread = VideoThread(filepath, self.speed_estimator, self.counter_region1, self.counter_region2, self.region3)
        self.video_thread.change_pixmap_original.connect(self.update_original_frame)
        self.video_thread.change_pixmap_predicted.connect(self.update_predicted_frame)
        self.video_thread.update_slider.connect(self.update_slider_value)
        self.video_thread.video_finished.connect(self.video_finished)
        self.video_thread.cars_counted.connect(self.handle_cars_counted)  # Kết nối tín hiệu đếm xe
        self.video_thread.start()

        # Bật các nút điều khiển video
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.slider.setEnabled(True)
        self.slider.setMaximum(self.video_thread.total_frames)

        # Cập nhật nhãn thời gian
        total_seconds = int(self.video_thread.total_frames / self.video_thread.fps)
        self.label_end_time.setText(self.format_time(total_seconds))

    def update_original_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = self.convert_cv_qt(rgb_image)
        self.label_original.setPixmap(qt_image)

    def update_predicted_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = self.convert_cv_qt(rgb_image)
        self.label_predicted.setPixmap(qt_image)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
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

    def play_video(self):
        if self.video_thread:
            self.video_thread.resume()
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.play()
        if self.audio_thread and not self.audio_thread.isRunning():
            self.audio_thread.start()

    def pause_video(self):
        if self.video_thread:
            self.video_thread.pause()
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.pause()
        if self.audio_thread:
            self.audio_thread._run_flag = False

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread = None
        # Nếu có QMediaPlayer đang chạy, dừng nó
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.stop()
            self.media_player = None

        # Xóa hình ảnh trên các QLabel
        self.label_original.clear()
        self.label_predicted.clear()
        self.label_original.setText("Original")
        self.label_predicted.setText("Predicted")
        # Vô hiệu hóa các nút điều khiển video
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.slider.setEnabled(False)
        self.slider.setValue(0)
        self.label_start_time.setText("00:00")
        self.label_end_time.setText("00:00")

    def video_finished(self):
        QMessageBox.information(self, "Thông Báo", "Video đã kết thúc.")
        self.stop_video()

    def prev_frame(self):
        if self.video_thread:
            target_frame = max(self.video_thread.current_frame - int(self.video_thread.fps), 0)  # Tua lùi 1 giây
            self.video_thread.seek(target_frame)
            if self.audio_thread:
                self.audio_thread.current_time = max(self.audio_thread.current_time - 1, 0)  # Điều chỉnh thời gian tương ứng

    def next_frame(self):
        if self.video_thread:
            target_frame = min(self.video_thread.current_frame + int(self.video_thread.fps), self.video_thread.total_frames - 1)  # Tua tới 1 giây
            self.video_thread.seek(target_frame)
            if self.audio_thread:
                self.audio_thread.current_time = min(self.audio_thread.current_time + 1, self.audio_thread.duration)  # Điều chỉnh thời gian tương ứng

    def update_slider_value(self, current_frame, total_frames):
        if not self.slider_moving:
            self.slider.setMaximum(total_frames)
            self.slider.setValue(current_frame)
            current_seconds = int(current_frame / self.video_thread.fps)
            self.label_start_time.setText(self.format_time(current_seconds))

    def slider_pressed(self):
        self.slider_moving = True

    def slider_released(self):
        if self.video_thread:
            target_frame = self.slider.value()
            self.video_thread.seek(target_frame)
            if self.audio_thread:
                self.audio_thread.current_time = target_frame / self.video_thread.fps
        self.slider_moving = False

    def format_time(self, seconds):
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"

    def print_audio_detection(self, predicted_class, start_time, end_time):
        time_str = f"{self.format_time(int(start_time))} - {self.format_time(int(end_time))}"
        print(f"[{time_str}] {predicted_class}")

    def handle_load_error(self, error_message):
        QMessageBox.warning(self, "Lỗi Tải Mô Hình/Âm Thanh", error_message)

    def handle_cars_counted(self, count, count_type):
        QMessageBox.information(self, "Kết Quả Đếm Xe", f"Đã đếm {count} xe với {count_type}.")

    def handle_traffic_light_state_changed(self, state):
        # Cập nhật trạng thái đèn giao thông và hộp kiểm
        no_parking = self.traffic_light_window.is_no_parking_checked()
        if self.video_thread:
            self.video_thread.update_traffic_light(state, no_parking)

    def handle_no_parking_changed(self, state):
        # Khi hộp kiểm "No Parking" thay đổi, cập nhật trạng thái
        current_state = self.traffic_light_window.current_state
        no_parking = self.traffic_light_window.is_no_parking_checked()
        if self.video_thread:
            self.video_thread.update_traffic_light(current_state, no_parking)

    def resizeEvent(self, event: QResizeEvent):
        # Khi cửa sổ thay đổi kích thước, cập nhật lại các pixmap
        if self.label_original.pixmap():
            self.label_original.setPixmap(self.label_original.pixmap().scaled(
                self.label_original.width(),
                self.label_original.height(),
                Qt.KeepAspectRatio
            ))
        if self.label_predicted.pixmap():
            self.label_predicted.setPixmap(self.label_predicted.pixmap().scaled(
                self.label_predicted.width(),
                self.label_predicted.height(),
                Qt.KeepAspectRatio
            ))
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
        # Đóng cửa sổ đèn giao thông
        if self.traffic_light_window:
            self.traffic_light_window.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
