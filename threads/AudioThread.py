# --------------------- Audio Processing Thread --------------------- #

from PyQt5.QtCore import QThread, pyqtSignal
from config.Config import Config
from config.config_logger import logger
from models.AudioModelActions import features_extractor, predict_cnn, predict_lstm, ensemble_predictions
from moviepy.editor import VideoFileClip
from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np
import os
import tempfile
import time
from config.ConfigModels import cnn_session, lstm_session



class AudioThread(QThread):
    """
    Thread to handle audio processing and predictions.
    Emits detected audio class and handles load errors.
    """
    detected_audio = pyqtSignal(str, float, float)  # predicted_class, start_time, end_time
    load_error = pyqtSignal(str)

    def __init__(self, video_file, speed_obj, violation_tracker, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.speed_obj = speed_obj
        self.violation_tracker = violation_tracker
        self._run_flag = True
        self.segment_duration = 3.0  # Duration of each audio segment in seconds
        self.current_time = 0.0
        self.logger = logger.getChild(self.__class__.__name__)

    def run(self):
        # Initialize LabelEncoder
        self.le = LabelEncoder()
        self.le.fit(['ambulance', 'firetruck', 'traffic'])

        # Load video and extract audio
        try:
            video = VideoFileClip(self.video_file)
            self.audio = video.audio
            if self.audio is None:  # Check if audio extraction failed
                raise ValueError("Audio track could not be extracted from video.")
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

            tmp_audio_file_path = None

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