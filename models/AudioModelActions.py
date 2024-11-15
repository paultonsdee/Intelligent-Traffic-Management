# --------------------- Feature Extraction and Prediction Functions --------------------- #

import numpy as np
import librosa
from config.config_logger import logger

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