# --------------------- Configuration Management --------------------- #

import os
from dotenv import load_dotenv

# Load environment variables from .env.private
load_dotenv(dotenv_path=".env.private")

class Config:
    """Configuration class to manage environment variables and settings."""

    # Email settings
    FROM_EMAIL = os.getenv("FROM_EMAIL")
    FROM_EMAIL_NICKNAME = "Traffic Management System (TMS)"
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    TO_EMAIL = os.getenv("TO_EMAIL")
    EMAIL_SUBJECT = "Violation Alert"
    SEND_EACH = 20  # Send email every 5 violations

    # Violation Thresholds
    VIOLATION_THRESHOLDS = {
        'large_vehicles': 10,
        'motorcycles': 15,
        'illegal_parking': 5,
        'red_light': 7
    }

    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = "gemini-1.5-flash-8b"

    # Model paths
    MODEL_ROOT = "assets/pretrained"
    CNN_MODEL_PATH = os.path.join(MODEL_ROOT, "CNN_model.onnx")
    LSTM_MODEL_PATH = os.path.join(MODEL_ROOT, "LSTM_model.onnx")
    ULTRALYTICS_MODEL_PATH = os.path.join(MODEL_ROOT, "best.pt")

    # Speed Estimator Model Path
    SPEED_ESTIMATOR_MODEL_PATH = os.path.join(MODEL_ROOT, "best.pt")

    REGIONS_DEFAULT_PATH = "assets/regions.json"

    # Logging settings
    LOG_FILE = "log/system.log"
    LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
    LOG_BACKUP_COUNT = 5

    DETECT_AMBULANCE_PROMPT = (
        "What is the position of the ambulance present in the image? "
        "Output ambulance in JSON format with both object names and positions as a JSON object: "
        '{"ambulance": [x_center, y_center, width, height]}. '
        "Put the answer in a JSON code block, and all numbers in pixel units. Please detect carefully because there may not be an ambulance in the photo. "
        "The bounding box must be wide enough to enclose the entire ambulance (note that in some images, the ambulance may have glare from its lights in dark conditions)."
    )