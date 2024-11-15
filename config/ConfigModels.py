import torch
import google.generativeai as genai
from config.Config import Config
from config.config_logger import logger
import onnxruntime as ort

class ConfigModels:
    def __init__(self):
        self.gemini_model = None
        self.pytorch_device = None
        self.onnx_models = {}

    def configure_pytorch(self):
        """
        Configures PyTorch to use MPS, CUDA, or CPU.
        """
        try:
            if torch.backends.mps.is_available():
                self.pytorch_device = torch.device('mps')
                logger.info(f"Using device: MPS ({torch.backends.mps.is_available()})")
            elif torch.cuda.is_available():
                self.pytorch_device = torch.device('cuda')
                cuda_name = torch.cuda.get_device_name(self.pytorch_device)
                logger.info(f"Using device: CUDA ({cuda_name})")
            else:
                self.pytorch_device = torch.device('cpu')
                logger.info("Using device: CPU")
        except Exception as e:
            logger.error(f"Failed to configure PyTorch device: {e}")
            self.pytorch_device = None

    def configure_gemini(self):
        """
        Configures the Gemini API using the provided API key.
        """
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")
            logger.info("Gemini API configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            self.gemini_model = None

    def load_onnx_model(self, model_path, model_name):
        """
        Loads an ONNX model and stores it in the onnx_models dictionary.
        """
        try:
            session = ort.InferenceSession(model_path)
            self.onnx_models[model_name] = session
            logger.info(f"Loaded ONNX model '{model_name}' from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model '{model_name}' from {model_path}: {e}")
            self.onnx_models[model_name] = None

    def get_onnx_model(self, model_name):
        """
        Retrieves the ONNX model session by name.
        """
        return self.onnx_models.get(model_name, None)


# Instantiate and configure the system
config = ConfigModels()

# Configure PyTorch
config.configure_pytorch()

# Configure Gemini API
config.configure_gemini()

# Load ONNX models
config.load_onnx_model(Config.CNN_MODEL_PATH, "cnn_model")
config.load_onnx_model(Config.LSTM_MODEL_PATH, "lstm_model")

# Access configurations and models
device_pt = config.pytorch_device
model_gemini = config.gemini_model
cnn_session = config.get_onnx_model("cnn_model")
lstm_session = config.get_onnx_model("lstm_model")
