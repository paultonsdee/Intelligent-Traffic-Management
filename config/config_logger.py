# --------------------- Logging Configuration --------------------- #

import logging
import os
from logging.handlers import RotatingFileHandler
from config.Config import Config

def setup_logger():
    # remove log file if it exists
    if os.path.exists(Config.LOG_FILE):
        os.remove(Config.LOG_FILE)
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)

    # Configure rotating logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

    handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=Config.LOG_MAX_BYTES,
        backupCount=Config.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False  # Prevent double logging

    logging.getLogger("ultralytics").setLevel(logging.WARNING)  # Suppress Ultralytics logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR) 

    return logger

logger = setup_logger()