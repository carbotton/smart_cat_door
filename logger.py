import logging
import os
from datetime import datetime

# Setup log directory
log_dir = os.path.join(os.getcwd(), "log")
os.makedirs(log_dir, exist_ok=True)

# Create logger
logger = logging.getLogger("cat_door_logger")
logger.setLevel(logging.INFO)

# Formatter with timestamp
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
log_filename = datetime.now().strftime("event_log_%Y%m%d.log")
log_path = os.path.join(log_dir, log_filename)
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Avoid duplicate handlers when re-importing
if not logger.hasHandlers():
    logger.addHandler(file_handler)
