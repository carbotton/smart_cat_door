import logging
import os
from datetime import datetime

log_dir = os.path.join(os.getcwd(), "log")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("cat_door_logger")
logger.setLevel(logging.INFO)

plain_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


class _ColorFormatter(logging.Formatter):
    _GREEN = "\033[92m"
    _RED   = "\033[91m"
    _RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        text = record.getMessage()
        if "DOOR OPENED" in text:
            return f"{self._GREEN}{msg}{self._RESET}"
        if "DOOR CLOSED" in text:
            return f"{self._RED}{msg}{self._RESET}"
        return msg


_color_formatter = _ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")

if not logger.handlers:
    log_filename = datetime.now().strftime("event_log_%Y%m%d.log")
    log_path = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(plain_formatter)   # no color codes in the file

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_color_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
