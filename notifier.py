import os
import logging
import requests

logger = logging.getLogger("cat_door_logger")

try:
    from credentials import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_snapshot(image_path, label: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured — skipping notification.")
        return

    caption = "PREY detected — door locked." if label == "prey" else "No prey — cat is clean."

    try:
        with open(image_path, "rb") as photo:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=15,
            )
        resp.raise_for_status()
        logger.info(f"  Telegram notification sent ({label})")
    except Exception as e:
        logger.warning(f"  Telegram send failed: {e}")
