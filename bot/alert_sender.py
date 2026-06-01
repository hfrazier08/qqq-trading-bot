import logging
import time
from typing import Dict

import requests

from bot.config import CONFIG

logger = logging.getLogger(__name__)


def send_discord_alert(payload: dict) -> bool:
    try:
        resp = requests.post(CONFIG.discord_webhook_url, json=payload, timeout=15)
        if resp.status_code == 429:
            retry_after = int(resp.json().get("retry_after", 1000)) / 1000
            logger.warning(f"Discord rate limited — sleeping {retry_after:.1f}s")
            time.sleep(retry_after + 0.1)
            resp = requests.post(CONFIG.discord_webhook_url, json=payload, timeout=15)
        if resp.status_code not in (200, 204):
            logger.error(f"Discord error {resp.status_code}: {resp.text[:300]}")
            return False
        logger.info("Discord alert sent successfully")
        return True
    except Exception as e:
        logger.error(f"Discord send failed: {e}")
        return False


def send_whatsapp_alert(message: str) -> bool:
    if not all([CONFIG.twilio_sid, CONFIG.twilio_token, CONFIG.twilio_from, CONFIG.twilio_to]):
        logger.debug("Twilio not configured — skipping WhatsApp alert")
        return False
    try:
        from twilio.rest import Client
        client = Client(CONFIG.twilio_sid, CONFIG.twilio_token)
        msg = client.messages.create(
            from_=CONFIG.twilio_from,
            to=CONFIG.twilio_to,
            body=message,
        )
        logger.info(f"WhatsApp sent: {msg.sid}")
        return True
    except Exception as e:
        logger.error(f"WhatsApp send failed: {e}")
        return False


def send_all_alerts(discord_payload: dict, whatsapp_message: str) -> Dict[str, bool]:
    results = {
        "discord": send_discord_alert(discord_payload),
        "whatsapp": send_whatsapp_alert(whatsapp_message),
    }
    return results
