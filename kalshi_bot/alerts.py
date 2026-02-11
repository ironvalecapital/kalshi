from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


def send_telegram(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    payload = {"chat_id": chat_id, "text": message}
    if context:
        payload["text"] = message + "\n" + str(context)
    try:
        httpx.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload, timeout=10.0)
    except Exception:
        return
