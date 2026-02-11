from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .ledger import Ledger
from .alerts import send_telegram


class AuditLogger:
    def __init__(self, ledger: Ledger, log_path: str) -> None:
        self.ledger = ledger
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, message: str, context: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "message": message,
            "context": context,
        }
        self.ledger.record_audit(event_type, message, context)
        if event_type in {"order", "cancel", "kill", "error"}:
            send_telegram(f"[{event_type}] {message}", context)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
