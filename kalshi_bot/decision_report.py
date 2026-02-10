from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import BotSettings


def write_decision_report(settings: BotSettings, report: Dict[str, Any]) -> Path:
    out_dir = Path(settings.decision_report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"decision_report_{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
