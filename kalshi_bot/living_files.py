from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class MemoryPack:
    updated_at: str
    top_markets: List[str]
    chosen_variant: str
    thresholds: Dict[str, Any]
    recent_metrics: Dict[str, Any]
    kill_switch: bool


def _base_dir() -> Path:
    return Path("living_files_kalshi")


def write_operating_rules() -> None:
    path = _base_dir() / "OPERATING_RULES.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(
        "# Operating Rules\n"
        "- Demo by default. Live requires explicit arming.\n"
        "- Kill switch: KALSHI_BOT_KILL=1\n"
        "- Pause on errors, 429 bursts, or stale data.\n"
        "- Maintain audit and ledger integrity.\n"
    )


def append_known_failure(message: str) -> None:
    path = _base_dir() / "KNOWN_FAILURES.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {stamp} {message}\n")


def update_playbook_weather(thresholds: Dict[str, Any]) -> None:
    path = _base_dir() / "PLAYBOOK_WEATHER.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ["# Weather Playbook", "", "Current thresholds:"]
    for k, v in thresholds.items():
        content.append(f"- {k}: {v}")
    path.write_text("\n".join(content))


def write_memory_pack(
    top_markets: List[str],
    chosen_variant: str,
    thresholds: Dict[str, Any],
    recent_metrics: Dict[str, Any],
    kill_switch: bool,
) -> None:
    pack = MemoryPack(
        updated_at=datetime.now(timezone.utc).isoformat(),
        top_markets=top_markets,
        chosen_variant=chosen_variant,
        thresholds=thresholds,
        recent_metrics=recent_metrics,
        kill_switch=kill_switch,
    )
    path = _base_dir() / "MEMORY_PACK.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(pack), indent=2))


def write_live_flag(ok: bool, metrics: Dict[str, Any]) -> None:
    path = Path("reports")
    path.mkdir(parents=True, exist_ok=True)
    flag = path / ("LIVE_READY.flag" if ok else "NOT_LIVE_READY.flag")
    flag.write_text(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "metrics": metrics}, indent=2))
