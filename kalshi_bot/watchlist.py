from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import BotSettings
from .data_rest import KalshiDataClient
from .market_picker import pick_weather_candidates
from .market_selector import pick_sports_candidates


@dataclass
class WatchlistItem:
    lane: str
    ticker: str
    title: str
    status: str
    spread_cents: Optional[float]
    trades_1h: Optional[int]
    trades_60m: Optional[int]
    depth_top3: Optional[int]
    close_time: Optional[str]
    updated_at: str


def build_watchlist(
    settings: BotSettings,
    data_client: KalshiDataClient,
    top: int = 20,
    include_weather: bool = True,
    include_sports: bool = True,
) -> List[WatchlistItem]:
    now = datetime.now(timezone.utc).isoformat()
    items: List[WatchlistItem] = []

    if include_weather:
        weather = pick_weather_candidates(settings, data_client, top_n=top)
        for c in weather:
            items.append(
                WatchlistItem(
                    lane="weather",
                    ticker=c.ticker,
                    title=c.title or "",
                    status="open",
                    spread_cents=c.spread_yes,
                    trades_1h=c.trades_1h,
                    trades_60m=None,
                    depth_top3=c.depth_yes + c.depth_no,
                    close_time=c.close_time.isoformat() if c.close_time else None,
                    updated_at=now,
                )
            )

    if include_sports:
        sports = pick_sports_candidates(settings, data_client, top_n=top)
        for c in sports:
            items.append(
                WatchlistItem(
                    lane="sports",
                    ticker=c.ticker,
                    title=c.title or "",
                    status=c.status or "open",
                    spread_cents=c.spread_yes,
                    trades_1h=None,
                    trades_60m=c.trades_60m,
                    depth_top3=c.depth_top3,
                    close_time=c.close_time.isoformat() if c.close_time else None,
                    updated_at=now,
                )
            )

    return items[:top]


def watchlist_as_dict(items: List[WatchlistItem]) -> Dict[str, Any]:
    return {"updated_at": datetime.now(timezone.utc).isoformat(), "items": [asdict(i) for i in items]}
