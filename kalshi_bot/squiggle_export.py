from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


SPORTS_FOCUS_KEYWORDS = (
    "UFC",
    "MMA",
    "NBA",
    "NCAAB",
    "BASKET",
    "EUROLEAGUE",
    "EURO LEAGUE",
    "EUROBASKET",
    "EURO BASKET",
    "KBL",
    "BSL",
    "ACB",
)


@dataclass
class SquiggleMarketRow:
    ticker: str
    title: str
    event_ticker: str
    implied_yes: float
    yes_bid: Optional[int]
    no_bid: Optional[int]
    spread_cents: Optional[int]
    close_time: Optional[str]



def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None



def _implied_yes(yes_bid: Optional[int], no_bid: Optional[int]) -> float:
    if yes_bid is not None and no_bid is not None:
        yes_ask = 100 - no_bid
        return max(0.0, min(1.0, (yes_bid + yes_ask) / 200.0))
    if yes_bid is not None:
        return max(0.0, min(1.0, yes_bid / 100.0))
    if no_bid is not None:
        return max(0.0, min(1.0, 1.0 - (no_bid / 100.0)))
    return 0.5



def _spread(yes_bid: Optional[int], no_bid: Optional[int]) -> Optional[int]:
    if yes_bid is None or no_bid is None:
        return None
    return int((100 - no_bid) - yes_bid)



def is_focus_sports_market(row: Dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(k, ""))
        for k in ("ticker", "event_ticker", "series_ticker", "title", "subtitle", "category")
    ).upper()
    return any(k in text for k in SPORTS_FOCUS_KEYWORDS)



def normalize_market_row(row: Dict[str, Any]) -> SquiggleMarketRow:
    yb = _as_int(row.get("yes_bid"))
    nb = _as_int(row.get("no_bid"))
    return SquiggleMarketRow(
        ticker=str(row.get("ticker") or ""),
        title=str(row.get("title") or row.get("subtitle") or ""),
        event_ticker=str(row.get("event_ticker") or ""),
        implied_yes=_implied_yes(yb, nb),
        yes_bid=yb,
        no_bid=nb,
        spread_cents=_spread(yb, nb),
        close_time=str(row.get("close_time") or row.get("expiration_time") or "") or None,
    )



def render_squiggle_program(markets: Iterable[SquiggleMarketRow]) -> str:
    rows = list(markets)
    generated = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append('// Squiggle model generated for Kalshi sports/UFC/basketball focus')
    lines.append(f'// generatedAt: {generated}')
    lines.append('')
    lines.append('@name("Global Kelly Safety")')
    lines.append('globalKellySafety = 80%')
    lines.append('')
    lines.append('kellyFraction(modelP, priceP) = {')
    lines.append('  boundedModelP = max(1%, min(99%, modelP))')
    lines.append('  boundedPriceP = max(1%, min(99%, priceP))')
    lines.append('  b = (1 - boundedPriceP) / boundedPriceP')
    lines.append('  q = 1 - boundedModelP')
    lines.append('  max(0%, min(100%, ((b * boundedModelP) - q) / b))')
    lines.append('}')
    lines.append('')
    lines.append('suggestRiskFraction(modelP, impliedP) = {')
    lines.append('  baseFraction = 25% * globalKellySafety')
    lines.append('  k = kellyFraction(modelP, impliedP)')
    lines.append('  min(2%, baseFraction * k)')
    lines.append('}')
    lines.append('')
    lines.append('// Market priors from Kalshi API (sports/UFC/basketball focus)')
    lines.append('markets = [')
    for m in rows:
        spread = 'null' if m.spread_cents is None else str(m.spread_cents)
        close = 'null' if not m.close_time else f'"{m.close_time}"'
        title = m.title.replace('"', "'")
        lines.append(
            f'  {{ticker: "{m.ticker}", eventTicker: "{m.event_ticker}", title: "{title}", impliedYes: {m.implied_yes:.4f}, yesBid: {m.yes_bid if m.yes_bid is not None else "null"}, noBid: {m.no_bid if m.no_bid is not None else "null"}, spreadCents: {spread}, closeTime: {close}}},'
        )
    lines.append(']')
    lines.append('')
    lines.append('// For each market, you can set your model probability and compute edge/risk:')
    lines.append('// edge = modelProb - impliedYes')
    lines.append('// riskFraction = suggestRiskFraction(modelProb, impliedYes)')
    return "\n".join(lines) + "\n"
