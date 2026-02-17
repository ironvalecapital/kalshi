from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Market(BaseModel):
    ticker: str
    event_ticker: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = None
    close_time: Optional[datetime] = None
    yes_bid: Optional[int] = None
    no_bid: Optional[int] = None


class BookLevel(BaseModel):
    price: int
    size: int


class OrderbookSnapshot(BaseModel):
    market_ticker: str
    ts: datetime = Field(default_factory=datetime.utcnow)
    yes: List[BookLevel] = Field(default_factory=list)
    no: List[BookLevel] = Field(default_factory=list)


class OrderbookDelta(BaseModel):
    market_ticker: str
    ts: datetime = Field(default_factory=datetime.utcnow)
    side: str
    price: int
    size: int


class TradePrint(BaseModel):
    market_ticker: str
    ts: datetime = Field(default_factory=datetime.utcnow)
    yes_price: Optional[int] = None
    no_price: Optional[int] = None
    count: int = 0
    taker_side: Optional[str] = None


class Candle(BaseModel):
    market_ticker: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class RiskFlags(BaseModel):
    drawdown_regime: bool = False
    exposure_cap_hit: bool = False
    thin_book: bool = False
    no_trade_reason: Optional[str] = None


class LLMSnapshot(BaseModel):
    market_ticker: str
    ts: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any]
    liquidity: Dict[str, Any]
    flow: Dict[str, Any]
    model: Dict[str, Any]
    risk: Dict[str, Any]
