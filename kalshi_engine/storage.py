from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def _date_partition(ts: Any) -> str:
    if isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return dt.date().isoformat()
    return datetime.now(timezone.utc).date().isoformat()


class ParquetStore:
    """
    Simple append-style parquet writer partitioned by date + market_ticker.
    """

    def __init__(self, root_dir: str = "data/kalshi_parquet") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _append_rows(self, table: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        for row in rows:
            ticker = str(row.get("market_ticker") or row.get("ticker") or "UNKNOWN")
            d = _date_partition(row.get("ts"))
            out_dir = self.root / table / f"date={d}" / f"market_ticker={ticker}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"part-{datetime.now(timezone.utc).strftime('%H%M%S%f')}.parquet"
            pd.DataFrame([row]).to_parquet(out_file, index=False)

    def write_ticks(self, rows: Iterable[Dict[str, Any]]) -> None:
        self._append_rows("ticks", list(rows))

    def write_trades(self, rows: Iterable[Dict[str, Any]]) -> None:
        self._append_rows("trades", list(rows))

    def write_features(self, rows: Iterable[Dict[str, Any]]) -> None:
        self._append_rows("features", list(rows))
