from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class Ledger:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_parent()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _ensure_parent(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                event_id TEXT,
                ts TEXT,
                bid REAL,
                ask REAL,
                mid REAL,
                spread REAL,
                volume REAL,
                open_interest REAL,
                source TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                ts TEXT,
                price REAL,
                count INTEGER,
                raw_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                client_order_id TEXT,
                market_id TEXT,
                side TEXT,
                action TEXT,
                price REAL,
                count INTEGER,
                status TEXT,
                ts TEXT,
                raw_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                order_id TEXT,
                market_id TEXT,
                side TEXT,
                action TEXT,
                price REAL,
                count INTEGER,
                fee REAL,
                pnl REAL,
                ts TEXT,
                raw_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                market_id TEXT,
                strategy TEXT,
                inputs_json TEXT,
                signals_json TEXT,
                expected_edge REAL,
                action TEXT,
                size INTEGER,
                risk_checks_json TEXT,
                order_result_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                market_id TEXT,
                p_yes REAL,
                implied_mid REAL,
                model_source TEXT,
                raw_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                event_type TEXT,
                message TEXT,
                context_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                market_ticker TEXT,
                variant TEXT,
                params_json TEXT,
                metrics_json TEXT,
                pnl REAL,
                sharpe REAL,
                max_dd REAL,
                fill_rate REAL,
                fee_drag REAL,
                spread_drag REAL
            )
            """
        )
        self.conn.commit()

    def record_tick(self, tick: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO ticks (market_id, event_id, ts, bid, ask, mid, spread, volume, open_interest, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tick.get("market_id"),
                tick.get("event_id"),
                tick.get("timestamp"),
                tick.get("bid"),
                tick.get("ask"),
                tick.get("mid"),
                tick.get("spread"),
                tick.get("volume"),
                tick.get("open_interest"),
                tick.get("source", "rest"),
            ),
        )
        self.conn.commit()

    def record_market_trade(self, market_id: str, ts: str, price: float, count: int, raw: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO market_trades (market_id, ts, price, count, raw_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (market_id, ts, price, count, json.dumps(raw)),
        )
        self.conn.commit()

    def record_order(
        self,
        market_id: str,
        side: str,
        action: str,
        price: float,
        count: int,
        response: Dict[str, Any],
    ) -> None:
        order_id = response.get("order", {}).get("order_id") or response.get("order_id")
        client_order_id = response.get("order", {}).get("client_order_id") or response.get("client_order_id")
        status = response.get("order", {}).get("status") or response.get("status")
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO orders (order_id, client_order_id, market_id, side, action, price, count, status, ts, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                client_order_id,
                market_id,
                side,
                action,
                price,
                count,
                status,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(response),
            ),
        )
        self.conn.commit()

    def record_trade(
        self,
        market_id: str,
        side: str,
        action: str,
        price: float,
        count: int,
        fee: float,
        pnl: float,
        response: Dict[str, Any],
    ) -> None:
        trade_id = response.get("trade_id")
        order_id = response.get("order_id")
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO trades (trade_id, order_id, market_id, side, action, price, count, fee, pnl, ts, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                order_id,
                market_id,
                side,
                action,
                price,
                count,
                fee,
                pnl,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(response),
            ),
        )
        self.conn.commit()

    def record_decision(
        self,
        market_id: str,
        strategy: str,
        inputs: Dict[str, Any],
        signals: Dict[str, Any],
        expected_edge: float,
        action: str,
        size: int,
        risk_checks: Dict[str, Any],
        order_result: Dict[str, Any],
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO decisions (ts, market_id, strategy, inputs_json, signals_json, expected_edge, action, size, risk_checks_json, order_result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                market_id,
                strategy,
                json.dumps(inputs),
                json.dumps(signals),
                expected_edge,
                action,
                size,
                json.dumps(risk_checks),
                json.dumps(order_result),
            ),
        )
        self.conn.commit()

    def record_prediction(self, market_id: str, p_yes: float, implied_mid: float, model_source: str, raw: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (ts, market_id, p_yes, implied_mid, model_source, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                market_id,
                p_yes,
                implied_mid,
                model_source,
                json.dumps(raw),
            ),
        )
        self.conn.commit()

    def record_audit(self, event_type: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO audit (ts, event_type, message, context_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                event_type,
                message,
                json.dumps(context or {}),
            ),
        )
        self.conn.commit()

    def record_experiment(
        self,
        market_ticker: str,
        variant: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        pnl: float,
        sharpe: float,
        max_dd: float,
        fill_rate: float,
        fee_drag: float,
        spread_drag: float,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO experiments (ts, market_ticker, variant, params_json, metrics_json, pnl, sharpe, max_dd, fill_rate, fee_drag, spread_drag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                market_ticker,
                variant,
                json.dumps(params),
                json.dumps(metrics),
                pnl,
                sharpe,
                max_dd,
                fill_rate,
                fee_drag,
                spread_drag,
            ),
        )
        self.conn.commit()

    def get_recent_experiments(self, limit: int = 200) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT ts, market_ticker, variant, params_json, metrics_json, pnl, sharpe, max_dd, fill_rate, fee_drag, spread_drag
            FROM experiments ORDER BY ts DESC LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "ts": row[0],
                    "market_ticker": row[1],
                    "variant": row[2],
                    "params": json.loads(row[3]) if row[3] else {},
                    "metrics": json.loads(row[4]) if row[4] else {},
                    "pnl": row[5],
                    "sharpe": row[6],
                    "max_dd": row[7],
                    "fill_rate": row[8],
                    "fee_drag": row[9],
                    "spread_drag": row[10],
                }
            )
        return results

    def get_ticks(self, market_id: str, start: str, end: str) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM ticks WHERE market_id = ? AND ts BETWEEN ? AND ? ORDER BY ts ASC
            """,
            (market_id, start, end),
        )
        return cur.fetchall()

    def count_trades_since(self, market_id: str, start: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT COUNT(1) FROM market_trades WHERE market_id = ? AND ts >= ?
            """,
            (market_id, start),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def compute_daily_pnl(self, day: str) -> float:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(SUM(pnl), 0) as pnl FROM trades WHERE ts LIKE ?
            """,
            (f"{day}%",),
        )
        row = cur.fetchone()
        return float(row[0]) if row else 0.0

    def get_trade_pnls(self, limit: int = 1000) -> List[float]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT pnl FROM trades WHERE pnl IS NOT NULL ORDER BY ts DESC LIMIT ?
            """,
            (limit,),
        )
        return [float(r[0]) for r in cur.fetchall() if r[0] is not None]

    def get_trade_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM trades")
        row = cur.fetchone()
        return int(row[0]) if row else 0
