from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np
import pandas as pd

from .btc_regime import fit_btc_regime_detector


@dataclass
class BTCRegimeTrainOutput:
    parquet_path: str
    summary_path: str
    rows: int


def find_latest_polygon_dir(base_dir: str = "data/bitcoin") -> Optional[Path]:
    base = Path(base_dir)
    if not base.exists():
        return None
    cands = sorted([p for p in base.glob("*_polygon") if p.is_dir()], key=lambda p: p.name)
    return cands[-1] if cands else None


def _read_polygon_csv(path: Path, interval_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns from downloader
    keep = [c for c in ["ts_ms", "ts_iso", "open", "high", "low", "close", "volume", "vwap", "transactions"] if c in df.columns]
    df = df[keep].copy()
    df["interval"] = interval_label
    df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce")
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df = df.dropna(subset=["ts_ms", "close"]).copy()
    df["ts_ms"] = df["ts_ms"].astype("int64")
    return df


def merge_polygon_to_parquet(input_dir: str, out_parquet: str) -> int:
    d = Path(input_dir)
    files = {
        "1d": d / "polygon_btcusd_1d.csv",
        "1h": d / "polygon_btcusd_1h_3y.csv",
        "5m": d / "polygon_btcusd_5m_180d.csv",
        "1m": d / "polygon_btcusd_1m_30d.csv",
    }
    parts: List[pd.DataFrame] = []
    for label, path in files.items():
        if path.exists():
            parts.append(_read_polygon_csv(path, label))
    if not parts:
        raise FileNotFoundError(f"No polygon CSV files found in {input_dir}")
    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.sort_values(["ts_ms", "interval"]).drop_duplicates(subset=["ts_ms", "interval"], keep="last")
    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out, index=False)
    return int(len(all_df))


def train_hmm_from_parquet(parquet_path: str, summary_path: str) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    # train on highest-resolution available first
    sub = df[df["interval"] == "1m"].copy()
    if sub.empty:
        sub = df[df["interval"] == "5m"].copy()
    if sub.empty:
        sub = df[df["interval"] == "1h"].copy()
    if sub.empty:
        raise ValueError("No 1m/5m/1h rows available for regime training")

    sub = sub.sort_values("ts_ms")
    close = sub["close"].astype(float).values
    rets = np.diff(np.log(np.clip(close, 1e-9, None)))
    if rets.size < 80:
        raise ValueError("Need at least 80 returns for HMM training")

    hmm, states = fit_btc_regime_detector(rets, n_states=3, random_state=7)
    unique, counts = np.unique(states, return_counts=True)
    vols = {}
    for s in unique:
        vals = rets[states == s]
        vols[int(s)] = float(np.std(vals)) if vals.size > 0 else 0.0

    summary: Dict[str, Any] = {
        "parquet_path": parquet_path,
        "rows_used": int(len(sub)),
        "returns_used": int(rets.size),
        "state_counts": {int(k): int(v) for k, v in zip(unique, counts)},
        "state_volatility": vols,
        "transition_matrix": np.asarray(hmm.transmat_).tolist(),
        "means": np.asarray(hmm.means_).reshape(-1).tolist(),
        "covars": np.asarray(hmm.covars_).reshape(-1).tolist(),
    }
    sp = Path(summary_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_and_train_polygon(
    input_dir: str,
    out_dir: str = "data/bitcoin/processed",
) -> BTCRegimeTrainOutput:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = Path(input_dir).name
    parquet_path = str(out / f"btc_polygon_merged_{tag}.parquet")
    summary_path = str(out / f"btc_regime_hmm_summary_{tag}.json")
    rows = merge_polygon_to_parquet(input_dir=input_dir, out_parquet=parquet_path)
    train_hmm_from_parquet(parquet_path=parquet_path, summary_path=summary_path)
    return BTCRegimeTrainOutput(parquet_path=parquet_path, summary_path=summary_path, rows=rows)
