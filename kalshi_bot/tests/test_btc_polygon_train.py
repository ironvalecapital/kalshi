from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_engine.btc_regime_train import merge_polygon_to_parquet, train_hmm_from_parquet


def test_merge_and_train_polygon_pipeline(tmp_path: Path):
    in_dir = tmp_path / "sample_polygon"
    in_dir.mkdir(parents=True, exist_ok=True)

    n = 220
    ts0 = 1_700_000_000_000
    close = np.exp(np.cumsum(np.random.default_rng(7).normal(0.0, 0.001, n))) * 40000
    df = pd.DataFrame(
        {
            "ts_ms": [ts0 + i * 60_000 for i in range(n)],
            "ts_iso": ["" for _ in range(n)],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.random.default_rng(8).uniform(1, 10, n),
            "vwap": close,
            "transactions": np.random.default_rng(9).integers(1, 20, n),
        }
    )
    df.to_csv(in_dir / "polygon_btcusd_1m_30d.csv", index=False)

    out_parquet = tmp_path / "merged.parquet"
    rows = merge_polygon_to_parquet(str(in_dir), str(out_parquet))
    assert rows == n
    assert out_parquet.exists()

    out_summary = tmp_path / "summary.json"
    summary = train_hmm_from_parquet(str(out_parquet), str(out_summary))
    assert summary["returns_used"] >= 80
    assert out_summary.exists()
