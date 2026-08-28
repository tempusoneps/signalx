from __future__ import annotations

from pathlib import Path

import pandas as pd

from signalx.constants import SignalState

REQUIRED_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe columns to lowercase OHLCV standard format.

    - Makes a copy of input DataFrame.
    - Strips and lowercases column names ('OPEN' -> 'open', etc.).
    - Verifies all required OHLCV columns exist (raises ValueError if missing).
    - Converts all 5 OHLCV columns to float/numeric.
    - Preserves any other columns.
    """
    df_out = df.copy()
    col_map = {col: str(col).strip().lower() for col in df_out.columns}
    df_out = df_out.rename(columns=col_map)

    missing = [c for c in REQUIRED_OHLCV_COLS if c not in df_out.columns]
    if missing:
        raise ValueError(
            f"Missing required OHLCV column(s): {missing}. Found: {list(df_out.columns)}"
        )

    for c in REQUIRED_OHLCV_COLS:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype(float)

    return df_out


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load DataFrame from CSV, TXT, or Parquet file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    elif suffix in [".csv", ".txt"]:
        return pd.read_csv(p)
    else:
        raise ValueError(
            f"Unsupported file format: {p.suffix}. Supported formats: .parquet, .csv, .txt"
        )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save DataFrame to CSV, TXT, or Parquet file, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    suffix = p.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        df.to_parquet(p, index=False)
    elif suffix in [".csv", ".txt"]:
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Supported formats: .parquet, .csv")

    return p


def compute_signal_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute distribution percentage of buy/sell/hold/none for all *_signal columns."""
    signal_cols = [c for c in df.columns if c.endswith("_signal")]
    stats: dict[str, dict[str, float]] = {}
    total_rows = len(df)

    for col in signal_cols:
        if total_rows == 0:
            stats[col] = {
                "total": 0.0,
                "buy_pct": 0.0,
                "sell_pct": 0.0,
                "hold_pct": 0.0,
                "none_pct": 0.0,
            }
            continue

        counts = df[col].value_counts().to_dict()
        stats[col] = {
            "total": float(total_rows),
            "buy_pct": round((counts.get(SignalState.BUY, 0) / total_rows) * 100.0, 2),
            "sell_pct": round((counts.get(SignalState.SELL, 0) / total_rows) * 100.0, 2),
            "hold_pct": round((counts.get(SignalState.HOLD, 0) / total_rows) * 100.0, 2),
            "none_pct": round((counts.get(SignalState.NONE, 0) / total_rows) * 100.0, 2),
        }

    return stats
