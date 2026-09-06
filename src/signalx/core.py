from __future__ import annotations

from typing import Literal

import pandas as pd

from signalx.signals import run_all_signal_generators
from signalx.utils import normalize_ohlcv

DATE_COLUMN_NAMES = {"date", "datetime", "timestamp", "time"}


def generate_signals(
    df: pd.DataFrame,
    drop_ohlcv: bool = False,
    show_progress: bool = False,
    naming: Literal["code", "semantic"] = "code",
) -> pd.DataFrame:
    """Generate 100+ standardized trading signal columns from an OHLCV dataset.

    Executes all 8 signal categories in order:
    1. Trend
    2. Momentum
    3. Volatility
    4. Volume
    5. Candlestick
    6. SMC
    7. Statistical
    8. Composite

    Parameters:
        df: Input DataFrame with open, high, low, close, volume columns.
        drop_ohlcv: If True, returns only date/datetime and signal columns.
                    If False, returns original DataFrame concatenated with signal columns.
        show_progress: If True, displays a real-time progress bar for each signal group.
        naming: Signal column naming convention: "code" (default) or "semantic".

    Returns:
        pd.DataFrame with all signals standardized to 'buy' | 'sell' | 'hold' | 'none'.
    """
    if naming not in ("code", "semantic"):
        raise ValueError(
            f"Invalid naming convention: {naming}. Allowed values: ('code', 'semantic')"
        )

    normalized = normalize_ohlcv(df)
    signals = run_all_signal_generators(
        normalized,
        show_progress=show_progress,
        naming=naming,
    )

    if drop_ohlcv:
        date_cols = [c for c in df.columns if str(c).strip().lower() in DATE_COLUMN_NAMES]
        if date_cols:
            return pd.concat([df[date_cols], signals], axis=1)
        return signals

    return pd.concat([df, signals], axis=1)


__all__ = ["generate_signals"]
