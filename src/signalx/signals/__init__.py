from __future__ import annotations

import pandas as pd

from signalx.signals.candlestick import generate_candlestick_signals
from signalx.signals.composite import generate_composite_signals
from signalx.signals.momentum import generate_momentum_signals
from signalx.signals.statistical import generate_statistical_signals
from signalx.signals.trend import generate_trend_signals
from signalx.signals.volatility import generate_volatility_signals
from signalx.signals.volume import generate_volume_signals
from signalx.utils import normalize_ohlcv


def run_all_signal_generators(df: pd.DataFrame) -> pd.DataFrame:
    """Run all category signal generators sequentially and compile standardized signals.

    Parameters:
        df: Input DataFrame containing OHLCV price series.

    Returns:
        pd.DataFrame containing all 114 signal columns sorted alphabetically.
    """
    normalized = normalize_ohlcv(df)

    trend_df = generate_trend_signals(normalized)
    mom_df = generate_momentum_signals(normalized)
    vol_df = generate_volatility_signals(normalized)
    volume_df = generate_volume_signals(normalized)
    cdl_df = generate_candlestick_signals(normalized)
    stat_df = generate_statistical_signals(normalized)

    intermediate = pd.concat([trend_df, mom_df, vol_df, volume_df, cdl_df, stat_df], axis=1)
    comp_df = generate_composite_signals(normalized, intermediate)

    full_signals = pd.concat([intermediate, comp_df], axis=1)
    return full_signals.reindex(sorted(full_signals.columns), axis=1)


__all__ = [
    "generate_candlestick_signals",
    "generate_composite_signals",
    "generate_momentum_signals",
    "generate_statistical_signals",
    "generate_trend_signals",
    "generate_volatility_signals",
    "generate_volume_signals",
    "run_all_signal_generators",
]
