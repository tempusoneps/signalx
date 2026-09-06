from __future__ import annotations

from typing import Literal

import pandas as pd

from signalx.metadata import to_code_names
from signalx.signals.candlestick import generate_candlestick_signals
from signalx.signals.composite import generate_composite_signals
from signalx.signals.momentum import generate_momentum_signals
from signalx.signals.smc import SMC_SIGNAL_COLUMNS, generate_smc_signals
from signalx.signals.statistical import generate_statistical_signals
from signalx.signals.trend import generate_trend_signals
from signalx.signals.volatility import generate_volatility_signals
from signalx.signals.volume import generate_volume_signals
from signalx.utils import normalize_ohlcv


def run_all_signal_generators(
    df: pd.DataFrame,
    show_progress: bool = False,
    naming: Literal["code", "semantic"] = "code",
) -> pd.DataFrame:
    """Run all category signal generators sequentially and compile standardized signals.

    Parameters:
        df: Input DataFrame containing OHLCV price series.
        show_progress: Whether to display real-time progress bars for each signal group.
        naming: Signal column naming convention: "code" (default) or "semantic".

    Returns:
        pd.DataFrame containing all signal columns sorted alphabetically.
    """
    if naming not in ("code", "semantic"):
        raise ValueError(
            f"Invalid naming convention: {naming}. Allowed values: ('code', 'semantic')"
        )

    normalized = normalize_ohlcv(df)

    trend_df = generate_trend_signals(normalized, show_progress=show_progress)
    mom_df = generate_momentum_signals(normalized, show_progress=show_progress)
    vol_df = generate_volatility_signals(normalized, show_progress=show_progress)
    volume_df = generate_volume_signals(normalized, show_progress=show_progress)
    cdl_df = generate_candlestick_signals(normalized, show_progress=show_progress)
    smc_df = generate_smc_signals(normalized, show_progress=show_progress)
    stat_df = generate_statistical_signals(normalized, show_progress=show_progress)

    intermediate = pd.concat([trend_df, mom_df, vol_df, volume_df, cdl_df, smc_df, stat_df], axis=1)
    comp_df = generate_composite_signals(normalized, intermediate, show_progress=show_progress)

    full_signals = pd.concat([intermediate, comp_df], axis=1)
    if naming == "code":
        full_signals = to_code_names(full_signals)

    return full_signals.reindex(sorted(full_signals.columns), axis=1)


__all__ = [
    "SMC_SIGNAL_COLUMNS",
    "generate_candlestick_signals",
    "generate_composite_signals",
    "generate_momentum_signals",
    "generate_smc_signals",
    "generate_statistical_signals",
    "generate_trend_signals",
    "generate_volatility_signals",
    "generate_volume_signals",
    "run_all_signal_generators",
]
