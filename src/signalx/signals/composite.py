from __future__ import annotations

import pandas as pd

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
from signalx.utils import normalize_ohlcv

COMPOSITE_SIGNAL_COLUMNS = [
    "comp_trend_consensus_signal",
    "comp_momentum_consensus_signal",
    "comp_master_ensemble_signal",
    "comp_ma_consensus_signal",
    "comp_trend_momentum_align_signal",
    "comp_breakout_volume_confirmed_signal",
    "comp_mean_reversion_confluence_signal",
    "comp_macd_hist_candle_reversal_signal",
]


def _calc_trend_consensus(intermediate: pd.DataFrame) -> pd.Series:
    """Calculate majority vote (>50%) across all available trend signals."""
    n = len(intermediate)
    if n == 0:
        return pd.Series(dtype=str, index=intermediate.index)

    trend_cols = [
        c for c in intermediate.columns if c.startswith("trend_") and c.endswith("_signal")
    ]
    if not trend_cols:
        return pd.Series(SignalState.NONE, index=intermediate.index, dtype=str)

    df_t = intermediate[trend_cols]
    num_cols = len(trend_cols)

    buy_counts = (df_t == SignalState.BUY).sum(axis=1)
    sell_counts = (df_t == SignalState.SELL).sum(axis=1)
    none_counts = (df_t == SignalState.NONE).sum(axis=1)

    all_none = none_counts == num_cols
    is_buy = buy_counts / num_cols > 0.5
    is_sell = sell_counts / num_cols > 0.5

    result = pd.Series(SignalState.HOLD, index=intermediate.index, dtype=str)
    result[all_none] = SignalState.NONE
    result[is_buy] = SignalState.BUY
    result[is_sell] = SignalState.SELL
    return result


def _calc_momentum_consensus(intermediate: pd.DataFrame) -> pd.Series:
    """Calculate majority vote (>50%) across all available momentum signals."""
    n = len(intermediate)
    if n == 0:
        return pd.Series(dtype=str, index=intermediate.index)

    mom_cols = [c for c in intermediate.columns if c.startswith("mom_") and c.endswith("_signal")]
    if not mom_cols:
        return pd.Series(SignalState.NONE, index=intermediate.index, dtype=str)

    df_m = intermediate[mom_cols]
    num_cols = len(mom_cols)

    buy_counts = (df_m == SignalState.BUY).sum(axis=1)
    sell_counts = (df_m == SignalState.SELL).sum(axis=1)
    none_counts = (df_m == SignalState.NONE).sum(axis=1)

    all_none = none_counts == num_cols
    is_buy = buy_counts / num_cols > 0.5
    is_sell = sell_counts / num_cols > 0.5

    result = pd.Series(SignalState.HOLD, index=intermediate.index, dtype=str)
    result[all_none] = SignalState.NONE
    result[is_buy] = SignalState.BUY
    result[is_sell] = SignalState.SELL
    return result


def _calc_master_ensemble(intermediate: pd.DataFrame) -> pd.Series:
    """Calculate weighted consensus across all library signals (>=35% vote threshold)."""
    n = len(intermediate)
    if n == 0:
        return pd.Series(dtype=str, index=intermediate.index)

    all_signal_cols = [
        c for c in intermediate.columns if c.endswith("_signal") and not c.startswith("comp_")
    ]
    if not all_signal_cols:
        return pd.Series(SignalState.NONE, index=intermediate.index, dtype=str)

    df_sig = intermediate[all_signal_cols]
    num_cols = len(all_signal_cols)

    buy_counts = (df_sig == SignalState.BUY).sum(axis=1)
    sell_counts = (df_sig == SignalState.SELL).sum(axis=1)
    none_counts = (df_sig == SignalState.NONE).sum(axis=1)

    all_none = none_counts == num_cols
    buy_ratio = buy_counts / num_cols
    sell_ratio = sell_counts / num_cols

    is_buy = (buy_ratio >= 0.35) & (buy_counts > sell_counts)
    is_sell = (sell_ratio >= 0.35) & (sell_counts > buy_counts)

    result = pd.Series(SignalState.HOLD, index=intermediate.index, dtype=str)
    result[all_none] = SignalState.NONE
    result[is_buy] = SignalState.BUY
    result[is_sell] = SignalState.SELL
    return result


def _calc_ma_consensus(intermediate: pd.DataFrame) -> pd.Series:
    """Calculate Moving Average consensus across multiple SMA/EMA periods."""
    n = len(intermediate)
    if n == 0:
        return pd.Series(dtype=str, index=intermediate.index)

    ma_keywords = ["sma", "ema", "dema", "tema", "hma", "vwma", "golden_cross", "above_"]
    ma_cols = [
        c
        for c in intermediate.columns
        if c.endswith("_signal") and any(kw in c for kw in ma_keywords)
    ]
    if not ma_cols:
        ma_cols = [
            c for c in intermediate.columns if c.startswith("trend_") and c.endswith("_signal")
        ]
    if not ma_cols:
        return pd.Series(SignalState.NONE, index=intermediate.index, dtype=str)

    df_ma = intermediate[ma_cols]
    num_cols = len(ma_cols)

    buy_counts = (df_ma == SignalState.BUY).sum(axis=1)
    sell_counts = (df_ma == SignalState.SELL).sum(axis=1)
    none_counts = (df_ma == SignalState.NONE).sum(axis=1)

    all_none = none_counts == num_cols
    is_buy = buy_counts / num_cols > 0.5
    is_sell = sell_counts / num_cols > 0.5

    result = pd.Series(SignalState.HOLD, index=intermediate.index, dtype=str)
    result[all_none] = SignalState.NONE
    result[is_buy] = SignalState.BUY
    result[is_sell] = SignalState.SELL
    return result


def _calc_trend_momentum_align(
    trend_consensus: pd.Series, momentum_consensus: pd.Series
) -> pd.Series:
    """Calculate trend and momentum alignment confluence signal."""
    n = len(trend_consensus)
    if n == 0:
        return pd.Series(dtype=str, index=trend_consensus.index)

    result = pd.Series(SignalState.HOLD, index=trend_consensus.index, dtype=str)

    either_none = (trend_consensus == SignalState.NONE) | (momentum_consensus == SignalState.NONE)
    is_buy = (trend_consensus == SignalState.BUY) & (momentum_consensus == SignalState.BUY)
    is_sell = (trend_consensus == SignalState.SELL) & (momentum_consensus == SignalState.SELL)

    result[either_none] = SignalState.NONE
    result[is_buy] = SignalState.BUY
    result[is_sell] = SignalState.SELL
    return result


def _calc_breakout_volume_confirmed(intermediate: pd.DataFrame) -> pd.Series:
    """Calculate price breakout confirmed with volume spike signal."""
    n = len(intermediate)
    if n == 0:
        return pd.Series(dtype=str, index=intermediate.index)

    breakout_cols = [
        c
        for c in intermediate.columns
        if any(kw in c for kw in ["donchian", "keltner", "vol_bb_breakout", "ttm_squeeze"])
    ]
    vol_spike_cols = [
        c
        for c in intermediate.columns
        if any(kw in c for kw in ["spike", "obv", "cmf", "vwap"]) and c.startswith("volume_")
    ]

    if not breakout_cols or not vol_spike_cols:
        return pd.Series(SignalState.NONE, index=intermediate.index, dtype=str)

    df_bo = intermediate[breakout_cols]
    df_vol = intermediate[vol_spike_cols]

    bo_buy = (df_bo == SignalState.BUY).any(axis=1)
    bo_sell = (df_bo == SignalState.SELL).any(axis=1)
    vol_buy = (df_vol == SignalState.BUY).any(axis=1)
    vol_sell = (df_vol == SignalState.SELL).any(axis=1)

    all_none_bo = (df_bo == SignalState.NONE).all(axis=1)
    all_none_vol = (df_vol == SignalState.NONE).all(axis=1)
    all_none = all_none_bo & all_none_vol

    is_buy = bo_buy & vol_buy
    is_sell = bo_sell & vol_sell

    result = pd.Series(SignalState.HOLD, index=intermediate.index, dtype=str)
    result[all_none] = SignalState.NONE
    result[is_buy] = SignalState.BUY
    result[is_sell] = SignalState.SELL
    return result


def _calc_mean_reversion_confluence(intermediate: pd.DataFrame) -> pd.Series:
    """Calculate multi-oscillator mean reversion confluence signal."""
    n = len(intermediate)
    if n == 0:
        return pd.Series(dtype=str, index=intermediate.index)

    rsi_cols = [c for c in intermediate.columns if "rsi_ob_os" in c]
    bb_bounce_cols = [
        c for c in intermediate.columns if any(kw in c for kw in ["bb_bounce", "pct_b_reversal"])
    ]
    zscore_cols = [c for c in intermediate.columns if "stat_price_zscore" in c]

    available_families = [cols for cols in [rsi_cols, bb_bounce_cols, zscore_cols] if cols]
    if not available_families:
        return pd.Series(SignalState.NONE, index=intermediate.index, dtype=str)

    buy_votes = pd.Series(0, index=intermediate.index, dtype=int)
    sell_votes = pd.Series(0, index=intermediate.index, dtype=int)
    none_votes = pd.Series(0, index=intermediate.index, dtype=int)

    for cols in available_families:
        sub = intermediate[cols]
        buy_votes += (sub == SignalState.BUY).any(axis=1).astype(int)
        sell_votes += (sub == SignalState.SELL).any(axis=1).astype(int)
        none_votes += (sub == SignalState.NONE).all(axis=1).astype(int)

    num_families = len(available_families)
    all_none = none_votes == num_families

    threshold = 2 if num_families >= 2 else 1
    is_buy = (buy_votes >= threshold) & (buy_votes > sell_votes)
    is_sell = (sell_votes >= threshold) & (sell_votes > buy_votes)

    result = pd.Series(SignalState.HOLD, index=intermediate.index, dtype=str)
    result[all_none] = SignalState.NONE
    result[is_buy] = SignalState.BUY
    result[is_sell] = SignalState.SELL
    return result


def generate_composite_signals(
    df: pd.DataFrame,
    intermediate_signals: pd.DataFrame | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Generate all 7 standardized composite and consensus signals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
    intermediate_signals : pd.DataFrame, optional
        Pre-computed intermediate signals DataFrame across trend, momentum, volatility,
        volume, candlestick, and statistical families. If None, intermediate signals
        will be computed automatically from df.
    show_progress : bool, default False
        Whether to display a real-time progress bar for this signal group.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 7 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)
    signals = pd.DataFrame(index=df_norm.index)

    with GroupProgressBar(
        "Composite Signals", total=len(COMPOSITE_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if df_norm.empty:
            for col in COMPOSITE_SIGNAL_COLUMNS:
                signals[col] = pd.Series(dtype=str)
            pbar.update(len(COMPOSITE_SIGNAL_COLUMNS))
            return signals

        if intermediate_signals is None or intermediate_signals.empty:
            # Import lazily to avoid circular dependencies
            from signalx.signals.candlestick import generate_candlestick_signals
            from signalx.signals.momentum import generate_momentum_signals
            from signalx.signals.statistical import generate_statistical_signals
            from signalx.signals.trend import generate_trend_signals
            from signalx.signals.volatility import generate_volatility_signals
            from signalx.signals.volume import generate_volume_signals

            trend = generate_trend_signals(df_norm, show_progress=show_progress)
            mom = generate_momentum_signals(df_norm, show_progress=show_progress)
            vol = generate_volatility_signals(df_norm, show_progress=show_progress)
            volume = generate_volume_signals(df_norm, show_progress=show_progress)
            cdl = generate_candlestick_signals(df_norm, show_progress=show_progress)
            stat = generate_statistical_signals(df_norm, show_progress=show_progress)
            intermediate = pd.concat([trend, mom, vol, volume, cdl, stat], axis=1)
        else:
            intermediate = intermediate_signals

        # 1. Trend Consensus Signal (1)
        trend_con = _calc_trend_consensus(intermediate)
        signals["comp_trend_consensus_signal"] = trend_con
        pbar.update(1)

        # 2. Momentum Consensus Signal (1)
        mom_con = _calc_momentum_consensus(intermediate)
        signals["comp_momentum_consensus_signal"] = mom_con
        pbar.update(1)

        # 3. Master Ensemble Signal (1)
        signals["comp_master_ensemble_signal"] = _calc_master_ensemble(intermediate)
        pbar.update(1)

        # 4. Moving Average Consensus Signal (1)
        signals["comp_ma_consensus_signal"] = _calc_ma_consensus(intermediate)
        pbar.update(1)

        # 5. Trend & Momentum Alignment Signal (1)
        signals["comp_trend_momentum_align_signal"] = _calc_trend_momentum_align(trend_con, mom_con)
        pbar.update(1)

        # 6. Breakout Volume Confirmed Signal (1)
        signals["comp_breakout_volume_confirmed_signal"] = _calc_breakout_volume_confirmed(
            intermediate
        )
        pbar.update(1)

        # 7. Mean Reversion Confluence Signal (1)
        signals["comp_mean_reversion_confluence_signal"] = _calc_mean_reversion_confluence(
            intermediate
        )
        pbar.update(1)

        # 8. MACD Histogram + Candle Reversal Signal (1)
        signals["comp_macd_hist_candle_reversal_signal"] = _calc_macd_hist_candle_reversal(df_norm)
        pbar.update(1)

        # Ensure all columns are present, filled with NONE, and matching index
        for col in COMPOSITE_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[COMPOSITE_SIGNAL_COLUMNS]


def _calc_macd_hist_candle_reversal(df_norm: pd.DataFrame) -> pd.Series:
    """Calculate MACD Histogram turning point combined with candlestick confirmation."""
    import numpy as np

    close = df_norm["close"]
    open_p = df_norm["open"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    is_max_macd_hist = macd_hist == macd_hist.rolling(10, min_periods=2).max()
    is_min_macd_hist = macd_hist == macd_hist.rolling(10, min_periods=2).min()
    hist_rolling_sum = macd_hist.rolling(5, min_periods=2).sum()

    o_prev = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    h = macd_hist.to_numpy(dtype=float, na_value=np.nan)
    h_sum = hist_rolling_sum.to_numpy(dtype=float, na_value=np.nan)

    is_min_shift = is_min_macd_hist.shift(1, fill_value=False).to_numpy(dtype=bool)
    is_max_shift = is_max_macd_hist.shift(1, fill_value=False).to_numpy(dtype=bool)

    valid = ~np.isnan(c) & ~np.isnan(o_prev) & ~np.isnan(c_prev) & ~np.isnan(h) & ~np.isnan(h_sum)
    bull = valid & (o_prev >= c_prev) & (h < 0) & is_min_shift & (h_sum < 0) & (c > o_prev)
    bear = valid & (o_prev <= c_prev) & (h > 0) & is_max_shift & (h_sum > 0) & (c < o_prev)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )
