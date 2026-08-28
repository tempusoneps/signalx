from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
from signalx.utils import normalize_ohlcv

VOLATILITY_SIGNAL_COLUMNS = [
    "vol_bb_breakout_20_20_signal",
    "vol_bb_bounce_20_20_signal",
    "vol_bb_breakout_50_25_signal",
    "vol_bb_bounce_50_25_signal",
    "vol_bb_pct_b_reversal_20_signal",
    "vol_bb_pct_b_reversal_50_signal",
    "vol_donchian_breakout_10_signal",
    "vol_donchian_breakout_20_signal",
    "vol_donchian_breakout_55_signal",
    "vol_keltner_breakout_20_15_signal",
    "vol_keltner_breakout_20_20_signal",
    "vol_ttm_squeeze_signal",
    "vol_bb_bandwidth_expansion_signal",
    "vol_atr_trailing_stop_2x_signal",
    "vol_atr_trailing_stop_3x_signal",
    "vol_chaikin_volatility_surge_signal",
    "vol_hv_ratio_breakout_10_30_signal",
    "vol_bb_rejection_20_signal",
    "vol_bb_bandwidth_regime_signal",
    "vol_compression_breakout_5_20_signal",
    "vol_atr_expansion_14_20_signal",
    "vol_range_breakout_10_signal",
    "vol_volatility_break_10_50_signal",
    "vol_range_compression_20_signal",
    "vol_volatility_drop_5_20_signal",
    "vol_range_expansion_20_signal",
    "vol_range_shift_5_10_signal",
    "vol_range_position_10_signal",
    "vol_range_flip_5_signal",
    "vol_linreg_channel_reversal_20_signal",
    "vol_keltner_reversal_20_signal",
    "vol_envelope_breakout_20_signal",
]


def _channel_breakout_signal(close: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Helper for channel breakout signals (upper/lower bounds)."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    l_arr = lower.to_numpy(dtype=float, na_value=np.nan)
    u_arr = upper.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(l_arr) & ~np.isnan(u_arr)
    condlist = [
        valid & (c > u_arr),
        valid & (c < l_arr),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _donchian_breakout_signal(close: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Helper for Donchian Channel breakout signals."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    l_arr = lower.to_numpy(dtype=float, na_value=np.nan)
    u_arr = upper.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(l_arr) & ~np.isnan(u_arr)
    condlist = [
        valid & (c >= u_arr),
        valid & (c <= l_arr),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_bb_bounce(close: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Helper to detect Bollinger Bands mean-reversion bounces back inside bands."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    l_arr = lower.to_numpy(dtype=float, na_value=np.nan)
    u_arr = upper.to_numpy(dtype=float, na_value=np.nan)

    prev_c = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    prev_l = lower.shift(1).to_numpy(dtype=float, na_value=np.nan)
    prev_u = upper.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(l_arr) & ~np.isnan(u_arr)
    prev_valid = ~np.isnan(prev_c) & ~np.isnan(prev_l) & ~np.isnan(prev_u)

    bounce_buy = valid & prev_valid & (prev_c <= prev_l) & (c > l_arr)
    bounce_sell = valid & prev_valid & (prev_c >= prev_u) & (c < u_arr)

    condlist = [
        bounce_buy,
        bounce_sell,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_pct_b_reversal(pct_b: pd.Series) -> pd.Series:
    """Helper to detect Bollinger %B extreme boundary reversals back into 0-1 zone."""
    b = pct_b.to_numpy(dtype=float, na_value=np.nan)
    prev_b = pct_b.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(b)
    prev_valid = ~np.isnan(prev_b)

    rev_buy = valid & prev_valid & (prev_b <= 0.0) & (b > 0.0)
    rev_sell = valid & prev_valid & (prev_b >= 1.0) & (b < 1.0)

    condlist = [
        rev_buy,
        rev_sell,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=pct_b.index, dtype=str)


def _calc_ttm_squeeze(
    bb_h: pd.Series,
    bb_l: pd.Series,
    kc_h: pd.Series,
    kc_l: pd.Series,
    close: pd.Series,
    baseline: pd.Series,
) -> pd.Series:
    """Helper to calculate TTM Squeeze breakout signals."""
    bb_h_arr = bb_h.to_numpy(dtype=float, na_value=np.nan)
    bb_l_arr = bb_l.to_numpy(dtype=float, na_value=np.nan)
    kc_h_arr = kc_h.to_numpy(dtype=float, na_value=np.nan)
    kc_l_arr = kc_l.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    base_arr = baseline.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(bb_h_arr)
        & ~np.isnan(bb_l_arr)
        & ~np.isnan(kc_h_arr)
        & ~np.isnan(kc_l_arr)
        & ~np.isnan(c_arr)
        & ~np.isnan(base_arr)
    )

    squeeze_on = valid & (bb_h_arr <= kc_h_arr) & (bb_l_arr >= kc_l_arr)
    squeeze_fired = valid & ~squeeze_on

    condlist = [
        squeeze_fired & (c_arr > base_arr),
        squeeze_fired & (c_arr < base_arr),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_atr_trailing_stop(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int = 14,
    multiplier: float = 2.0,
) -> pd.Series:
    """Calculate iterative ATR trailing stop price direction regime."""
    n = len(close)
    if n == 0:
        return pd.Series(dtype=str, index=close.index)

    res = np.full(n, SignalState.NONE, dtype=object)
    if n < window or window <= 0:
        return pd.Series(res, index=close.index, dtype=str)

    try:
        atr_ind = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=window, fillna=False
        )
        atr_arr = atr_ind.average_true_range().to_numpy(dtype=float, na_value=np.nan)
    except Exception:
        return pd.Series(res, index=close.index, dtype=str)

    close_arr = close.to_numpy(dtype=float, na_value=np.nan)
    high_arr = high.to_numpy(dtype=float, na_value=np.nan)
    low_arr = low.to_numpy(dtype=float, na_value=np.nan)

    state = SignalState.NONE
    stop = np.nan

    for i in range(n):
        if np.isnan(atr_arr[i]) or np.isnan(close_arr[i]):
            res[i] = SignalState.NONE
            continue

        c = close_arr[i]
        a = atr_arr[i]

        if state == SignalState.NONE:
            mid = (high_arr[i] + low_arr[i]) / 2.0 if not np.isnan(high_arr[i]) else c
            if c >= mid:
                state = SignalState.BUY
                stop = c - multiplier * a
            else:
                state = SignalState.SELL
                stop = c + multiplier * a
            res[i] = state
            continue

        if state == SignalState.BUY:
            stop = max(stop, c - multiplier * a)
            if c < stop:
                state = SignalState.SELL
                stop = c + multiplier * a
            res[i] = state
        elif state == SignalState.SELL:
            stop = min(stop, c + multiplier * a)
            if c > stop:
                state = SignalState.BUY
                stop = c - multiplier * a
            res[i] = state

    return pd.Series(res, index=close.index, dtype=str)


def _calc_chaikin_volatility(
    high: pd.Series, low: pd.Series, length: int = 10, roc_length: int = 10
) -> pd.Series:
    """Calculate Chaikin Volatility indicator surge signal."""
    hl = high - low
    ema_hl = hl.ewm(span=length, adjust=False).mean()
    prev_ema = ema_hl.shift(roc_length).replace(0, np.nan)
    cv = ((ema_hl - prev_ema) / prev_ema) * 100.0

    cv_arr = cv.to_numpy(dtype=float, na_value=np.nan)
    valid = ~np.isnan(cv_arr)

    condlist = [
        valid & (cv_arr > 0),
        valid & (cv_arr < 0),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=high.index, dtype=str)


def _calc_hv_ratio_breakout(
    close: pd.Series,
    window_fast: int = 10,
    window_slow: int = 30,
    threshold: float = 1.5,
) -> pd.Series:
    """Calculate Historical Volatility ratio (fast/slow) breakout signal."""
    log_ret = np.log(close / close.shift(1).replace(0, np.nan))
    hv_fast = log_ret.rolling(window=window_fast).std()
    hv_slow = log_ret.rolling(window=window_slow).std()
    hv_ratio = hv_fast / hv_slow.replace(0, np.nan)

    ratio_arr = hv_ratio.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    prev_c_arr = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(ratio_arr) & ~np.isnan(c_arr) & ~np.isnan(prev_c_arr)
    surge = valid & (ratio_arr > threshold)

    condlist = [
        surge & (c_arr > prev_c_arr),
        surge & (c_arr < prev_c_arr),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_bb_rejection(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_lower: pd.Series,
    bb_upper: pd.Series,
    rsi: pd.Series,
) -> pd.Series:
    """Calculate Bollinger Band Lower/Upper rejection with wick and RSI confluence."""
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    lb = bb_lower.to_numpy(dtype=float, na_value=np.nan)
    ub = bb_upper.to_numpy(dtype=float, na_value=np.nan)
    r = rsi.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(lb) & ~np.isnan(ub) & ~np.isnan(r)
    body_abs = np.abs(c - o)
    upwick = h - np.maximum(c, o)
    lowwick = np.minimum(c, o) - l_arr

    bull_rej = valid & (l_arr < lb) & (c > lb) & (r < 35.0) & (lowwick > body_abs)
    bear_rej = valid & (h > ub) & (c < ub) & (r > 65.0) & (upwick > body_abs)
    hold = valid & ~bull_rej & ~bear_rej

    conds = [bull_rej, bear_rej, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_bb_bandwidth_regime(bb_width: pd.Series) -> pd.Series:
    """Calculate Bollinger Bandwidth Squeeze vs Expansion Regime signal."""
    sma20 = bb_width.rolling(20, min_periods=5).mean()
    w = bb_width.to_numpy(dtype=float, na_value=np.nan)
    s = sma20.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(w) & ~np.isnan(s)
    squeeze = valid & (w < (s * 0.7))
    expansion = valid & (w > (s * 1.3))
    hold = valid & ~squeeze & ~expansion

    conds = [squeeze, expansion, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=bb_width.index, dtype=str
    )


def _calc_compression_breakout(close: pd.Series) -> pd.Series:
    """Calculate Volatility Compression STD(5) < 0.5*STD(20) Breakout signal."""
    std5 = close.rolling(5, min_periods=2).std()
    std20 = close.rolling(20, min_periods=5).std()
    comp = std5 < (std20 * 0.5)

    c = close.to_numpy(dtype=float, na_value=np.nan)
    max10 = close.rolling(10, min_periods=2).max().shift(1).to_numpy(dtype=float, na_value=np.nan)
    min10 = close.rolling(10, min_periods=2).min().shift(1).to_numpy(dtype=float, na_value=np.nan)
    comp_arr = comp.to_numpy(dtype=bool)

    valid = ~np.isnan(c) & ~np.isnan(max10) & ~np.isnan(min10)
    bull = valid & comp_arr & (c > max10)
    bear = valid & comp_arr & (c < min10)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_atr_expansion(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate ATR(14) Expansion > 1.5x SMA20(ATR) with directional price move."""
    try:
        atr = ta.volatility.AverageTrueRange(
            high, low, close, window=14, fillna=False
        ).average_true_range()
    except Exception:
        atr = high - low
    atr_sma = atr.rolling(20, min_periods=5).mean()

    a = atr.to_numpy(dtype=float, na_value=np.nan)
    s = atr_sma.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(a) & ~np.isnan(s) & ~np.isnan(c) & ~np.isnan(c1)
    is_expanded = valid & (a > (1.5 * s))
    bull = is_expanded & (c > c1)
    bear = is_expanded & (c < c1)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_range_breakout(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10
) -> pd.Series:
    """Calculate rolling N-period High/Low Range Breakout signal."""
    h_max = (
        high.rolling(window, min_periods=2).max().shift(1).to_numpy(dtype=float, na_value=np.nan)
    )
    l_min = low.rolling(window, min_periods=2).min().shift(1).to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(h_max) & ~np.isnan(l_min)
    bull = valid & (c > h_max)
    bear = valid & (c < l_min)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_volatility_break(close: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """Calculate Volatility Break STD(fast) > STD(slow) signal."""
    std_fast = (
        close.rolling(fast, min_periods=fast // 2).std().to_numpy(dtype=float, na_value=np.nan)
    )
    std_slow = (
        close.rolling(slow, min_periods=slow // 2).std().to_numpy(dtype=float, na_value=np.nan)
    )

    valid = ~np.isnan(std_fast) & ~np.isnan(std_slow)
    bull = valid & (std_fast > std_slow)
    bear = valid & (std_fast < std_slow)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_range_compression(
    high: pd.Series, low: pd.Series, close: pd.Series, sma: pd.Series
) -> pd.Series:
    """Calculate Range Compression (Height < 0.5x SMA20 Range) with SMA20 position."""
    height = high - low
    range_sma = height.rolling(20, min_periods=5).mean()

    h_arr = height.to_numpy(dtype=float, na_value=np.nan)
    s_arr = range_sma.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = sma.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(h_arr) & ~np.isnan(s_arr) & ~np.isnan(c) & ~np.isnan(sma_arr)
    compressed = valid & (h_arr < (0.5 * s_arr))
    bull = compressed & (c >= sma_arr)
    bear = compressed & (c < sma_arr)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_volatility_drop(close: pd.Series, sma: pd.Series) -> pd.Series:
    """Calculate Volatility Drop (STD5 < 0.5x STD20) relative to SMA20."""
    std5 = close.rolling(5, min_periods=2).std().to_numpy(dtype=float, na_value=np.nan)
    std20 = close.rolling(20, min_periods=5).std().to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = sma.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(std5) & ~np.isnan(std20) & ~np.isnan(c) & ~np.isnan(sma_arr)
    drop = valid & (std5 < (0.5 * std20))
    bull = drop & (c >= sma_arr)
    bear = drop & (c < sma_arr)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_range_expansion(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Range Expansion (Height > 1.5x SMA20 Range) with directional candle."""
    height = high - low
    range_sma = height.rolling(20, min_periods=5).mean()

    h_arr = height.to_numpy(dtype=float, na_value=np.nan)
    s_arr = range_sma.to_numpy(dtype=float, na_value=np.nan)
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(h_arr) & ~np.isnan(s_arr) & ~np.isnan(o) & ~np.isnan(c)
    expanded = valid & (h_arr > (1.5 * s_arr))
    bull = expanded & (c > o)
    bear = expanded & (c < o)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_range_shift(high: pd.Series, low: pd.Series) -> pd.Series:
    """Calculate Range Shift (5-bar vs 10-bar extreme shift)."""
    low5 = low.rolling(5, min_periods=2).min().to_numpy(dtype=float, na_value=np.nan)
    low10 = low.rolling(10, min_periods=3).min().to_numpy(dtype=float, na_value=np.nan)
    high5 = high.rolling(5, min_periods=2).max().to_numpy(dtype=float, na_value=np.nan)
    high10 = high.rolling(10, min_periods=3).max().to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(low5) & ~np.isnan(low10) & ~np.isnan(high5) & ~np.isnan(high10)
    bull = valid & (low5 > low10)
    bear = valid & (high5 < high10)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=high.index, dtype=str
    )


def _calc_range_position(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10
) -> pd.Series:
    """Calculate Range Position (lower 20% vs upper 20% of 10-bar channel)."""
    h_max = high.rolling(window, min_periods=2).max().to_numpy(dtype=float, na_value=np.nan)
    l_min = low.rolling(window, min_periods=2).min().to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(h_max) & ~np.isnan(l_min)
    rng = h_max - l_min
    lower_pos = l_min + 0.2 * rng
    upper_pos = h_max - 0.2 * rng

    bull = valid & (c < lower_pos)
    bear = valid & (c > upper_pos)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_linreg_channel_reversal(close: pd.Series, length: int = 20) -> pd.Series:
    """Calculate 20-period Linear Regression Channel lower/upper band reversal."""
    n = len(close)
    if n < length:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    x = np.arange(length)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    mid_vals = np.full(n, np.nan, dtype=float)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)

    for i in range(length - 1, n):
        y = c_arr[i - length + 1 : i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        mid_vals[i] = slope * (length - 1) + intercept

    mid_ser = pd.Series(mid_vals, index=close.index)
    std_ser = close.rolling(length, min_periods=5).std()
    upper = mid_ser + 2.0 * std_ser
    lower = mid_ser - 2.0 * std_ser

    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    u_arr = upper.to_numpy(dtype=float, na_value=np.nan)
    u1 = upper.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_arr = lower.to_numpy(dtype=float, na_value=np.nan)
    l1 = lower.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(c_arr)
        & ~np.isnan(c1)
        & ~np.isnan(l_arr)
        & ~np.isnan(l1)
        & ~np.isnan(u_arr)
        & ~np.isnan(u1)
    )
    bull = valid & (c_arr < l_arr) & (c1 > l1)
    bear = valid & (c_arr > u_arr) & (c1 < u1)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_keltner_reversal(close: pd.Series, kc_lower: pd.Series, kc_upper: pd.Series) -> pd.Series:
    """Calculate Keltner Channel Lower/Upper reversal signal."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_arr = kc_lower.to_numpy(dtype=float, na_value=np.nan)
    l1 = kc_lower.shift(1).to_numpy(dtype=float, na_value=np.nan)
    u_arr = kc_upper.to_numpy(dtype=float, na_value=np.nan)
    u1 = kc_upper.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(c)
        & ~np.isnan(c1)
        & ~np.isnan(l_arr)
        & ~np.isnan(l1)
        & ~np.isnan(u_arr)
        & ~np.isnan(u1)
    )
    bull = valid & (c < l_arr) & (c1 > l1)
    bear = valid & (c > u_arr) & (c1 < u1)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_envelope_breakout(close: pd.Series, sma: pd.Series, pct: float = 0.025) -> pd.Series:
    """Calculate Moving Average Envelope (SMA +/- pct) breakout signal."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    s = sma.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(s)
    upper = s * (1.0 + pct)
    lower = s * (1.0 - pct)

    bull = valid & (c > upper)
    bear = valid & (c < lower)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def generate_volatility_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Generate all 32 volatility, channel, and band signals from OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
    show_progress : bool, default False
        Whether to display a real-time progress bar for this signal group.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 32 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)

    with GroupProgressBar(
        "Volatility Signals", total=len(VOLATILITY_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if len(df_norm) == 0:
            pbar.update(len(VOLATILITY_SIGNAL_COLUMNS))
            return pd.DataFrame(
                {col: pd.Series(dtype=str, index=df.index) for col in VOLATILITY_SIGNAL_COLUMNS},
                index=df.index,
            )

        open_p = df_norm["open"]
        high = df_norm["high"]
        low = df_norm["low"]
        close = df_norm["close"]

        signals = pd.DataFrame(index=df_norm.index)

        # 1. Bollinger Bands (20, 2.0) and (50, 2.5) (4)
        bb20 = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0, fillna=False)
        bb20_h = bb20.bollinger_hband()
        bb20_l = bb20.bollinger_lband()
        bb20_m = bb20.bollinger_mavg()
        bb20_p = bb20.bollinger_pband()
        bb20_w = bb20.bollinger_wband()

        signals["vol_bb_breakout_20_20_signal"] = _channel_breakout_signal(close, bb20_l, bb20_h)
        signals["vol_bb_bounce_20_20_signal"] = _calc_bb_bounce(close, bb20_l, bb20_h)

        bb50 = ta.volatility.BollingerBands(close=close, window=50, window_dev=2.5, fillna=False)
        signals["vol_bb_breakout_50_25_signal"] = _channel_breakout_signal(
            close, bb50.bollinger_lband(), bb50.bollinger_hband()
        )
        signals["vol_bb_bounce_50_25_signal"] = _calc_bb_bounce(
            close, bb50.bollinger_lband(), bb50.bollinger_hband()
        )
        pbar.update(4)

        # 2. Bollinger %B Reversals (20, 50) (2)
        signals["vol_bb_pct_b_reversal_20_signal"] = _calc_pct_b_reversal(bb20_p)
        signals["vol_bb_pct_b_reversal_50_signal"] = _calc_pct_b_reversal(bb50.bollinger_pband())
        pbar.update(2)

        # 3. Donchian Channel Breakouts (10, 20, 55) (3)
        dc10 = ta.volatility.DonchianChannel(
            high=high, low=low, close=close, window=10, fillna=False
        )
        signals["vol_donchian_breakout_10_signal"] = _donchian_breakout_signal(
            close, dc10.donchian_channel_lband().shift(1), dc10.donchian_channel_hband().shift(1)
        )

        dc20 = ta.volatility.DonchianChannel(
            high=high, low=low, close=close, window=20, fillna=False
        )
        signals["vol_donchian_breakout_20_signal"] = _donchian_breakout_signal(
            close, dc20.donchian_channel_lband().shift(1), dc20.donchian_channel_hband().shift(1)
        )

        dc55 = ta.volatility.DonchianChannel(
            high=high, low=low, close=close, window=55, fillna=False
        )
        signals["vol_donchian_breakout_55_signal"] = _donchian_breakout_signal(
            close, dc55.donchian_channel_lband().shift(1), dc55.donchian_channel_hband().shift(1)
        )
        pbar.update(3)

        # 4. Keltner Channel Breakouts (20/1.5, 20/2.0) (2)
        kc15 = ta.volatility.KeltnerChannel(
            high=high,
            low=low,
            close=close,
            window=20,
            window_atr=10,
            multiplier=1.5,
            original_version=False,
            fillna=False,
        )
        signals["vol_keltner_breakout_20_15_signal"] = _channel_breakout_signal(
            close, kc15.keltner_channel_lband(), kc15.keltner_channel_hband()
        )

        kc20 = ta.volatility.KeltnerChannel(
            high=high,
            low=low,
            close=close,
            window=20,
            window_atr=10,
            multiplier=2.0,
            original_version=False,
            fillna=False,
        )
        signals["vol_keltner_breakout_20_20_signal"] = _channel_breakout_signal(
            close, kc20.keltner_channel_lband(), kc20.keltner_channel_hband()
        )
        pbar.update(2)

        # 5. TTM Squeeze Breakout (1)
        sig_ttm = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
        if len(df_norm) >= 20:
            try:
                sig_ttm = _calc_ttm_squeeze(
                    bb20_h,
                    bb20_l,
                    kc15.keltner_channel_hband(),
                    kc15.keltner_channel_lband(),
                    close,
                    bb20_m,
                )
            except Exception:
                pass
        signals["vol_ttm_squeeze_signal"] = sig_ttm
        pbar.update(1)

        # 6. Bollinger Bandwidth Expansion Surge (1)
        sig_wband = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
        if len(df_norm) >= 20:
            try:
                q80 = bb20_w.rolling(window=20, min_periods=5).quantile(0.80)
                w_arr = bb20_w.to_numpy(dtype=float, na_value=np.nan)
                q_arr = q80.to_numpy(dtype=float, na_value=np.nan)
                c_arr = close.to_numpy(dtype=float, na_value=np.nan)
                prev_c_arr = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

                valid_w = (
                    ~np.isnan(w_arr) & ~np.isnan(q_arr) & ~np.isnan(c_arr) & ~np.isnan(prev_c_arr)
                )
                surge = valid_w & (w_arr > q_arr)

                condlist_w = [
                    surge & (c_arr > prev_c_arr),
                    surge & (c_arr < prev_c_arr),
                    valid_w,
                ]
                choicelist_w = [
                    SignalState.BUY,
                    SignalState.SELL,
                    SignalState.HOLD,
                ]
                res_w = np.select(condlist_w, choicelist_w, default=SignalState.NONE)
                sig_wband = pd.Series(res_w, index=df_norm.index, dtype=str)
            except Exception:
                pass
        signals["vol_bb_bandwidth_expansion_signal"] = sig_wband
        pbar.update(1)

        # 7. ATR Trailing Stop Direction (2x, 3x) (2)
        signals["vol_atr_trailing_stop_2x_signal"] = _calc_atr_trailing_stop(
            close, high, low, window=14, multiplier=2.0
        )
        signals["vol_atr_trailing_stop_3x_signal"] = _calc_atr_trailing_stop(
            close, high, low, window=14, multiplier=3.0
        )
        pbar.update(2)

        # 8. Chaikin Volatility Surge (1)
        signals["vol_chaikin_volatility_surge_signal"] = _calc_chaikin_volatility(
            high, low, length=10, roc_length=10
        )
        pbar.update(1)

        # 9. Historical Volatility Ratio (10/30) Breakout (1)
        signals["vol_hv_ratio_breakout_10_30_signal"] = _calc_hv_ratio_breakout(
            close, window_fast=10, window_slow=30, threshold=1.5
        )
        pbar.update(1)

        # 10. BB Rejection (1)
        rsi14 = ta.momentum.RSIIndicator(close=close, window=14, fillna=False).rsi()
        signals["vol_bb_rejection_20_signal"] = _calc_bb_rejection(
            open_p, high, low, close, bb20_l, bb20_h, rsi14
        )
        pbar.update(1)

        # 11. BB Bandwidth Regime (1)
        signals["vol_bb_bandwidth_regime_signal"] = _calc_bb_bandwidth_regime(bb20_w)
        pbar.update(1)

        # 12. Volatility Compression Breakout (1)
        signals["vol_compression_breakout_5_20_signal"] = _calc_compression_breakout(close)
        pbar.update(1)

        # 13. ATR Expansion (1)
        signals["vol_atr_expansion_14_20_signal"] = _calc_atr_expansion(high, low, close)
        pbar.update(1)

        # 14. 10-bar Range Breakout (1)
        signals["vol_range_breakout_10_signal"] = _calc_range_breakout(high, low, close, window=10)
        pbar.update(1)

        # 15. Volatility Break (1)
        signals["vol_volatility_break_10_50_signal"] = _calc_volatility_break(
            close, fast=10, slow=50
        )
        pbar.update(1)

        # 16. Range Compression (1)
        signals["vol_range_compression_20_signal"] = _calc_range_compression(
            high, low, close, bb20_m
        )
        pbar.update(1)

        # 17. Volatility Drop (1)
        signals["vol_volatility_drop_5_20_signal"] = _calc_volatility_drop(close, bb20_m)
        pbar.update(1)

        # 18. Range Expansion (1)
        signals["vol_range_expansion_20_signal"] = _calc_range_expansion(open_p, high, low, close)
        pbar.update(1)

        # 19. Range Shift (1)
        signals["vol_range_shift_5_10_signal"] = _calc_range_shift(high, low)
        pbar.update(1)

        # 20. Range Position (1)
        signals["vol_range_position_10_signal"] = _calc_range_position(high, low, close, window=10)
        pbar.update(1)

        # 21. Range Flip (1)
        signals["vol_range_flip_5_signal"] = _calc_range_breakout(high, low, close, window=5)
        pbar.update(1)

        # 22. Linear Regression Channel Reversal (1)
        signals["vol_linreg_channel_reversal_20_signal"] = _calc_linreg_channel_reversal(
            close, length=20
        )
        pbar.update(1)

        # 23. Keltner Channel Reversal (1)
        signals["vol_keltner_reversal_20_signal"] = _calc_keltner_reversal(
            close, kc20.keltner_channel_lband(), kc20.keltner_channel_hband()
        )
        pbar.update(1)

        # 24. Moving Average Envelope Breakout (1)
        signals["vol_envelope_breakout_20_signal"] = _calc_envelope_breakout(
            close, bb20_m, pct=0.025
        )
        pbar.update(1)

        # Ensure all columns are present, filled with NONE, and matching index
        for col in VOLATILITY_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[VOLATILITY_SIGNAL_COLUMNS]
