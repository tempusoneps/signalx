from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
from signalx.signals.session_helper import extract_session_context
from signalx.utils import normalize_ohlcv

VOLUME_SIGNAL_COLUMNS = [
    "volume_obv_ema_cross_20_signal",
    "volume_cmf_zero_cross_20_signal",
    "volume_cmf_threshold_cross_20_signal",
    "volume_vwap_cross_20_signal",
    "volume_vwap_cross_50_signal",
    "volume_vwap_cross_100_signal",
    "volume_vwap_band_reversal_20_signal",
    "volume_spike_direction_20_signal",
    "volume_pvt_ma_cross_14_signal",
    "volume_adl_ma_cross_signal",
    "volume_force_index_13_signal",
    "volume_eom_zero_14_signal",
    "volume_vsa_confirmation_20_signal",
    "volume_price_confirmation_20_signal",
    "volume_vpt_divergence_5_signal",
    "volume_trend_3_bar_signal",
    "volume_price_divergence_signal",
    "volume_amv_cross_20_signal",
    "volume_klinger_osc_cross_signal",
    "volume_elder_ray_bull_bear_signal",
    "volume_climax_absorption_signal",
    "volume_twiggs_money_flow_cross_signal",
    "volume_nvi_pvi_cross_signal",
    "volume_vwap_anchored_dev1_signal",
    "volume_vwap_anchored_dev3_signal",
    "volume_delta_proxy_surge_signal",
    "volume_vwma_sma_divergence_signal",
    "volume_volume_weighted_rsi_14_signal",
    "volume_session_vwap_cross_signal",
    "volume_rvol_time_bucket_signal",
    "volume_cvd_divergence_signal",
    "volume_stopping_climax_signal",
]


def _crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Helper to convert two series into buy/sell/hold/none based on cross and position."""
    n = len(fast)
    if n == 0:
        return pd.Series(dtype=str, index=fast.index)

    fast_arr = fast.to_numpy(dtype=float, na_value=np.nan)
    slow_arr = slow.to_numpy(dtype=float, na_value=np.nan)

    valid_mask = ~np.isnan(fast_arr) & ~np.isnan(slow_arr)
    if not valid_mask.any():
        return pd.Series(SignalState.NONE, index=fast.index, dtype=str)

    prev_fast = fast.shift(1).to_numpy(dtype=float, na_value=np.nan)
    prev_slow = slow.shift(1).to_numpy(dtype=float, na_value=np.nan)
    prev_valid = ~np.isnan(prev_fast) & ~np.isnan(prev_slow)

    cross_up = valid_mask & prev_valid & (fast_arr > slow_arr) & (prev_fast <= prev_slow)
    cross_down = valid_mask & prev_valid & (fast_arr < slow_arr) & (prev_fast >= prev_slow)
    bullish = valid_mask & (fast_arr > slow_arr)

    condlist = [
        cross_up,
        cross_down,
        bullish,
        valid_mask,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
        SignalState.NONE,
    ]
    res_arr = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res_arr, index=fast.index, dtype=str)


def _bound_signal(
    val: pd.Series, lower: float, upper: float, *, buy_below: bool = True
) -> pd.Series:
    """Helper to generate buy/sell/hold signals based on upper and lower bounds."""
    arr = val.to_numpy(dtype=float, na_value=np.nan)
    valid = ~np.isnan(arr)

    if buy_below:
        condlist = [
            valid & (arr < lower),
            valid & (arr > upper),
            valid,
        ]
    else:
        condlist = [
            valid & (arr > upper),
            valid & (arr < lower),
            valid,
        ]

    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res_arr = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res_arr, index=val.index, dtype=str)


def _calc_rolling_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int
) -> pd.Series:
    """Calculate rolling Volume Weighted Average Price over a given window."""
    if len(close) < window or window <= 0:
        return pd.Series(np.nan, index=close.index)
    tp = (high + low + close) / 3.0
    pv = tp * volume
    vol_sum = volume.rolling(window).sum()
    pv_sum = pv.rolling(window).sum()
    vwap = pv_sum / vol_sum.replace(0, np.nan)
    # Fallback to rolling average of typical price if volume is zero or unavailable
    return vwap.fillna(tp.rolling(window).mean())


def _calc_vwap_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate rolling VWAP and standard deviation bands (lower and upper)."""
    if len(close) < window or window <= 0:
        nan_s = pd.Series(np.nan, index=close.index)
        return nan_s, nan_s, nan_s

    tp = (high + low + close) / 3.0
    pv = tp * volume
    vol_sum = volume.rolling(window).sum()
    pv_sum = pv.rolling(window).sum()
    vwap = pv_sum / vol_sum.replace(0, np.nan)
    vwap = vwap.fillna(tp.rolling(window).mean())

    vwap_var = (volume * (tp - vwap) ** 2).rolling(window).sum() / vol_sum.replace(0, np.nan)
    std_vwap = np.sqrt(vwap_var.clip(lower=0.0))
    std_vwap = std_vwap.fillna(close.rolling(window).std())

    upper_band = vwap + num_std * std_vwap
    lower_band = vwap - num_std * std_vwap
    return vwap, lower_band, upper_band


def _calc_vwap_band_reversal(close: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Helper to detect VWAP standard deviation band mean-reversion bounces."""
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


def _calc_volume_spike_direction(
    open_p: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
    threshold: float = 2.0,
) -> pd.Series:
    """Calculate volume spike signal combined with directional candle body."""
    if len(volume) < window or window <= 0:
        return pd.Series(SignalState.NONE, index=volume.index, dtype=str)

    vol_sma = volume.rolling(window).mean()
    v_arr = volume.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = vol_sma.to_numpy(dtype=float, na_value=np.nan)
    o_arr = open_p.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(v_arr) & ~np.isnan(sma_arr) & ~np.isnan(o_arr) & ~np.isnan(c_arr)
    is_spike = valid & (v_arr > threshold * sma_arr)

    condlist = [
        is_spike & (c_arr > o_arr),
        is_spike & (c_arr < o_arr),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=volume.index, dtype=str)


def _calc_vsa_confirmation(
    open_p: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Calculate Volume Spread Analysis (VSA) confirmation (> 1.2x SMA20 Volume)."""
    vol_sma = volume.rolling(window, min_periods=5).mean().to_numpy(dtype=float, na_value=np.nan)
    v = volume.to_numpy(dtype=float, na_value=np.nan)
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(v) & ~np.isnan(vol_sma) & ~np.isnan(o) & ~np.isnan(c)
    vol_surge = valid & (v > (1.2 * vol_sma))
    bull = vol_surge & (c > o)
    bear = vol_surge & (c < o)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_volume_price_confirmation(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Calculate Volume Price Confirmation (Volume > SMA20 with directional close)."""
    vol_sma = volume.rolling(window, min_periods=5).mean().to_numpy(dtype=float, na_value=np.nan)
    v = volume.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(v) & ~np.isnan(vol_sma) & ~np.isnan(c) & ~np.isnan(c1)
    has_vol = valid & (v > vol_sma)
    bull = has_vol & (c > c1)
    bear = has_vol & (c < c1)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_vpt_divergence(
    high: pd.Series, low: pd.Series, vpt: pd.Series, lookback: int = 5
) -> pd.Series:
    """Calculate 5-bar regular Volume Price Trend (VPT) divergence."""
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    v_arr = vpt.to_numpy(dtype=float, na_value=np.nan)

    l_prev = low.shift(lookback).to_numpy(dtype=float, na_value=np.nan)
    h_prev = high.shift(lookback).to_numpy(dtype=float, na_value=np.nan)
    v_prev = vpt.shift(lookback).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(l_arr) & ~np.isnan(v_arr) & ~np.isnan(l_prev) & ~np.isnan(v_prev)
    bull = valid & (l_arr < l_prev) & (v_arr > v_prev)
    bear = valid & (h_arr > h_prev) & (v_arr < v_prev)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=low.index, dtype=str
    )


def _calc_volume_trend(volume: pd.Series) -> pd.Series:
    """Calculate 3-bar Volume Trend (expansion vs contraction)."""
    v = volume.to_numpy(dtype=float, na_value=np.nan)
    v1 = volume.shift(1).to_numpy(dtype=float, na_value=np.nan)
    v2 = volume.shift(2).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(v) & ~np.isnan(v1) & ~np.isnan(v2)
    bull = valid & (v > v1) & (v1 > v2)
    bear = valid & (v < v1) & (v1 < v2)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=volume.index, dtype=str
    )


def _calc_volume_price_divergence(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume-Price Divergence (absorption vs churn)."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    v = volume.to_numpy(dtype=float, na_value=np.nan)
    v1 = volume.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(c1) & ~np.isnan(v) & ~np.isnan(v1)
    bull = valid & (c < c1) & (v > v1)
    bear = valid & (c > c1) & (v > v1)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_amv(close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    """Calculate Adaptive Moving Volume (AMV) cross signal."""
    pv = close * volume
    vwap_roll = pv.rolling(length, min_periods=5).sum() / volume.rolling(
        length, min_periods=5
    ).sum().replace(0, np.nan)
    return _crossover_signal(close, vwap_roll)


def _calc_klinger_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Calculate Klinger Volume Oscillator (34, 55, 13) signal cross."""
    n = len(close)
    if n < 5:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    tp = (high + low + close) / 3.0
    tp_arr = tp.to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    v_arr = volume.to_numpy(dtype=float, na_value=np.nan)

    dm = h_arr - l_arr
    trend = np.ones(n, dtype=float)
    for i in range(1, n):
        if np.isnan(tp_arr[i]) or np.isnan(tp_arr[i - 1]):
            trend[i] = trend[i - 1]
        elif tp_arr[i] > tp_arr[i - 1]:
            trend[i] = 1.0
        elif tp_arr[i] < tp_arr[i - 1]:
            trend[i] = -1.0
        else:
            trend[i] = trend[i - 1]

    cm = np.zeros(n, dtype=float)
    cm[0] = dm[0] if not np.isnan(dm[0]) else 0.0
    for i in range(1, n):
        if trend[i] == trend[i - 1]:
            cm[i] = cm[i - 1] + (dm[i] if not np.isnan(dm[i]) else 0.0)
        else:
            cm[i] = (dm[i - 1] if not np.isnan(dm[i - 1]) else 0.0) + (
                dm[i] if not np.isnan(dm[i]) else 0.0
            )

    ratio = np.where(cm != 0, dm / cm, 0.0)
    vf = v_arr * np.abs(2.0 * ratio - 1.0) * trend * 100.0
    vf_s = pd.Series(vf, index=close.index)
    kvo = vf_s.ewm(span=34, adjust=False).mean() - vf_s.ewm(span=55, adjust=False).mean()
    kvo_signal = kvo.ewm(span=13, adjust=False).mean()
    return _crossover_signal(kvo, kvo_signal)


def _calc_elder_ray_bull_bear(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Elder Ray Bull & Bear Power signal."""
    if len(close) < 2:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    ema13 = close.ewm(span=13, adjust=False).mean()
    bull_power = high - ema13
    bear_power = low - ema13

    bp = bear_power.to_numpy(dtype=float, na_value=np.nan)
    bp_prev = bear_power.shift(1).to_numpy(dtype=float, na_value=np.nan)
    bull = bull_power.to_numpy(dtype=float, na_value=np.nan)
    bull_prev = bull_power.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(bp) & ~np.isnan(bp_prev) & ~np.isnan(bull) & ~np.isnan(bull_prev)
    buy = valid & (bp < 0.0) & (bp > bp_prev) & (bull > 0.0)
    sell = valid & (bull > 0.0) & (bull < bull_prev) & (bp < 0.0)
    hold = valid & ~buy & ~sell

    conds = [buy, sell, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_volume_climax_absorption(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate Volume Climax Absorption signal (> 3.0x SMA20 with long wick absorption)."""
    vol_sma = volume.rolling(window, min_periods=5).mean().to_numpy(dtype=float, na_value=np.nan)
    v = volume.to_numpy(dtype=float, na_value=np.nan)
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    lo = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(v)
        & ~np.isnan(vol_sma)
        & ~np.isnan(o)
        & ~np.isnan(h)
        & ~np.isnan(lo)
        & ~np.isnan(c)
    )
    candle_range = h - lo
    has_range = valid & (candle_range > 1e-9)
    is_climax = has_range & (v >= 3.0 * vol_sma)

    lower_wick = np.minimum(o, c) - lo
    upper_wick = h - np.maximum(o, c)

    bull = is_climax & (lower_wick >= 0.40 * candle_range) & (c > o)
    bear = is_climax & (upper_wick >= 0.40 * candle_range) & (c < o)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_twiggs_money_flow(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 21
) -> pd.Series:
    """Calculate Twiggs Money Flow (21 period) zero centerline cross."""
    c_prev = close.shift(1)
    trh = np.maximum(high, c_prev)
    trl = np.minimum(low, c_prev)
    tr = trh - trl

    adv = np.where(tr > 1e-9, volume * (2.0 * close - trl - trh) / tr, 0.0)
    adv_s = pd.Series(adv, index=close.index)
    ema_adv = adv_s.ewm(span=length, adjust=False).mean()
    ema_vol = volume.ewm(span=length, adjust=False).mean().replace(0, np.nan)
    tmf = (ema_adv / ema_vol).fillna(0.0)
    return _crossover_signal(tmf, pd.Series(0.0, index=close.index))


def _calc_nvi_cross(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Negative Volume Index (NVI) 20-period EMA crossover."""
    n = len(close)
    if n < 2:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    v_arr = volume.to_numpy(dtype=float, na_value=np.nan)

    nvi = np.zeros(n, dtype=float)
    nvi[0] = 1000.0
    for i in range(1, n):
        if (
            np.isnan(c_arr[i])
            or np.isnan(c_arr[i - 1])
            or np.isnan(v_arr[i])
            or np.isnan(v_arr[i - 1])
        ):
            nvi[i] = nvi[i - 1]
        elif v_arr[i] < v_arr[i - 1] and c_arr[i - 1] != 0.0:
            ret = (c_arr[i] - c_arr[i - 1]) / c_arr[i - 1]
            nvi[i] = nvi[i - 1] + nvi[i - 1] * ret
        else:
            nvi[i] = nvi[i - 1]

    nvi_s = pd.Series(nvi, index=close.index)
    nvi_ema20 = nvi_s.ewm(span=20, adjust=False).mean()
    return _crossover_signal(nvi_s, nvi_ema20)


def _calc_vwap_dev1_bounce(
    high: pd.Series, low: pd.Series, close: pd.Series, lower: pd.Series, upper: pd.Series
) -> pd.Series:
    """Calculate Rolling 20-period VWAP +-1.0 std band bounce & rejection."""
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    l_band = lower.to_numpy(dtype=float, na_value=np.nan)
    u_band = upper.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(l_arr)
        & ~np.isnan(h_arr)
        & ~np.isnan(c_arr)
        & ~np.isnan(l_band)
        & ~np.isnan(u_band)
    )
    buy = valid & (l_arr <= l_band) & (c_arr > l_band)
    sell = valid & (h_arr >= u_band) & (c_arr < u_band)
    hold = valid & ~buy & ~sell

    conds = [buy, sell, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_vwap_dev3_extreme(
    high: pd.Series, low: pd.Series, close: pd.Series, lower: pd.Series, upper: pd.Series
) -> pd.Series:
    """Calculate Rolling 20-period VWAP +-3.0 std extreme touch/cross mean reversion."""
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    l_band = lower.to_numpy(dtype=float, na_value=np.nan)
    u_band = upper.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(l_arr)
        & ~np.isnan(h_arr)
        & ~np.isnan(c_arr)
        & ~np.isnan(l_band)
        & ~np.isnan(u_band)
    )
    buy = valid & ((l_arr <= l_band) | (c_arr <= l_band))
    sell = valid & ((h_arr >= u_band) | (c_arr >= u_band))
    hold = valid & ~buy & ~sell

    conds = [buy, sell, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_volume_delta_proxy_surge(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Calculate Intrabar Delta Volume Proxy surge."""
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    lo = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    v = volume.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(lo) & ~np.isnan(c) & ~np.isnan(v)
    rng = h - lo + 1e-9
    delta_buy = v * (c - lo) / rng
    delta_sell = v * (h - c) / rng

    buy = valid & (delta_buy > 0.70 * v) & (c > o)
    sell = valid & (delta_sell > 0.70 * v) & (c < o)
    hold = valid & ~buy & ~sell

    conds = [buy, sell, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_vwma_sma_divergence(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate VWMA(20) vs SMA(20) Divergence signal."""
    pv = close * volume
    vol_sum = volume.rolling(window, min_periods=5).sum()
    pv_sum = pv.rolling(window, min_periods=5).sum()
    sma = close.rolling(window, min_periods=5).mean()
    vwma = (pv_sum / vol_sum.replace(0, np.nan)).fillna(sma)

    vwma_arr = vwma.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = sma.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(vwma_arr) & ~np.isnan(sma_arr)
    buy = valid & (vwma_arr > sma_arr * 1.005)
    sell = valid & (vwma_arr < sma_arr * 0.995)
    hold = valid & ~buy & ~sell

    conds = [buy, sell, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_volume_weighted_rsi(close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Volume-Weighted RSI (14) overbought/oversold boundaries."""
    delta = close.diff()
    d_arr = delta.to_numpy(dtype=float, na_value=np.nan)
    v_arr = volume.to_numpy(dtype=float, na_value=np.nan)

    u = np.where(d_arr > 0, d_arr * v_arr, 0.0)
    d = np.where(d_arr < 0, -d_arr * v_arr, 0.0)

    u_s = pd.Series(u, index=close.index)
    d_s = pd.Series(d, index=close.index)

    smooth_u = u_s.ewm(span=length, adjust=False).mean()
    smooth_d = d_s.ewm(span=length, adjust=False).mean().replace(0, np.nan)
    rs = (smooth_u / smooth_d).fillna(1.0)
    vrsi = 100.0 - (100.0 / (1.0 + rs))

    vrsi_arr = vrsi.to_numpy(dtype=float, na_value=np.nan)
    vrsi_prev = vrsi.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(vrsi_arr) & ~np.isnan(vrsi_prev)
    buy = valid & (vrsi_prev <= 30.0) & (vrsi_arr > 30.0)
    sell = valid & (vrsi_prev >= 70.0) & (vrsi_arr < 70.0)
    hold = valid & ~buy & ~sell

    conds = [buy, sell, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_session_vwap_cross(
    df: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Cumulative Session VWAP crossover signal."""
    if len(close) == 0:
        return pd.Series(dtype=str, index=close.index)

    ctx = extract_session_context(df)
    tp = (high + low + close) / 3.0
    pv = tp * volume

    cum_pv = pv.groupby(ctx.session_id, sort=False).cumsum()
    cum_vol = volume.groupby(ctx.session_id, sort=False).cumsum()
    session_vwap = (cum_pv / cum_vol.replace(0, np.nan)).fillna(tp)

    sig = _crossover_signal(close, session_vwap)
    return pd.Series(
        np.where(ctx.bar_in_session == 0, SignalState.NONE, sig),
        index=close.index,
        dtype=str,
    )


def _calc_rvol_time_bucket(
    df: pd.DataFrame,
    open_p: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Relative Volume (RVOL) normalized by time-of-day bucket."""
    if len(close) == 0:
        return pd.Series(dtype=str, index=close.index)

    ctx = extract_session_context(df)
    bucket = ctx.time_minutes

    vol_shift = volume.groupby(bucket, sort=False).shift(1)
    mean_vol = vol_shift.groupby(bucket, sort=False).transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )

    v_arr = volume.to_numpy(dtype=float, na_value=np.nan)
    m_arr = mean_vol.to_numpy(dtype=float, na_value=np.nan)
    o_arr = open_p.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(v_arr) & ~np.isnan(m_arr) & (m_arr > 0.0) & ~np.isnan(o_arr) & ~np.isnan(c_arr)
    )
    rvol = np.where(valid, v_arr / m_arr, 0.0)

    buy = valid & (rvol >= 2.0) & (c_arr > o_arr)
    sell = valid & (rvol >= 2.0) & (c_arr < o_arr)

    condlist = [buy, sell]
    choicelist = [SignalState.BUY, SignalState.SELL]
    return pd.Series(
        np.select(condlist, choicelist, default=SignalState.NONE),
        index=close.index,
        dtype=str,
    )


def _calc_cvd_divergence(
    df: pd.DataFrame,
    open_p: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Intraday Proxy Cumulative Volume Delta (CVD) Price Divergence."""
    if len(close) == 0:
        return pd.Series(dtype=str, index=close.index)

    ctx = extract_session_context(df)
    rng = high - low
    delta = volume * (2.0 * close - high - low) / (rng + 1e-9)
    cvd = delta.groupby(ctx.session_id, sort=False).cumsum()

    close_max = close.groupby(ctx.session_id, sort=False).transform(
        lambda s: s.rolling(10, min_periods=3).max()
    )
    close_min = close.groupby(ctx.session_id, sort=False).transform(
        lambda s: s.rolling(10, min_periods=3).min()
    )
    cvd_max = cvd.groupby(ctx.session_id, sort=False).transform(
        lambda s: s.rolling(10, min_periods=3).max()
    )
    cvd_min = cvd.groupby(ctx.session_id, sort=False).transform(
        lambda s: s.rolling(10, min_periods=3).min()
    )

    c = close.to_numpy(dtype=float, na_value=np.nan)
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    cvd_arr = cvd.to_numpy(dtype=float, na_value=np.nan)
    c_max = close_max.to_numpy(dtype=float, na_value=np.nan)
    c_min = close_min.to_numpy(dtype=float, na_value=np.nan)
    cvd_max_arr = cvd_max.to_numpy(dtype=float, na_value=np.nan)
    cvd_min_arr = cvd_min.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(c)
        & ~np.isnan(o)
        & ~np.isnan(cvd_arr)
        & ~np.isnan(c_max)
        & ~np.isnan(c_min)
        & ~np.isnan(cvd_max_arr)
        & ~np.isnan(cvd_min_arr)
    )

    bear_div = valid & (c >= c_max - 1e-9) & (cvd_arr < cvd_max_arr - 1e-9) & (c < o)
    bull_div = valid & (c <= c_min + 1e-9) & (cvd_arr > cvd_min_arr + 1e-9) & (c > o)

    condlist = [bull_div, bear_div]
    choicelist = [SignalState.BUY, SignalState.SELL]
    return pd.Series(
        np.select(condlist, choicelist, default=SignalState.NONE),
        index=close.index,
        dtype=str,
    )


def _calc_stopping_climax(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Stopping Volume / Intraday Climax Exhaustion Bar."""
    if len(close) == 0:
        return pd.Series(dtype=str, index=close.index)

    vol_sma = volume.rolling(20, min_periods=5).mean().to_numpy(dtype=float, na_value=np.nan)
    v = volume.to_numpy(dtype=float, na_value=np.nan)
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    lo = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(v)
        & ~np.isnan(vol_sma)
        & ~np.isnan(o)
        & ~np.isnan(h)
        & ~np.isnan(lo)
        & ~np.isnan(c)
    )
    rng = h - lo
    has_range = valid & (rng > 1e-9)
    is_surge = has_range & (v >= 2.5 * vol_sma)

    lower_wick = np.minimum(o, c) - lo
    upper_wick = h - np.maximum(o, c)

    buy = is_surge & (lower_wick >= 0.40 * rng) & (c >= lo + 0.30 * rng)
    sell = is_surge & (upper_wick >= 0.40 * rng) & (c <= h - 0.30 * rng)

    condlist = [
        buy & (~sell | (lower_wick >= upper_wick)),
        sell & (~buy | (upper_wick > lower_wick)),
    ]
    choicelist = [SignalState.BUY, SignalState.SELL]
    return pd.Series(
        np.select(condlist, choicelist, default=SignalState.NONE),
        index=close.index,
        dtype=str,
    )


def generate_volume_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Generate all 32 volume, flow, and VWAP signals from OHLCV dataframe.

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
        "Volume Signals", total=len(VOLUME_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if len(df_norm) == 0:
            pbar.update(len(VOLUME_SIGNAL_COLUMNS))
            return pd.DataFrame(
                {col: pd.Series(dtype=str, index=df.index) for col in VOLUME_SIGNAL_COLUMNS},
                index=df.index,
            )

        open_p = df_norm["open"]
        high = df_norm["high"]
        low = df_norm["low"]
        close = df_norm["close"]
        volume = df_norm["volume"]

        signals = pd.DataFrame(index=df_norm.index)

        # 1. On-Balance Volume (OBV) crosses 20-period EMA of OBV (1)
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume, fillna=False
        ).on_balance_volume()
        obv_ema20 = obv.ewm(span=20, adjust=False).mean()
        signals["volume_obv_ema_cross_20_signal"] = _crossover_signal(obv, obv_ema20)
        pbar.update(1)

        # 2. Chaikin Money Flow (CMF, 20-period) zero-line and threshold crosses (2)
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            high=high, low=low, close=close, volume=volume, window=20, fillna=False
        ).chaikin_money_flow()
        signals["volume_cmf_zero_cross_20_signal"] = _crossover_signal(
            cmf, pd.Series(0.0, index=df_norm.index)
        )
        signals["volume_cmf_threshold_cross_20_signal"] = _bound_signal(
            cmf, -0.05, 0.05, buy_below=False
        )
        pbar.update(2)

        # 3. Rolling VWAP crossovers (20, 50, 100 periods) (3)
        vwap20 = _calc_rolling_vwap(high, low, close, volume, 20)
        vwap50 = _calc_rolling_vwap(high, low, close, volume, 50)
        vwap100 = _calc_rolling_vwap(high, low, close, volume, 100)

        signals["volume_vwap_cross_20_signal"] = _crossover_signal(close, vwap20)
        signals["volume_vwap_cross_50_signal"] = _crossover_signal(close, vwap50)
        signals["volume_vwap_cross_100_signal"] = _crossover_signal(close, vwap100)
        pbar.update(3)

        # 4. Rolling VWAP Standard Deviation Bands Reversal (20 period, 2.0 std) (1)
        _, vwap20_lband, vwap20_hband = _calc_vwap_bands(
            high, low, close, volume, window=20, num_std=2.0
        )
        signals["volume_vwap_band_reversal_20_signal"] = _calc_vwap_band_reversal(
            close, vwap20_lband, vwap20_hband
        )
        pbar.update(1)

        # 5. Volume Spike Direction (Volume > 2.0 * 20-period SMA(Volume)) (1)
        signals["volume_spike_direction_20_signal"] = _calc_volume_spike_direction(
            open_p, close, volume, window=20, threshold=2.0
        )
        pbar.update(1)

        # 6. Price Volume Trend (PVT) crosses 14-period SMA (1)
        pvt = ta.volume.VolumePriceTrendIndicator(
            close=close, volume=volume, fillna=False
        ).volume_price_trend()
        pvt_sma14 = pvt.rolling(14).mean()
        signals["volume_pvt_ma_cross_14_signal"] = _crossover_signal(pvt, pvt_sma14)
        pbar.update(1)

        # 7. Accumulation / Distribution Line (ADL) crosses 20-period SMA (1)
        adl = ta.volume.AccDistIndexIndicator(
            high=high, low=low, close=close, volume=volume, fillna=False
        ).acc_dist_index()
        adl_sma20 = adl.rolling(20).mean()
        signals["volume_adl_ma_cross_signal"] = _crossover_signal(adl, adl_sma20)
        pbar.update(1)

        # 8. Elder's Force Index (13 period) zero line crossover (1)
        fi13 = ta.volume.ForceIndexIndicator(
            close=close, volume=volume, window=13, fillna=False
        ).force_index()
        signals["volume_force_index_13_signal"] = _crossover_signal(
            fi13, pd.Series(0.0, index=df_norm.index)
        )
        pbar.update(1)

        # 9. Ease of Movement (14 period) zero line crossover (1)
        eom14 = ta.volume.EaseOfMovementIndicator(
            high=high, low=low, volume=volume, window=14, fillna=False
        ).sma_ease_of_movement()
        signals["volume_eom_zero_14_signal"] = _crossover_signal(
            eom14, pd.Series(0.0, index=df_norm.index)
        )
        pbar.update(1)

        # 10. Volume Spread Analysis (VSA) confirmation (1)
        signals["volume_vsa_confirmation_20_signal"] = _calc_vsa_confirmation(
            open_p, close, volume, window=20
        )
        pbar.update(1)

        # 11. Volume Price Confirmation (1)
        signals["volume_price_confirmation_20_signal"] = _calc_volume_price_confirmation(
            close, volume, window=20
        )
        pbar.update(1)

        # 12. VPT 5-bar Divergence (1)
        signals["volume_vpt_divergence_5_signal"] = _calc_vpt_divergence(high, low, pvt, lookback=5)
        pbar.update(1)

        # 13. Volume 3-bar Trend (1)
        signals["volume_trend_3_bar_signal"] = _calc_volume_trend(volume)
        pbar.update(1)

        # 14. Volume-Price Divergence (1)
        signals["volume_price_divergence_signal"] = _calc_volume_price_divergence(close, volume)
        pbar.update(1)

        # 15. AMV Cross (1)
        signals["volume_amv_cross_20_signal"] = _calc_amv(close, volume, length=20)
        pbar.update(1)

        # 16. Klinger Volume Oscillator Cross (1)
        signals["volume_klinger_osc_cross_signal"] = _calc_klinger_oscillator(
            high, low, close, volume
        )
        pbar.update(1)

        # 17. Elder Ray Bull / Bear Power (1)
        signals["volume_elder_ray_bull_bear_signal"] = _calc_elder_ray_bull_bear(high, low, close)
        pbar.update(1)

        # 18. Volume Climax Absorption (1)
        signals["volume_climax_absorption_signal"] = _calc_volume_climax_absorption(
            open_p, high, low, close, volume, window=20
        )
        pbar.update(1)

        # 19. Twiggs Money Flow Cross (1)
        signals["volume_twiggs_money_flow_cross_signal"] = _calc_twiggs_money_flow(
            high, low, close, volume, length=21
        )
        pbar.update(1)

        # 20. Negative Volume Index EMA Cross (1)
        signals["volume_nvi_pvi_cross_signal"] = _calc_nvi_cross(close, volume)
        pbar.update(1)

        # 21. Rolling VWAP +-1.0 Std Band Reversal (1)
        _, vwap20_l1, vwap20_u1 = _calc_vwap_bands(high, low, close, volume, window=20, num_std=1.0)
        signals["volume_vwap_anchored_dev1_signal"] = _calc_vwap_dev1_bounce(
            high, low, close, vwap20_l1, vwap20_u1
        )
        pbar.update(1)

        # 22. Rolling VWAP +-3.0 Std Band Extreme Mean Reversion (1)
        _, vwap20_l3, vwap20_u3 = _calc_vwap_bands(high, low, close, volume, window=20, num_std=3.0)
        signals["volume_vwap_anchored_dev3_signal"] = _calc_vwap_dev3_extreme(
            high, low, close, vwap20_l3, vwap20_u3
        )
        pbar.update(1)

        # 23. Delta Volume Proxy Surge (1)
        signals["volume_delta_proxy_surge_signal"] = _calc_volume_delta_proxy_surge(
            open_p, high, low, close, volume
        )
        pbar.update(1)

        # 24. VWMA vs SMA Divergence (1)
        signals["volume_vwma_sma_divergence_signal"] = _calc_vwma_sma_divergence(
            close, volume, window=20
        )
        pbar.update(1)

        # 25. Volume-Weighted RSI 14 (1)
        signals["volume_volume_weighted_rsi_14_signal"] = _calc_volume_weighted_rsi(
            close, volume, length=14
        )
        pbar.update(1)

        # 26. Cumulative Session VWAP Cross (1)
        signals["volume_session_vwap_cross_signal"] = _calc_session_vwap_cross(
            df_norm, close, high, low, volume
        )
        pbar.update(1)

        # 27. Relative Volume by Time Bucket (1)
        signals["volume_rvol_time_bucket_signal"] = _calc_rvol_time_bucket(
            df_norm, open_p, close, volume
        )
        pbar.update(1)

        # 28. Intraday CVD Price Divergence (1)
        signals["volume_cvd_divergence_signal"] = _calc_cvd_divergence(
            df_norm, open_p, close, high, low, volume
        )
        pbar.update(1)

        # 29. Stopping Volume / Climax Exhaustion (1)
        signals["volume_stopping_climax_signal"] = _calc_stopping_climax(
            open_p, high, low, close, volume
        )
        pbar.update(1)

        # Ensure all columns are present, filled with NONE, and matching index
        for col in VOLUME_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[VOLUME_SIGNAL_COLUMNS]
