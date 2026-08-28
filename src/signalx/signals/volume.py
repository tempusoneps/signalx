from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from signalx.constants import SignalState
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


def generate_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all 12 standardized volume & order flow signals from normalized OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 12 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)
    signals = pd.DataFrame(index=df_norm.index)

    if df_norm.empty:
        for col in VOLUME_SIGNAL_COLUMNS:
            signals[col] = pd.Series(dtype=str)
        return signals

    open_p = df_norm["open"]
    high = df_norm["high"]
    low = df_norm["low"]
    close = df_norm["close"]
    volume = df_norm["volume"]

    # 1. On-Balance Volume (OBV) crosses its 20-period EMA
    obv = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume, fillna=False
    ).on_balance_volume()
    obv_ema20 = obv.ewm(span=20, adjust=False).mean()
    signals["volume_obv_ema_cross_20_signal"] = _crossover_signal(obv, obv_ema20)

    # 2. Chaikin Money Flow (20 period) zero cross and +/-0.05 threshold signals
    cmf20 = ta.volume.ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=volume, window=20, fillna=False
    ).chaikin_money_flow()
    signals["volume_cmf_zero_cross_20_signal"] = _crossover_signal(
        cmf20, pd.Series(0.0, index=df_norm.index)
    )
    signals["volume_cmf_threshold_cross_20_signal"] = _bound_signal(
        cmf20, lower=-0.05, upper=0.05, buy_below=False
    )

    # 3. Rolling VWAP price crossovers (20, 50, 100)
    vwap20 = _calc_rolling_vwap(high, low, close, volume, window=20)
    signals["volume_vwap_cross_20_signal"] = _crossover_signal(close, vwap20)

    vwap50 = _calc_rolling_vwap(high, low, close, volume, window=50)
    signals["volume_vwap_cross_50_signal"] = _crossover_signal(close, vwap50)

    vwap100 = _calc_rolling_vwap(high, low, close, volume, window=100)
    signals["volume_vwap_cross_100_signal"] = _crossover_signal(close, vwap100)

    # 4. Rolling 20-period VWAP standard deviation band reversal
    _, vwap20_lband, vwap20_hband = _calc_vwap_bands(
        high, low, close, volume, window=20, num_std=2.0
    )
    signals["volume_vwap_band_reversal_20_signal"] = _calc_vwap_band_reversal(
        close, vwap20_lband, vwap20_hband
    )

    # 5. Volume Spike (>2.0x 20-SMA) combined with directional candle body
    signals["volume_spike_direction_20_signal"] = _calc_volume_spike_direction(
        open_p, close, volume, window=20, threshold=2.0
    )

    # 6. Price Volume Trend (PVT) crosses 14-period SMA
    pvt = ta.volume.VolumePriceTrendIndicator(
        close=close, volume=volume, fillna=False
    ).volume_price_trend()
    pvt_sma14 = pvt.rolling(14).mean()
    signals["volume_pvt_ma_cross_14_signal"] = _crossover_signal(pvt, pvt_sma14)

    # 7. Accumulation / Distribution Line (ADL) crosses 20-period SMA
    adl = ta.volume.AccDistIndexIndicator(
        high=high, low=low, close=close, volume=volume, fillna=False
    ).acc_dist_index()
    adl_sma20 = adl.rolling(20).mean()
    signals["volume_adl_ma_cross_signal"] = _crossover_signal(adl, adl_sma20)

    # 8. Elder's Force Index (13 period) zero line crossover
    fi13 = ta.volume.ForceIndexIndicator(
        close=close, volume=volume, window=13, fillna=False
    ).force_index()
    signals["volume_force_index_13_signal"] = _crossover_signal(
        fi13, pd.Series(0.0, index=df_norm.index)
    )

    # 9. Ease of Movement (14 period) zero line crossover
    eom14 = ta.volume.EaseOfMovementIndicator(
        high=high, low=low, volume=volume, window=14, fillna=False
    ).sma_ease_of_movement()
    signals["volume_eom_zero_14_signal"] = _crossover_signal(
        eom14, pd.Series(0.0, index=df_norm.index)
    )

    # Ensure all columns are present, filled with NONE, and matching index
    for col in VOLUME_SIGNAL_COLUMNS:
        if col not in signals.columns:
            signals[col] = SignalState.NONE
        else:
            signals[col] = signals[col].fillna(SignalState.NONE)

    return signals[VOLUME_SIGNAL_COLUMNS]
