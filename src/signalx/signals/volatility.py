from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from signalx.constants import SignalState
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


def generate_volatility_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all 17 standardized volatility & breakout signals from normalized OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 17 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)
    signals = pd.DataFrame(index=df_norm.index)

    if df_norm.empty:
        for col in VOLATILITY_SIGNAL_COLUMNS:
            signals[col] = pd.Series(dtype=str)
        return signals

    close = df_norm["close"]
    high = df_norm["high"]
    low = df_norm["low"]

    # 1. Bollinger Bands (20, 2.0 std) Breakout & Bounce
    bb20 = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0, fillna=False)
    bb20_h = bb20.bollinger_hband()
    bb20_l = bb20.bollinger_lband()
    bb20_m = bb20.bollinger_mavg()
    bb20_p = bb20.bollinger_pband()
    bb20_w = bb20.bollinger_wband()

    signals["vol_bb_breakout_20_20_signal"] = _channel_breakout_signal(close, bb20_l, bb20_h)
    signals["vol_bb_bounce_20_20_signal"] = _calc_bb_bounce(close, bb20_l, bb20_h)

    # 2. Bollinger Bands (50, 2.5 std) Breakout & Bounce
    bb50 = ta.volatility.BollingerBands(close=close, window=50, window_dev=2.5, fillna=False)
    bb50_h = bb50.bollinger_hband()
    bb50_l = bb50.bollinger_lband()

    signals["vol_bb_breakout_50_25_signal"] = _channel_breakout_signal(close, bb50_l, bb50_h)
    signals["vol_bb_bounce_50_25_signal"] = _calc_bb_bounce(close, bb50_l, bb50_h)

    # 3. Bollinger %B Reversals (20, 50)
    signals["vol_bb_pct_b_reversal_20_signal"] = _calc_pct_b_reversal(bb20_p)

    bb50_p = ta.volatility.BollingerBands(
        close=close, window=50, window_dev=2.0, fillna=False
    ).bollinger_pband()
    signals["vol_bb_pct_b_reversal_50_signal"] = _calc_pct_b_reversal(bb50_p)

    # 4. Donchian Channel Breakouts (10, 20, 55)
    for period in [10, 20, 55]:
        col_name = f"vol_donchian_breakout_{period}_signal"
        sig_dc = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
        if len(df_norm) >= period:
            try:
                dc = ta.volatility.DonchianChannel(
                    high=high, low=low, close=close, window=period, fillna=False
                )
                sig_dc = _donchian_breakout_signal(
                    close,
                    dc.donchian_channel_lband().shift(1),
                    dc.donchian_channel_hband().shift(1),
                )
            except Exception:
                pass
        signals[col_name] = sig_dc

    # 5. Keltner Channel Breakouts (20/1.5, 20/2.0)
    for mult, mult_str in [(1.5, "15"), (2.0, "20")]:
        col_name = f"vol_keltner_breakout_20_{mult_str}_signal"
        sig_kc = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
        if len(df_norm) >= 20:
            try:
                kc = ta.volatility.KeltnerChannel(
                    high=high,
                    low=low,
                    close=close,
                    window=20,
                    window_atr=10,
                    multiplier=mult,
                    original_version=False,
                    fillna=False,
                )
                sig_kc = _channel_breakout_signal(
                    close, kc.keltner_channel_lband(), kc.keltner_channel_hband()
                )
            except Exception:
                pass
        signals[col_name] = sig_kc

    # 6. TTM Squeeze Breakout
    sig_ttm = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
    if len(df_norm) >= 20:
        try:
            kc_ttm = ta.volatility.KeltnerChannel(
                high=high,
                low=low,
                close=close,
                window=20,
                window_atr=10,
                multiplier=1.5,
                original_version=False,
                fillna=False,
            )
            sig_ttm = _calc_ttm_squeeze(
                bb20_h,
                bb20_l,
                kc_ttm.keltner_channel_hband(),
                kc_ttm.keltner_channel_lband(),
                close,
                bb20_m,
            )
        except Exception:
            pass
    signals["vol_ttm_squeeze_signal"] = sig_ttm

    # 7. Bollinger Bandwidth Expansion Surge
    sig_wband = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
    if len(df_norm) >= 20:
        try:
            q80 = bb20_w.rolling(window=20, min_periods=5).quantile(0.80)
            w_arr = bb20_w.to_numpy(dtype=float, na_value=np.nan)
            q_arr = q80.to_numpy(dtype=float, na_value=np.nan)
            c_arr = close.to_numpy(dtype=float, na_value=np.nan)
            prev_c_arr = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

            valid_w = ~np.isnan(w_arr) & ~np.isnan(q_arr) & ~np.isnan(c_arr) & ~np.isnan(prev_c_arr)
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

    # 8. ATR Trailing Stop Direction (2x, 3x)
    signals["vol_atr_trailing_stop_2x_signal"] = _calc_atr_trailing_stop(
        close, high, low, window=14, multiplier=2.0
    )
    signals["vol_atr_trailing_stop_3x_signal"] = _calc_atr_trailing_stop(
        close, high, low, window=14, multiplier=3.0
    )

    # 9. Chaikin Volatility Surge
    signals["vol_chaikin_volatility_surge_signal"] = _calc_chaikin_volatility(
        high, low, length=10, roc_length=10
    )

    # 10. Historical Volatility Ratio (10/30) Breakout
    signals["vol_hv_ratio_breakout_10_30_signal"] = _calc_hv_ratio_breakout(
        close, window_fast=10, window_slow=30, threshold=1.5
    )

    # Ensure all columns are present, filled with NONE, and matching index
    for col in VOLATILITY_SIGNAL_COLUMNS:
        if col not in signals.columns:
            signals[col] = SignalState.NONE
        else:
            signals[col] = signals[col].fillna(SignalState.NONE)

    return signals[VOLATILITY_SIGNAL_COLUMNS]
