from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as pta
import ta

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
from signalx.signals.session_helper import SessionContext, extract_session_context
from signalx.utils import normalize_ohlcv

TREND_SIGNAL_COLUMNS = [
    "trend_sma_cross_5_20_signal",
    "trend_sma_cross_10_50_signal",
    "trend_sma_cross_20_50_signal",
    "trend_golden_cross_50_200_signal",
    "trend_ema_cross_9_21_signal",
    "trend_ema_cross_12_26_signal",
    "trend_ema_cross_50_200_signal",
    "trend_dema_cross_10_30_signal",
    "trend_tema_cross_10_30_signal",
    "trend_hma_cross_9_21_signal",
    "trend_vwma_cross_10_30_signal",
    "trend_price_above_sma20_signal",
    "trend_price_above_ema50_signal",
    "trend_price_above_ema200_signal",
    "trend_macd_cross_signal",
    "trend_macd_zero_cross_signal",
    "trend_macd_hist_reversal_signal",
    "trend_macd_fast_cross_signal",
    "trend_macd_slow_cross_signal",
    "trend_supertrend_10_3_signal",
    "trend_supertrend_7_2_signal",
    "trend_supertrend_14_4_signal",
    "trend_psar_reversal_signal",
    "trend_aroon_cross_14_signal",
    "trend_aroon_cross_25_signal",
    "trend_adx_dmi_14_signal",
    "trend_adx_dmi_28_signal",
    "trend_ichimoku_tk_cross_signal",
    "trend_ichimoku_cloud_breakout_signal",
    "trend_vortex_cross_14_signal",
    "trend_trix_cross_15_signal",
    "trend_kama_reversal_10_signal",
    "trend_tma_cross_10_signal",
    "trend_ma_alignment_20_50_signal",
    "trend_pullback_sma20_50_signal",
    "trend_micro_trend_3_signal",
    "trend_micro_reversal_signal",
    "trend_price_cross_sma20_signal",
    "trend_prev_bar_break_signal",
    "trend_tii_14_signal",
    "trend_ehlers_super_smoother_cross_signal",
    "trend_mcginley_dynamic_cross_signal",
    "trend_gmma_ribbon_expansion_signal",
    "trend_gmma_compression_breakout_signal",
    "trend_rainbow_ema_alignment_signal",
    "trend_ehlers_instantaneous_trend_signal",
    "trend_coral_trend_filter_signal",
    "trend_supertrend_atr_20_5_signal",
    "trend_donchian_middle_cross_20_signal",
    "trend_alligator_lips_jaw_cross_signal",
    "trend_alma_cross_9_signal",
    "trend_zero_lag_ema_cross_21_signal",
    "trend_vn30_prior_auction_bias_signal",
    "trend_vn30_asymmetric_persistence_signal",
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


def _calc_dema(close: pd.Series, length: int) -> pd.Series:
    """Calculate Double Exponential Moving Average (DEMA) with fallback."""
    try:
        res = pta.dema(close, length=length)
        if res is not None and isinstance(res, pd.Series) and not res.empty:
            return res
    except Exception:
        pass
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return 2 * ema1 - ema2


def _calc_tema(close: pd.Series, length: int) -> pd.Series:
    """Calculate Triple Exponential Moving Average (TEMA) with fallback."""
    try:
        res = pta.tema(close, length=length)
        if res is not None and isinstance(res, pd.Series) and not res.empty:
            return res
    except Exception:
        pass
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3


def _calc_wma(s: pd.Series, length: int) -> pd.Series:
    """Calculate Weighted Moving Average (WMA)."""
    if len(s) < length or length <= 0:
        return pd.Series(np.nan, index=s.index)
    weights = np.arange(1, length + 1)
    return s.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _calc_hma(close: pd.Series, length: int) -> pd.Series:
    """Calculate Hull Moving Average (HMA) with fallback."""
    try:
        res = pta.hma(close, length=length)
        if res is not None and isinstance(res, pd.Series) and not res.empty:
            return res
    except Exception:
        pass
    half_len = max(int(length / 2), 1)
    sqrt_len = max(int(np.sqrt(length)), 1)
    wma_half = _calc_wma(close, half_len)
    wma_full = _calc_wma(close, length)
    diff = 2 * wma_half - wma_full
    return _calc_wma(diff, sqrt_len)


def _calc_vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Calculate Volume-Weighted Moving Average (VWMA) with fallback."""
    try:
        res = pta.vwma(close, volume, length=length)
        if res is not None and isinstance(res, pd.Series) and not res.empty:
            return res
    except Exception:
        pass
    pv = close * volume
    vol_sum = volume.rolling(length).sum()
    pv_sum = pv.rolling(length).sum()
    return pv_sum / vol_sum.replace(0, np.nan)


def _price_above_ma(close: pd.Series, ma: pd.Series) -> pd.Series:
    """Helper to signal whether price is above or below a moving average baseline."""
    close_arr = close.to_numpy(dtype=float, na_value=np.nan)
    ma_arr = ma.to_numpy(dtype=float, na_value=np.nan)
    valid = ~np.isnan(close_arr) & ~np.isnan(ma_arr)

    condlist = [
        valid & (close_arr > ma_arr),
        valid & (close_arr < ma_arr),
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate standard MACD line, signal line, and histogram."""
    macd_ind = ta.trend.MACD(
        close, window_fast=fast, window_slow=slow, window_sign=signal, fillna=False
    )
    return macd_ind.macd(), macd_ind.macd_signal(), macd_ind.macd_diff()


def _calc_macd_hist_reversal(macd_diff: pd.Series) -> pd.Series:
    """Calculate MACD Histogram sign turn reversal signal."""
    macd_diff_arr = macd_diff.to_numpy(dtype=float, na_value=np.nan)
    prev_diff_arr = macd_diff.shift(1).to_numpy(dtype=float, na_value=np.nan)
    valid_hist = ~np.isnan(macd_diff_arr) & ~np.isnan(prev_diff_arr)
    turn_pos = valid_hist & (macd_diff_arr > 0) & (prev_diff_arr <= 0)
    turn_neg = valid_hist & (macd_diff_arr < 0) & (prev_diff_arr >= 0)
    holding_pos = valid_hist & (macd_diff_arr > 0)

    conds_hist = [turn_pos, turn_neg, holding_pos]
    choices_hist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res_hist = np.select(conds_hist, choices_hist, default=SignalState.NONE)
    return pd.Series(res_hist, index=macd_diff.index, dtype=str)


def _calc_supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int, multiplier: float
) -> pd.Series:
    """Calculate SuperTrend trend direction signal."""
    res_st = np.full(len(close), SignalState.NONE, dtype=object)
    try:
        st = pta.supertrend(high=high, low=low, close=close, length=length, multiplier=multiplier)
        if st is not None and not st.empty:
            dir_cols = [c for c in st.columns if "d" in c.lower() or "dir" in c.lower()]
            if dir_cols:
                st_dir = st[dir_cols[0]].to_numpy(dtype=float, na_value=np.nan)
                valid_st = ~np.isnan(st_dir)
                conds_st = [valid_st & (st_dir > 0), valid_st & (st_dir < 0)]
                choices_st = [SignalState.BUY, SignalState.SELL]
                res_st = np.select(conds_st, choices_st, default=SignalState.NONE)
    except Exception:
        pass
    return pd.Series(res_st, index=close.index, dtype=str)


def _calc_psar(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Parabolic SAR direction and flip reversal signal."""
    res_psar = np.full(len(close), SignalState.NONE, dtype=object)
    if len(close) >= 2:
        try:
            psar = ta.trend.PSARIndicator(
                high=high, low=low, close=close, step=0.02, max_step=0.2, fillna=False
            )
            up_ind = psar.psar_up_indicator().to_numpy(dtype=float, na_value=np.nan)
            down_ind = psar.psar_down_indicator().to_numpy(dtype=float, na_value=np.nan)
            psar_val = psar.psar().to_numpy(dtype=float, na_value=np.nan)
            close_arr = close.to_numpy(dtype=float, na_value=np.nan)

            valid_psar = ~np.isnan(up_ind) & ~np.isnan(down_ind)
            flip_buy = valid_psar & (up_ind == 1.0)
            flip_sell = valid_psar & (down_ind == 1.0)
            holding_bullish = (
                valid_psar & ~flip_buy & ~flip_sell & ~np.isnan(psar_val) & (close_arr > psar_val)
            )

            conds_psar = [flip_buy, flip_sell, holding_bullish]
            choices_psar = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
            res_psar = np.select(conds_psar, choices_psar, default=SignalState.NONE)
        except Exception:
            pass
    return pd.Series(res_psar, index=close.index, dtype=str)


def _calc_aroon_cross(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Calculate Aroon oscillator crossover signal."""
    if len(high) < window:
        return pd.Series(SignalState.NONE, index=high.index, dtype=str)
    try:
        aroon = ta.trend.AroonIndicator(high=high, low=low, window=window, fillna=False)
        return _crossover_signal(aroon.aroon_up(), aroon.aroon_down())
    except Exception:
        return pd.Series(SignalState.NONE, index=high.index, dtype=str)


def _calc_adx_dmi(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate ADX / DMI directional strength signal."""
    res_adx = np.full(len(close), SignalState.NONE, dtype=object)
    if len(close) >= window:
        try:
            adx_ind = ta.trend.ADXIndicator(
                high=high, low=low, close=close, window=window, fillna=False
            )
            adx_val = adx_ind.adx().to_numpy(dtype=float, na_value=np.nan)
            pdi = adx_ind.adx_pos().to_numpy(dtype=float, na_value=np.nan)
            ndi = adx_ind.adx_neg().to_numpy(dtype=float, na_value=np.nan)

            valid_adx = ~np.isnan(adx_val) & ~np.isnan(pdi) & ~np.isnan(ndi)
            buy_adx = valid_adx & (pdi > ndi) & (adx_val > 25.0)
            sell_adx = valid_adx & (ndi > pdi) & (adx_val > 25.0)
            hold_adx = valid_adx & (pdi > ndi) & (adx_val <= 25.0)

            conds_adx = [buy_adx, sell_adx, hold_adx]
            choices_adx = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
            res_adx = np.select(conds_adx, choices_adx, default=SignalState.NONE)
        except Exception:
            pass
    return pd.Series(res_adx, index=close.index, dtype=str)


def _calc_trix(close: pd.Series, length: int = 15, signal_length: int = 9) -> pd.Series:
    """Calculate TRIX indicator and signal line crossover."""
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, np.nan) * 100.0
    trix_sig = trix.ewm(span=signal_length, adjust=False).mean()
    return _crossover_signal(trix, trix_sig)


def _calc_kama_reversal(close: pd.Series, length: int = 10) -> pd.Series:
    """Calculate Kaufman Adaptive Moving Average (KAMA) trend reversal."""
    try:
        kama = pta.kama(close, length=length)
    except Exception:
        kama = None
    if kama is None or not isinstance(kama, pd.Series) or kama.empty:
        kama = close.ewm(span=length, adjust=False).mean()

    k_arr = kama.to_numpy(dtype=float, na_value=np.nan)
    k1 = kama.shift(1).to_numpy(dtype=float, na_value=np.nan)
    k2 = kama.shift(2).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(k_arr) & ~np.isnan(k1) & ~np.isnan(k2)
    bull = valid & (k_arr > k1) & (k1 < k2)
    bear = valid & (k_arr < k1) & (k1 > k2)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_tma_cross(close: pd.Series, length: int = 10) -> pd.Series:
    """Calculate Triangular Moving Average (TMA) price crossover."""
    n1 = (length + 1) // 2
    n2 = length // 2 + 1
    tma = close.rolling(n1, min_periods=1).mean().rolling(n2, min_periods=1).mean()
    return _crossover_signal(close, tma)


def _calc_tii(close: pd.Series, length: int = 14, sma_length: int = 20) -> pd.Series:
    """Calculate Trend Intensity Index (TII) signal."""
    sma = close.rolling(sma_length, min_periods=sma_length // 2).mean()
    pos_dev = (close - sma).clip(lower=0)
    neg_dev = (sma - close).clip(lower=0)
    sum_pos = pos_dev.rolling(length, min_periods=length // 2).sum()
    sum_neg = neg_dev.rolling(length, min_periods=length // 2).sum()
    tii = 100.0 * sum_pos / (sum_pos + sum_neg).replace(0, np.nan)

    tii_arr = tii.to_numpy(dtype=float, na_value=np.nan)
    valid = ~np.isnan(tii_arr)
    bull = valid & (tii_arr > 80.0)
    bear = valid & (tii_arr < 20.0)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_micro_trend(close: pd.Series) -> pd.Series:
    """Calculate 3-bar Micro Trend signal."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c2 = close.shift(2).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(c1) & ~np.isnan(c2)
    bull = valid & (c > c1) & (c1 > c2)
    bear = valid & (c < c1) & (c1 < c2)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_micro_reversal(close: pd.Series) -> pd.Series:
    """Calculate Micro Reversal after 2-bar run signal."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c2 = close.shift(2).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(c1) & ~np.isnan(c2)
    bull = valid & (c > c1) & (c1 < c2)
    bear = valid & (c < c1) & (c1 > c2)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_prev_bar_break(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Previous Bar High/Low Break signal."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    h1 = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l1 = low.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(h1) & ~np.isnan(l1)
    bull = valid & (c > h1)
    bear = valid & (c < l1)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_pullback_sma(close: pd.Series, sma_fast: pd.Series, sma_slow: pd.Series) -> pd.Series:
    """Calculate Pullback to fast SMA within established higher-timeframe trend."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    f = sma_fast.to_numpy(dtype=float, na_value=np.nan)
    s = sma_slow.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(f) & ~np.isnan(s)
    bull = valid & (c < f) & (f > s)
    bear = valid & (c > f) & (f < s)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_ma_alignment(close: pd.Series, sma_fast: pd.Series, sma_slow: pd.Series) -> pd.Series:
    """Calculate MA Trend Alignment signal."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    f = sma_fast.to_numpy(dtype=float, na_value=np.nan)
    s = sma_slow.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(f) & ~np.isnan(s)
    bull = valid & (c > s) & (f > s)
    bear = valid & (c < s) & (f < s)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_vortex_cross_14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    if len(close) < 14:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)
    try:
        vortex = ta.trend.VortexIndicator(high=high, low=low, close=close, window=14, fillna=False)
        return _crossover_signal(vortex.vortex_indicator_pos(), vortex.vortex_indicator_neg())
    except Exception:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)


def _calc_ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> tuple[pd.Series, pd.Series]:
    sig_tk = pd.Series(SignalState.NONE, index=close.index, dtype=str)
    res_cloud = np.full(len(close), SignalState.NONE, dtype=object)
    if len(close) >= 9:
        try:
            ichimoku = ta.trend.IchimokuIndicator(
                high=high, low=low, window1=9, window2=26, window3=52, fillna=False
            )
            tenkan = ichimoku.ichimoku_conversion_line()
            kijun = ichimoku.ichimoku_base_line()
            span_a = ichimoku.ichimoku_a().to_numpy(dtype=float, na_value=np.nan)
            span_b = ichimoku.ichimoku_b().to_numpy(dtype=float, na_value=np.nan)
            close_arr = close.to_numpy(dtype=float, na_value=np.nan)

            sig_tk = _crossover_signal(tenkan, kijun)

            valid_cloud = ~np.isnan(close_arr) & ~np.isnan(span_a) & ~np.isnan(span_b)
            kumo_high = np.maximum(span_a, span_b)
            kumo_low = np.minimum(span_a, span_b)

            buy_cloud = valid_cloud & (close_arr > kumo_high)
            sell_cloud = valid_cloud & (close_arr < kumo_low)
            hold_cloud = valid_cloud & (close_arr >= kumo_low) & (close_arr <= kumo_high)

            conds_cloud = [buy_cloud, sell_cloud, hold_cloud]
            choices_cloud = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
            res_cloud = np.select(conds_cloud, choices_cloud, default=SignalState.NONE)
        except Exception:
            pass
    return sig_tk, pd.Series(res_cloud, index=close.index, dtype=str)


def _calc_ehlers_super_smoother(close: pd.Series, length: int = 10) -> pd.Series:
    """Calculate Ehlers 2-Pole Super Smoother Filter."""
    n = len(close)
    if n == 0:
        return pd.Series(dtype=float, index=close.index)

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    ss = np.full(n, np.nan, dtype=float)

    a1 = np.exp(-np.sqrt(2.0) * np.pi / length)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / length)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

    for i in range(n):
        if np.isnan(c_arr[i]):
            continue
        if i == 0 or np.isnan(ss[i - 1]):
            ss[i] = c_arr[i]
        elif i == 1 or np.isnan(ss[i - 2]):
            ss[i] = c1 * c_arr[i] + c2 * ss[i - 1]
        else:
            ss[i] = c1 * c_arr[i] + c2 * ss[i - 1] + c3 * ss[i - 2]

    return pd.Series(ss, index=close.index)


def _calc_mcginley_dynamic(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate McGinley Dynamic moving average."""
    n = len(close)
    if n == 0:
        return pd.Series(dtype=float, index=close.index)

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    md = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if np.isnan(c_arr[i]):
            continue
        if i == 0 or np.isnan(md[i - 1]) or md[i - 1] <= 0:
            md[i] = c_arr[i]
        else:
            prev_md = md[i - 1]
            ratio = c_arr[i] / prev_md
            denom = length * (ratio**4)
            if denom == 0 or np.isnan(denom):
                md[i] = prev_md
            else:
                md[i] = prev_md + (c_arr[i] - prev_md) / denom

    return pd.Series(md, index=close.index)


def _calc_gmma_ribbon_expansion(close: pd.Series) -> pd.Series:
    """Calculate Guppy GMMA Ribbon Expansion signal."""
    fast_periods = [3, 5, 8, 10, 12, 15]
    slow_periods = [30, 35, 40, 45, 50, 60]

    fast_emas = pd.DataFrame(
        {f"f_{p}": close.ewm(span=p, adjust=False).mean() for p in fast_periods}
    )
    slow_emas = pd.DataFrame(
        {f"s_{p}": close.ewm(span=p, adjust=False).mean() for p in slow_periods}
    )

    fast_min = fast_emas.min(axis=1)
    fast_max = fast_emas.max(axis=1)
    slow_min = slow_emas.min(axis=1)
    slow_max = slow_emas.max(axis=1)

    bull_spread = fast_min - slow_max
    bear_spread = slow_min - fast_max

    prev_bull_spread = bull_spread.shift(1)
    prev_bear_spread = bear_spread.shift(1)

    bull_expand = (fast_min > slow_max) & (bull_spread > prev_bull_spread)
    bear_expand = (fast_max < slow_min) & (bear_spread > prev_bear_spread)
    bull_hold = fast_min > slow_max

    conds = [bull_expand, bear_expand, bull_hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(conds, choices, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_gmma_compression_breakout(close: pd.Series) -> pd.Series:
    """Calculate GMMA Compression Breakout signal."""
    periods = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
    emas = pd.DataFrame({f"e_{p}": close.ewm(span=p, adjust=False).mean() for p in periods})

    ribbon_max = emas.max(axis=1)
    ribbon_min = emas.min(axis=1)
    spread_pct = (ribbon_max - ribbon_min) / ribbon_min.replace(0, np.nan)

    compressed = spread_pct.rolling(3, min_periods=1).min() <= 0.015

    c = close.to_numpy(dtype=float, na_value=np.nan)
    r_max = ribbon_max.to_numpy(dtype=float, na_value=np.nan)
    r_min = ribbon_min.to_numpy(dtype=float, na_value=np.nan)
    comp = compressed.to_numpy(dtype=bool)

    valid = ~np.isnan(c) & ~np.isnan(r_max) & ~np.isnan(r_min)
    bull = valid & comp & (c > r_max)
    bear = valid & comp & (c < r_min)
    hold = valid & (c > r_max)

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(conds, choices, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_rainbow_ema_alignment(close: pd.Series) -> pd.Series:
    """Calculate Rainbow 5-EMA Alignment signal."""
    ema8 = close.ewm(span=8, adjust=False).mean().to_numpy(dtype=float, na_value=np.nan)
    ema13 = close.ewm(span=13, adjust=False).mean().to_numpy(dtype=float, na_value=np.nan)
    ema21 = close.ewm(span=21, adjust=False).mean().to_numpy(dtype=float, na_value=np.nan)
    ema34 = close.ewm(span=34, adjust=False).mean().to_numpy(dtype=float, na_value=np.nan)
    ema55 = close.ewm(span=55, adjust=False).mean().to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(ema8) & ~np.isnan(ema13) & ~np.isnan(ema21) & ~np.isnan(ema34) & ~np.isnan(ema55)
    )
    bull = valid & (ema8 > ema13) & (ema13 > ema21) & (ema21 > ema34) & (ema34 > ema55)
    bear = valid & (ema8 < ema13) & (ema13 < ema21) & (ema21 < ema34) & (ema34 < ema55)

    conds = [bull, bear]
    choices = [SignalState.BUY, SignalState.SELL]
    res = np.select(conds, choices, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_ehlers_instantaneous_trend(
    high: pd.Series, low: pd.Series, close: pd.Series, alpha: float = 0.07
) -> pd.Series:
    """Calculate Ehlers Instantaneous Trendline."""
    n = len(close)
    if n == 0:
        return pd.Series(dtype=float, index=close.index)

    price = ((high + low) / 2.0).to_numpy(dtype=float, na_value=np.nan)
    itrend = np.full(n, np.nan, dtype=float)

    a = alpha
    a2 = a * a
    c1 = a - a2 / 4.0
    c2 = 0.5 * a2
    c3 = -(a - 0.75 * a2)
    d1 = 2.0 * (1.0 - a)
    d2 = -((1.0 - a) ** 2)

    for i in range(n):
        if np.isnan(price[i]):
            continue
        if i < 2:
            itrend[i] = price[i]
        elif i < 7:
            itrend[i] = (price[i] + 2.0 * price[i - 1] + price[i - 2]) / 4.0
        else:
            itrend[i] = (
                c1 * price[i]
                + c2 * price[i - 1]
                + c3 * price[i - 2]
                + d1 * itrend[i - 1]
                + d2 * itrend[i - 2]
            )

    return pd.Series(itrend, index=close.index)


def _calc_coral_trend_filter(close: pd.Series, length: int = 21, cd: float = 0.4) -> pd.Series:
    """Calculate Coral Trend Filter slope signal."""
    di = (length - 1) // 2 + 1
    c1 = -(cd**3)
    c2 = 3.0 * (cd**2) + 3.0 * (cd**3)
    c3 = -6.0 * (cd**2) - 3.0 * cd - 3.0 * (cd**3)
    c4 = 1.0 + 3.0 * cd + (cd**3) + 3.0 * (cd**2)

    i1 = close.ewm(span=di, adjust=False).mean()
    i2 = i1.ewm(span=di, adjust=False).mean()
    i3 = i2.ewm(span=di, adjust=False).mean()
    i4 = i3.ewm(span=di, adjust=False).mean()
    i5 = i4.ewm(span=di, adjust=False).mean()
    i6 = i5.ewm(span=di, adjust=False).mean()

    coral = c1 * i6 + c2 * i5 + c3 * i4 + c4 * i3
    coral_arr = coral.to_numpy(dtype=float, na_value=np.nan)
    prev_coral = coral.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(coral_arr) & ~np.isnan(prev_coral)
    bull = valid & (coral_arr > prev_coral)
    bear = valid & (coral_arr < prev_coral)

    conds = [bull, bear]
    choices = [SignalState.BUY, SignalState.SELL]
    res = np.select(conds, choices, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_alma(
    close: pd.Series, length: int = 9, sigma: float = 6.0, offset: float = 0.85
) -> pd.Series:
    """Calculate Arnaud Legoux Moving Average (ALMA) with fallback."""
    try:
        res = pta.alma(close, length=length, sigma=sigma, offset=offset)
        if res is not None and isinstance(res, pd.Series) and not res.empty:
            return res
    except Exception:
        pass
    if len(close) < length or length <= 0:
        return pd.Series(np.nan, index=close.index)
    m = offset * (length - 1)
    s = length / sigma
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2.0 * s * s))
    weights /= weights.sum()
    return close.rolling(length).apply(lambda x: np.dot(x, weights), raw=True)


def _calc_zero_lag_ema(close: pd.Series, length: int = 21) -> pd.Series:
    """Calculate Zero-Lag Exponential Moving Average (ZLEMA)."""
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return 2.0 * ema1 - ema2


def _calc_vn30_prior_auction_bias(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    session_ctx: SessionContext,
    adx_window: int = 14,
) -> pd.Series:
    """Prior-Session 13:55 Afternoon Auction Bias (TRD053 / trend_vn30_prior_auction_bias_signal).

    Evaluates late-session momentum (13:55), body rate, and ADX at the prior session's
    afternoon auction setup, shifting the directional bias by 1 session to prevent lookahead.

    Triggers:
    - buy: prev_day_bias == 'long' and close > ema_55 and rsi_21 > 50
    - sell: prev_day_bias == 'short' and close < ema_55 and rsi_21 < 50
    - hold: (bias == 'long' and close >= ema_55) | (bias == 'short' and close <= ema_55)
    - none: default
    """
    n = len(close)
    if n == 0:
        return pd.Series(dtype=str, index=close.index)
    if n < 14:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    sess_id = session_ctx.session_id
    time_minutes = session_ctx.time_minutes

    ema_55 = close.ewm(span=55, adjust=False).mean()
    try:
        rsi_21 = ta.momentum.RSIIndicator(close=close, window=21, fillna=False).rsi()
    except Exception:
        rsi_21 = pd.Series(50.0, index=close.index)

    try:
        adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=adx_window)
        adx = adx_ind.adx()
    except Exception:
        adx = pd.Series(np.nan, index=close.index)

    last_close_per_sess = close.groupby(sess_id).last()
    pdc_per_sess = last_close_per_sess.shift(1)

    unique_sessions = sess_id.unique()
    session_bias = pd.Series("", index=unique_sessions, dtype=str)

    for sid in unique_sessions:
        pdc_val = pdc_per_sess.get(sid, np.nan)
        if pdc_val is None or np.isnan(pdc_val) or pdc_val <= 0:
            continue

        mask = sess_id == sid
        s_times = time_minutes[mask]
        s_high = high[mask]
        s_low = low[mask]
        s_close = close[mask]
        s_adx = adx[mask]

        if len(s_close) == 0:
            continue

        # 13:55 is 835 minutes. Pick last bar <= 835, fallback to last bar of session
        cand = s_times[s_times <= 835]
        eval_idx = cand.index[-1] if len(cand) > 0 else s_close.index[-1]

        close_1355 = s_close.loc[eval_idx]
        if isinstance(close_1355, pd.Series):
            close_1355 = close_1355.iloc[-1]
        mom_y = 100.0 * (float(close_1355) - float(pdc_val)) / (float(pdc_val) + 1e-8)

        # 09:15 is 555 minutes. Pick first bar == 555, fallback to first bar of session
        at_0915 = s_times[s_times == 555]
        if len(at_0915) > 0:
            fc = s_close.loc[at_0915.index[0]]
            first_close = float(fc.iloc[0]) if isinstance(fc, pd.Series) else float(fc)
        else:
            first_close = float(s_close.iloc[0])

        pre_1345 = s_times < 825
        if pre_1345.any():
            high_pre_1345 = float(s_high[pre_1345].max())
        else:
            eval_pos = s_close.index.get_loc(eval_idx)
            if isinstance(eval_pos, (slice, np.ndarray)):
                bars_pre = s_high.iloc[: len(s_high)]
            else:
                bars_pre = s_high.iloc[: eval_pos + 1]
            high_pre_1345 = float(bars_pre.max()) if len(bars_pre) > 0 else float(s_high.iloc[0])

        pre_1355 = s_times < 835
        if pre_1355.any():
            low_pre_1355 = float(s_low[pre_1355].min())
        else:
            eval_pos = s_close.index.get_loc(eval_idx)
            if isinstance(eval_pos, (slice, np.ndarray)):
                bars_pre = s_low.iloc[: len(s_low)]
            else:
                bars_pre = s_low.iloc[: eval_pos + 1]
            low_pre_1355 = float(bars_pre.min()) if len(bars_pre) > 0 else float(s_low.iloc[0])

        body_rate = (float(close_1355) - first_close) / (high_pre_1345 - low_pre_1355 + 1e-8)

        adx_val_item = s_adx.loc[eval_idx]
        adx_val = (
            float(adx_val_item.iloc[-1])
            if isinstance(adx_val_item, pd.Series)
            else float(adx_val_item)
        )

        if not np.isnan(adx_val):
            if mom_y > 0.26 and body_rate > 0.65 and adx_val < 26.5:
                session_bias[sid] = "long"
            elif mom_y < -0.18 and body_rate < -0.39 and adx_val < 26.5:
                session_bias[sid] = "short"

    session_bias_shifted = session_bias.shift(1).fillna("")
    bar_bias = sess_id.map(session_bias_shifted).fillna("")

    bias_long = bar_bias == "long"
    bias_short = bar_bias == "short"

    c_above_ema = close > ema_55
    c_below_ema = close < ema_55
    c_ge_ema = close >= ema_55
    c_le_ema = close <= ema_55

    rsi_bull = rsi_21 > 50.0
    rsi_bear = rsi_21 < 50.0

    buy = bias_long & c_above_ema & rsi_bull
    sell = bias_short & c_below_ema & rsi_bear
    hold = (bias_long & c_ge_ema) | (bias_short & c_le_ema)

    condlist = [buy, sell, hold]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)

    return pd.Series(res, index=close.index, dtype=str).fillna(SignalState.NONE)


def _calc_vn30_asymmetric_persistence(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    session_ctx: SessionContext,
) -> pd.Series:
    """VN30 Asymmetric Range Position & Selling Persistence (TRD054 / trend_vn30_asymmetric_persistence_signal).

    Leverages structural market asymmetry:
    - Longs: Session high range position continuation (pos_range > 0.79, DI+ > DI-, slope_8 > 0, RSI_5 > 60).
    - Shorts: Sustained selling persistence (persist_short >= 0.42, DI- > DI+, slope_8 < 0, RSI_5 < 40).
    - Hold: Directional trend alignment.
    - None: Default.
    """
    n = len(close)
    if n == 0:
        return pd.Series(dtype=str, index=close.index)
    if n < 14:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    sess_id = session_ctx.session_id
    running_high = high.groupby(sess_id).cummax()
    running_low = low.groupby(sess_id).cummin()
    session_open = open_p.groupby(sess_id).transform("first")

    pos_range = (close - running_low) / (running_high - running_low + 1e-8)
    persist_short = (close < session_open).astype(float).rolling(12, min_periods=1).mean()

    try:
        adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        di_plus = adx_ind.adx_pos()
        di_minus = adx_ind.adx_neg()
    except Exception:
        di_plus = pd.Series(np.nan, index=close.index)
        di_minus = pd.Series(np.nan, index=close.index)

    slope_8 = close.diff(8) / 8.0

    try:
        rsi_5 = ta.momentum.RSIIndicator(close=close, window=5, fillna=False).rsi()
    except Exception:
        rsi_5 = pd.Series(50.0, index=close.index)

    buy = (pos_range > 0.79) & (di_plus > di_minus) & (slope_8 > 0.0) & (rsi_5 > 60.0)
    sell = (persist_short >= 0.42) & (di_minus > di_plus) & (slope_8 < 0.0) & (rsi_5 < 40.0)
    hold = ((di_plus > di_minus) & (slope_8 > 0.0)) | ((di_minus > di_plus) & (slope_8 < 0.0))

    condlist = [buy, sell, hold]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)

    return pd.Series(res, index=close.index, dtype=str).fillna(SignalState.NONE)


def generate_trend_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Generate all 54 trend and moving average signals from OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
    show_progress : bool, default False
        Whether to display a real-time progress bar for this signal group.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 54 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)

    with GroupProgressBar(
        "Trend Signals", total=len(TREND_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if len(df_norm) == 0:
            pbar.update(len(TREND_SIGNAL_COLUMNS))
            return pd.DataFrame(
                {col: pd.Series(dtype=str, index=df.index) for col in TREND_SIGNAL_COLUMNS},
                index=df.index,
            )

        close = df_norm["close"]
        open_p = df_norm["open"]
        high = df_norm["high"]
        low = df_norm["low"]
        volume = df_norm["volume"]
        session_ctx = extract_session_context(df_norm)

        signals = pd.DataFrame(index=df_norm.index)

        # 1. Moving Averages (11)
        sma5 = close.rolling(5, min_periods=1).mean()
        sma10 = close.rolling(10, min_periods=1).mean()
        sma20 = close.rolling(20, min_periods=1).mean()
        sma50 = close.rolling(50, min_periods=1).mean()
        sma200 = close.rolling(200, min_periods=1).mean()

        signals["trend_sma_cross_5_20_signal"] = _crossover_signal(sma5, sma20)
        signals["trend_sma_cross_10_50_signal"] = _crossover_signal(sma10, sma50)
        signals["trend_sma_cross_20_50_signal"] = _crossover_signal(sma20, sma50)
        signals["trend_golden_cross_50_200_signal"] = _crossover_signal(sma50, sma200)

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        signals["trend_ema_cross_9_21_signal"] = _crossover_signal(ema9, ema21)
        signals["trend_ema_cross_12_26_signal"] = _crossover_signal(ema12, ema26)
        signals["trend_ema_cross_50_200_signal"] = _crossover_signal(ema50, ema200)

        dema10 = _calc_dema(close, 10)
        dema30 = _calc_dema(close, 30)
        signals["trend_dema_cross_10_30_signal"] = _crossover_signal(dema10, dema30)

        tema10 = _calc_tema(close, 10)
        tema30 = _calc_tema(close, 30)
        signals["trend_tema_cross_10_30_signal"] = _crossover_signal(tema10, tema30)

        hma9 = _calc_hma(close, 9)
        hma21 = _calc_hma(close, 21)
        signals["trend_hma_cross_9_21_signal"] = _crossover_signal(hma9, hma21)

        vwma10 = _calc_vwma(close, volume, 10)
        vwma30 = _calc_vwma(close, volume, 30)
        signals["trend_vwma_cross_10_30_signal"] = _crossover_signal(vwma10, vwma30)
        pbar.update(11)

        # 2. Price Above MA Baselines (3)
        signals["trend_price_above_sma20_signal"] = _price_above_ma(close, sma20)
        signals["trend_price_above_ema50_signal"] = _price_above_ma(close, ema50)
        signals["trend_price_above_ema200_signal"] = _price_above_ma(close, ema200)
        pbar.update(3)

        # 3. MACD Variants (5)
        macd_line, macd_signal, macd_hist = _calc_macd(close, fast=12, slow=26, signal=9)
        signals["trend_macd_cross_signal"] = _crossover_signal(macd_line, macd_signal)
        signals["trend_macd_zero_cross_signal"] = _crossover_signal(
            macd_line, pd.Series(0.0, index=close.index)
        )
        signals["trend_macd_hist_reversal_signal"] = _calc_macd_hist_reversal(macd_hist)

        fast_line, fast_sig, _ = _calc_macd(close, fast=6, slow=13, signal=4)
        signals["trend_macd_fast_cross_signal"] = _crossover_signal(fast_line, fast_sig)

        slow_line, slow_sig, _ = _calc_macd(close, fast=24, slow=52, signal=18)
        signals["trend_macd_slow_cross_signal"] = _crossover_signal(slow_line, slow_sig)
        pbar.update(5)

        # 4. SuperTrend Variants (3)
        signals["trend_supertrend_10_3_signal"] = _calc_supertrend(
            high, low, close, length=10, multiplier=3.0
        )
        signals["trend_supertrend_7_2_signal"] = _calc_supertrend(
            high, low, close, length=7, multiplier=2.0
        )
        signals["trend_supertrend_14_4_signal"] = _calc_supertrend(
            high, low, close, length=14, multiplier=4.0
        )
        pbar.update(3)

        # 5. Parabolic SAR (1)
        signals["trend_psar_reversal_signal"] = _calc_psar(high, low, close)
        pbar.update(1)

        # 6. Aroon Oscillator & Crosses (2)
        signals["trend_aroon_cross_14_signal"] = _calc_aroon_cross(high, low, window=14)
        signals["trend_aroon_cross_25_signal"] = _calc_aroon_cross(high, low, window=25)
        pbar.update(2)

        # 7. ADX / DMI Directional Trends (2)
        signals["trend_adx_dmi_14_signal"] = _calc_adx_dmi(high, low, close, window=14)
        signals["trend_adx_dmi_28_signal"] = _calc_adx_dmi(high, low, close, window=28)
        pbar.update(2)

        # 8. Ichimoku Cloud (2)
        sig_tk, sig_cloud = _calc_ichimoku(high, low, close)
        signals["trend_ichimoku_tk_cross_signal"] = sig_tk
        signals["trend_ichimoku_cloud_breakout_signal"] = sig_cloud
        pbar.update(2)

        # 9. Vortex Indicator Cross (14) (1)
        signals["trend_vortex_cross_14_signal"] = _calc_vortex_cross_14(high, low, close)
        pbar.update(1)

        # 10. TRIX Crossover (15) (1)
        signals["trend_trix_cross_15_signal"] = _calc_trix(close, length=15, signal_length=9)
        pbar.update(1)

        # 11. KAMA Trend Reversal (10) (1)
        signals["trend_kama_reversal_10_signal"] = _calc_kama_reversal(close, length=10)
        pbar.update(1)

        # 12. TMA Crossover (10) (1)
        signals["trend_tma_cross_10_signal"] = _calc_tma_cross(close, length=10)
        pbar.update(1)

        # 13. MA Alignment (20, 50) (1)
        signals["trend_ma_alignment_20_50_signal"] = _calc_ma_alignment(close, sma20, sma50)
        pbar.update(1)

        # 15. Pullback to SMA20 (20, 50) (1)
        signals["trend_pullback_sma20_50_signal"] = _calc_pullback_sma(close, sma20, sma50)
        pbar.update(1)

        # 16. Micro Trend (3 bars) (1)
        signals["trend_micro_trend_3_signal"] = _calc_micro_trend(close)
        pbar.update(1)

        # 17. Micro Reversal (1)
        signals["trend_micro_reversal_signal"] = _calc_micro_reversal(close)
        pbar.update(1)

        # 18. Price Cross SMA20 (1)
        signals["trend_price_cross_sma20_signal"] = _crossover_signal(close, sma20)
        pbar.update(1)

        # 19. Previous Bar High/Low Break (1)
        signals["trend_prev_bar_break_signal"] = _calc_prev_bar_break(high, low, close)
        pbar.update(1)

        # 20. Trend Intensity Index (TII 14) (1)
        signals["trend_tii_14_signal"] = _calc_tii(close, length=14, sma_length=20)
        pbar.update(1)

        # 21. Ehlers 2-Pole Super Smoother Filter (1)
        ss10 = _calc_ehlers_super_smoother(close, length=10)
        signals["trend_ehlers_super_smoother_cross_signal"] = _crossover_signal(close, ss10)
        pbar.update(1)

        # 22. McGinley Dynamic (14) (1)
        md14 = _calc_mcginley_dynamic(close, length=14)
        signals["trend_mcginley_dynamic_cross_signal"] = _crossover_signal(close, md14)
        pbar.update(1)

        # 23. GMMA Ribbon Expansion (1)
        signals["trend_gmma_ribbon_expansion_signal"] = _calc_gmma_ribbon_expansion(close)
        pbar.update(1)

        # 24. GMMA Compression Breakout (1)
        signals["trend_gmma_compression_breakout_signal"] = _calc_gmma_compression_breakout(close)
        pbar.update(1)

        # 25. Rainbow 5-EMA Alignment (1)
        signals["trend_rainbow_ema_alignment_signal"] = _calc_rainbow_ema_alignment(close)
        pbar.update(1)

        # 26. Ehlers Instantaneous Trendline (1)
        itrend = _calc_ehlers_instantaneous_trend(high, low, close, alpha=0.07)
        signals["trend_ehlers_instantaneous_trend_signal"] = _crossover_signal(close, itrend)
        pbar.update(1)

        # 27. Coral Trend Filter (1)
        signals["trend_coral_trend_filter_signal"] = _calc_coral_trend_filter(
            close, length=21, cd=0.4
        )
        pbar.update(1)

        # 28. Conservative Slow SuperTrend (20, 5.0) (1)
        signals["trend_supertrend_atr_20_5_signal"] = _calc_supertrend(
            high, low, close, length=20, multiplier=5.0
        )
        pbar.update(1)

        # 29. Donchian Channel 20 Middle Line (1)
        donchian_high = high.rolling(20, min_periods=1).max()
        donchian_low = low.rolling(20, min_periods=1).min()
        donchian_mid = (donchian_high + donchian_low) / 2.0
        signals["trend_donchian_middle_cross_20_signal"] = _crossover_signal(close, donchian_mid)
        pbar.update(1)

        # 30. Bill Williams Alligator Lips / Jaw Cross (1)
        med_price = (high + low) / 2.0
        alligator_lips = med_price.ewm(span=5, adjust=False).mean().shift(3)
        alligator_jaw = med_price.ewm(span=13, adjust=False).mean().shift(8)
        signals["trend_alligator_lips_jaw_cross_signal"] = _crossover_signal(
            alligator_lips, alligator_jaw
        )
        pbar.update(1)

        # 31. Arnaud Legoux Moving Average (ALMA 9) (1)
        alma9 = _calc_alma(close, length=9, sigma=6.0, offset=0.85)
        signals["trend_alma_cross_9_signal"] = _crossover_signal(close, alma9)
        pbar.update(1)

        # 32. Zero-Lag EMA (21) (1)
        zlema21 = _calc_zero_lag_ema(close, length=21)
        signals["trend_zero_lag_ema_cross_21_signal"] = _crossover_signal(close, zlema21)
        pbar.update(1)

        # 33. VN30 Prior Auction Bias (1)
        signals["trend_vn30_prior_auction_bias_signal"] = _calc_vn30_prior_auction_bias(
            open_p, high, low, close, session_ctx
        )
        pbar.update(1)

        # 34. VN30 Asymmetric Range Position & Selling Persistence (1)
        signals["trend_vn30_asymmetric_persistence_signal"] = _calc_vn30_asymmetric_persistence(
            open_p, high, low, close, session_ctx
        )
        pbar.update(1)

        # Ensure all columns are present, filled with NONE, and matching index
        for col in TREND_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[TREND_SIGNAL_COLUMNS]
