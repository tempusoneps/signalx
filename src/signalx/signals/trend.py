from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as pta
import ta

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
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


def generate_trend_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Generate all 30 standardized trend signals from normalized OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
    show_progress : bool, default False
        Whether to display a real-time progress bar for this signal group.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 30 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)
    signals = pd.DataFrame(index=df_norm.index)

    with GroupProgressBar(
        "Trend Signals", total=len(TREND_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if df_norm.empty:
            for col in TREND_SIGNAL_COLUMNS:
                signals[col] = pd.Series(dtype=str)
            pbar.update(len(TREND_SIGNAL_COLUMNS))
            return signals

        close = df_norm["close"]
        high = df_norm["high"]
        low = df_norm["low"]
        volume = df_norm["volume"]

        # 1. Simple Moving Average (SMA) Crosses (4)
        sma5 = ta.trend.sma_indicator(close, window=5, fillna=False)
        sma10 = ta.trend.sma_indicator(close, window=10, fillna=False)
        sma20 = ta.trend.sma_indicator(close, window=20, fillna=False)
        sma50 = ta.trend.sma_indicator(close, window=50, fillna=False)
        sma200 = ta.trend.sma_indicator(close, window=200, fillna=False)

        signals["trend_sma_cross_5_20_signal"] = _crossover_signal(sma5, sma20)
        signals["trend_sma_cross_10_50_signal"] = _crossover_signal(sma10, sma50)
        signals["trend_sma_cross_20_50_signal"] = _crossover_signal(sma20, sma50)
        signals["trend_golden_cross_50_200_signal"] = _crossover_signal(sma50, sma200)
        pbar.update(4)

        # 2. Exponential Moving Average (EMA) Crosses (3)
        ema9 = ta.trend.ema_indicator(close, window=9, fillna=False)
        ema12 = ta.trend.ema_indicator(close, window=12, fillna=False)
        ema21 = ta.trend.ema_indicator(close, window=21, fillna=False)
        ema26 = ta.trend.ema_indicator(close, window=26, fillna=False)
        ema50 = ta.trend.ema_indicator(close, window=50, fillna=False)
        ema200 = ta.trend.ema_indicator(close, window=200, fillna=False)

        signals["trend_ema_cross_9_21_signal"] = _crossover_signal(ema9, ema21)
        signals["trend_ema_cross_12_26_signal"] = _crossover_signal(ema12, ema26)
        signals["trend_ema_cross_50_200_signal"] = _crossover_signal(ema50, ema200)
        pbar.update(3)

        # 3. Advanced Moving Average Crosses (DEMA, TEMA, HMA, VWMA) (4)
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
        pbar.update(4)

        # 4. Price vs Moving Average Positions (3)
        signals["trend_price_above_sma20_signal"] = _price_above_ma(close, sma20)
        signals["trend_price_above_ema50_signal"] = _price_above_ma(close, ema50)
        signals["trend_price_above_ema200_signal"] = _price_above_ma(close, ema200)
        pbar.update(3)

    # 5. MACD Variants (Standard, Zero Cross, Hist Reversal, Fast, Slow)
    macd_std = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9, fillna=False)
    macd_line = macd_std.macd()
    macd_signal = macd_std.macd_signal()
    macd_diff = macd_std.macd_diff()

    signals["trend_macd_cross_signal"] = _crossover_signal(macd_line, macd_signal)
    signals["trend_macd_zero_cross_signal"] = _crossover_signal(
        macd_line, pd.Series(0.0, index=df_norm.index)
    )

    macd_diff_arr = macd_diff.to_numpy(dtype=float, na_value=np.nan)
    prev_diff_arr = macd_diff.shift(1).to_numpy(dtype=float, na_value=np.nan)
    valid_hist = ~np.isnan(macd_diff_arr) & ~np.isnan(prev_diff_arr)
    turn_pos = valid_hist & (macd_diff_arr > 0) & (prev_diff_arr <= 0)
    turn_neg = valid_hist & (macd_diff_arr < 0) & (prev_diff_arr >= 0)
    holding_pos = valid_hist & (macd_diff_arr > 0)

    conds_hist = [turn_pos, turn_neg, holding_pos]
    choices_hist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res_hist = np.select(conds_hist, choices_hist, default=SignalState.NONE)
    signals["trend_macd_hist_reversal_signal"] = pd.Series(res_hist, index=df_norm.index, dtype=str)

    macd_fast = ta.trend.MACD(close, window_fast=6, window_slow=13, window_sign=5, fillna=False)
    signals["trend_macd_fast_cross_signal"] = _crossover_signal(
        macd_fast.macd(), macd_fast.macd_signal()
    )

    macd_slow = ta.trend.MACD(close, window_fast=19, window_slow=39, window_sign=9, fillna=False)
    signals["trend_macd_slow_cross_signal"] = _crossover_signal(
        macd_slow.macd(), macd_slow.macd_signal()
    )
    pbar.update(5)

    # 6. SuperTrend Regimes (10/3, 7/2, 14/4) (3)
    for length, mult in [(10, 3.0), (7, 2.0), (14, 4.0)]:
        col_name = f"trend_supertrend_{length}_{int(mult)}_signal"
        res_st = np.full(len(df_norm), SignalState.NONE, dtype=object)
        try:
            st = pta.supertrend(high=high, low=low, close=close, length=length, multiplier=mult)
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
        signals[col_name] = pd.Series(res_st, index=df_norm.index, dtype=str)
    pbar.update(3)

    # 7. Parabolic SAR Reversal (1)
    res_psar = np.full(len(df_norm), SignalState.NONE, dtype=object)
    if len(df_norm) >= 2:
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
    signals["trend_psar_reversal_signal"] = pd.Series(res_psar, index=df_norm.index, dtype=str)
    pbar.update(1)

    # 8. Aroon Cross (14, 25) (2)
    for period in [14, 25]:
        col_name = f"trend_aroon_cross_{period}_signal"
        sig_aroon = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
        if len(df_norm) >= period:
            try:
                aroon = ta.trend.AroonIndicator(high=high, low=low, window=period, fillna=False)
                sig_aroon = _crossover_signal(aroon.aroon_up(), aroon.aroon_down())
            except Exception:
                pass
        signals[col_name] = sig_aroon
    pbar.update(2)

    # 9. ADX / DMI Directional Strength (14, 28) (2)
    for period in [14, 28]:
        col_name = f"trend_adx_dmi_{period}_signal"
        res_adx = np.full(len(df_norm), SignalState.NONE, dtype=object)
        if len(df_norm) >= period:
            try:
                adx_ind = ta.trend.ADXIndicator(
                    high=high, low=low, close=close, window=period, fillna=False
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
        signals[col_name] = pd.Series(res_adx, index=df_norm.index, dtype=str)
    pbar.update(2)

    # 10. Ichimoku Kinko Hyo (TK Cross and Cloud Breakout) (2)
    sig_tk = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
    res_cloud = np.full(len(df_norm), SignalState.NONE, dtype=object)
    if len(df_norm) >= 9:
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
    signals["trend_ichimoku_tk_cross_signal"] = sig_tk
    signals["trend_ichimoku_cloud_breakout_signal"] = pd.Series(
        res_cloud, index=df_norm.index, dtype=str
    )
    pbar.update(2)

    # 11. Vortex Indicator Cross (14) (1)
    sig_vortex = pd.Series(SignalState.NONE, index=df_norm.index, dtype=str)
    if len(df_norm) >= 14:
        try:
            vortex = ta.trend.VortexIndicator(
                high=high, low=low, close=close, window=14, fillna=False
            )
            sig_vortex = _crossover_signal(
                vortex.vortex_indicator_pos(), vortex.vortex_indicator_neg()
            )
        except Exception:
            pass
    signals["trend_vortex_cross_14_signal"] = sig_vortex
    pbar.update(1)

    # Ensure all columns are present, filled with NONE, and matching index
    for col in TREND_SIGNAL_COLUMNS:
        if col not in signals.columns:
            signals[col] = SignalState.NONE
        else:
            signals[col] = signals[col].fillna(SignalState.NONE)

    return signals[TREND_SIGNAL_COLUMNS]
