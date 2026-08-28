from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as pta
import ta

from signalx.constants import SignalState
from signalx.utils import normalize_ohlcv

MOMENTUM_SIGNAL_COLUMNS = [
    "mom_rsi_ob_os_14_signal",
    "mom_rsi_ob_os_7_signal",
    "mom_rsi_ob_os_21_signal",
    "mom_rsi_ob_os_28_signal",
    "mom_rsi_50_cross_14_signal",
    "mom_rsi_50_cross_21_signal",
    "mom_stoch_kd_cross_14_3_3_signal",
    "mom_stoch_kd_cross_5_3_3_signal",
    "mom_stoch_rsi_cross_14_signal",
    "mom_williams_r_14_signal",
    "mom_williams_r_28_signal",
    "mom_cci_100_14_signal",
    "mom_cci_200_20_signal",
    "mom_roc_zero_cross_5_signal",
    "mom_roc_zero_cross_10_signal",
    "mom_roc_zero_cross_20_signal",
    "mom_mfi_ob_os_14_signal",
    "mom_tsi_cross_13_25_signal",
    "mom_fisher_cross_9_signal",
    "mom_ao_zero_cross_signal",
    "mom_ao_saucer_signal",
    "mom_ultimate_osc_signal",
    "mom_cmo_14_signal",
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
    """Helper to generate buy/sell/hold signals based on upper and lower bounds.

    Parameters
    ----------
    val : pd.Series
        Indicator values.
    lower : float
        Lower threshold.
    upper : float
        Upper threshold.
    buy_below : bool, default True
        If True (e.g. RSI, MFI, Williams %R), buying occurs when below lower (oversold)
        and selling occurs when above upper (overbought).
        If False (e.g. CCI breakout, CMO), buying occurs when above upper (bullish breakout)
        and selling occurs when below lower (bearish breakdown).
    """
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


def _calc_fisher(high: pd.Series, low: pd.Series, length: int = 9) -> tuple[pd.Series, pd.Series]:
    """Calculate Fisher Transform line and signal line with graceful fallback."""
    if len(high) >= length:
        try:
            res = pta.fisher(high=high, low=low, length=length)
            if res is not None and isinstance(res, pd.DataFrame) and len(res.columns) >= 2:
                f_cols = [c for c in res.columns if not c.endswith("s") and "s_" not in c]
                s_cols = [c for c in res.columns if c.endswith("s") or "s_" in c]
                if f_cols and s_cols:
                    return res[f_cols[0]], res[s_cols[0]]
                return res.iloc[:, 0], res.iloc[:, 1]
        except Exception:
            pass

    # Native Ehlers Fisher Transform fallback
    n = len(high)
    if n < length or length <= 0:
        nan_s = pd.Series(np.nan, index=high.index)
        return nan_s, nan_s

    med = (high + low) / 2.0
    lowest = low.rolling(length).min()
    highest = high.rolling(length).max()
    denom = (highest - lowest).replace(0, np.nan)
    raw = 2.0 * ((med - lowest) / denom - 0.5)

    val = np.zeros(n)
    fish = np.zeros(n)
    raw_arr = raw.to_numpy(dtype=float, na_value=np.nan)

    for i in range(length - 1, n):
        r = raw_arr[i] if not np.isnan(raw_arr[i]) else 0.0
        v = 0.33 * 2.0 * r + 0.67 * val[i - 1]
        v = np.clip(v, -0.999, 0.999)
        val[i] = v
        f = 0.5 * np.log((1.0 + v) / (1.0 - v)) + 0.5 * fish[i - 1]
        fish[i] = f

    fish_s = pd.Series(fish, index=high.index)
    fish_s.iloc[: length - 1] = np.nan
    fish_sig = fish_s.shift(1)
    return fish_s, fish_sig


def _calc_cmo(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Chande Momentum Oscillator (CMO) with graceful fallback."""
    if len(close) >= length:
        try:
            res = pta.cmo(close, length=length)
            if res is not None and isinstance(res, pd.Series) and not res.empty:
                return res
        except Exception:
            pass

    diff = close.diff()
    pos = diff.clip(lower=0.0)
    neg = (-diff).clip(lower=0.0)
    pos_rma = pos.ewm(alpha=1.0 / length, adjust=False).mean()
    neg_rma = neg.ewm(alpha=1.0 / length, adjust=False).mean()
    denom = (pos_rma + neg_rma).replace(0, np.nan)
    return 100.0 * (pos_rma - neg_rma) / denom


def _calc_ao_saucer(ao: pd.Series) -> pd.Series:
    """Helper to detect Awesome Oscillator Saucer setup patterns."""
    ao_arr = ao.to_numpy(dtype=float, na_value=np.nan)
    prev1 = ao.shift(1).to_numpy(dtype=float, na_value=np.nan)
    prev2 = ao.shift(2).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(ao_arr) & ~np.isnan(prev1) & ~np.isnan(prev2)

    bull_saucer = (
        valid & (ao_arr > 0) & (prev1 > 0) & (prev2 > 0) & (prev1 < prev2) & (ao_arr > prev1)
    )
    bear_saucer = (
        valid & (ao_arr < 0) & (prev1 < 0) & (prev2 < 0) & (prev1 > prev2) & (ao_arr < prev1)
    )
    holding_pos = valid & (ao_arr > 0)

    condlist = [bull_saucer, bear_saucer, holding_pos]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res_arr = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res_arr, index=ao.index, dtype=str)


def generate_momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all 23 standardized momentum & oscillator signals from normalized OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 23 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)
    signals = pd.DataFrame(index=df_norm.index)

    if df_norm.empty:
        for col in MOMENTUM_SIGNAL_COLUMNS:
            signals[col] = pd.Series(dtype=str)
        return signals

    close = df_norm["close"]
    high = df_norm["high"]
    low = df_norm["low"]
    volume = df_norm["volume"]

    # 1. RSI Overbought/Oversold Signals (14, 7, 21, 28)
    rsi14 = ta.momentum.RSIIndicator(close, window=14, fillna=False).rsi()
    rsi7 = ta.momentum.RSIIndicator(close, window=7, fillna=False).rsi()
    rsi21 = ta.momentum.RSIIndicator(close, window=21, fillna=False).rsi()
    rsi28 = ta.momentum.RSIIndicator(close, window=28, fillna=False).rsi()

    signals["mom_rsi_ob_os_14_signal"] = _bound_signal(rsi14, 30.0, 70.0, buy_below=True)
    signals["mom_rsi_ob_os_7_signal"] = _bound_signal(rsi7, 20.0, 80.0, buy_below=True)
    signals["mom_rsi_ob_os_21_signal"] = _bound_signal(rsi21, 30.0, 70.0, buy_below=True)
    signals["mom_rsi_ob_os_28_signal"] = _bound_signal(rsi28, 30.0, 70.0, buy_below=True)

    # 2. RSI Centerline 50 Momentum Shifts (14, 21)
    centerline_50 = pd.Series(50.0, index=df_norm.index)
    signals["mom_rsi_50_cross_14_signal"] = _crossover_signal(rsi14, centerline_50)
    signals["mom_rsi_50_cross_21_signal"] = _crossover_signal(rsi21, centerline_50)

    # 3. Stochastic KD Crossovers (14,3,3 and 5,3,3)
    stoch14 = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3, fillna=False
    )
    signals["mom_stoch_kd_cross_14_3_3_signal"] = _crossover_signal(
        stoch14.stoch(), stoch14.stoch_signal()
    )

    stoch5 = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=5, smooth_window=3, fillna=False
    )
    signals["mom_stoch_kd_cross_5_3_3_signal"] = _crossover_signal(
        stoch5.stoch(), stoch5.stoch_signal()
    )

    # 4. StochRSI Crossover (14)
    stoch_rsi = ta.momentum.StochRSIIndicator(
        close=close, window=14, smooth1=3, smooth2=3, fillna=False
    )
    signals["mom_stoch_rsi_cross_14_signal"] = _crossover_signal(
        stoch_rsi.stochrsi_k(), stoch_rsi.stochrsi_d()
    )

    # 5. Williams %R Signals (14, 28)
    wr14 = ta.momentum.WilliamsRIndicator(
        high=high, low=low, close=close, lbp=14, fillna=False
    ).williams_r()
    wr28 = ta.momentum.WilliamsRIndicator(
        high=high, low=low, close=close, lbp=28, fillna=False
    ).williams_r()

    signals["mom_williams_r_14_signal"] = _bound_signal(wr14, -80.0, -20.0, buy_below=True)
    signals["mom_williams_r_28_signal"] = _bound_signal(wr28, -80.0, -20.0, buy_below=True)

    # 6. Commodity Channel Index (CCI) Signals (14 period +/-100, 20 period +/-200)
    cci14 = ta.trend.CCIIndicator(high=high, low=low, close=close, window=14, fillna=False).cci()
    cci20 = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20, fillna=False).cci()

    signals["mom_cci_100_14_signal"] = _bound_signal(cci14, -100.0, 100.0, buy_below=False)
    signals["mom_cci_200_20_signal"] = _bound_signal(cci20, -200.0, 200.0, buy_below=False)

    # 7. Rate of Change (ROC) Zero Centerline Crossovers (5, 10, 20)
    centerline_zero = pd.Series(0.0, index=df_norm.index)
    roc5 = ta.momentum.ROCIndicator(close=close, window=5, fillna=False).roc()
    roc10 = ta.momentum.ROCIndicator(close=close, window=10, fillna=False).roc()
    roc20 = ta.momentum.ROCIndicator(close=close, window=20, fillna=False).roc()

    signals["mom_roc_zero_cross_5_signal"] = _crossover_signal(roc5, centerline_zero)
    signals["mom_roc_zero_cross_10_signal"] = _crossover_signal(roc10, centerline_zero)
    signals["mom_roc_zero_cross_20_signal"] = _crossover_signal(roc20, centerline_zero)

    # 8. Money Flow Index (MFI) Overbought/Oversold (14)
    mfi14 = ta.volume.MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=14, fillna=False
    ).money_flow_index()
    signals["mom_mfi_ob_os_14_signal"] = _bound_signal(mfi14, 20.0, 80.0, buy_below=True)

    # 9. True Strength Index (TSI) Zero Line Crossover (13, 25)
    tsi = ta.momentum.TSIIndicator(close=close, window_slow=25, window_fast=13, fillna=False).tsi()
    signals["mom_tsi_cross_13_25_signal"] = _crossover_signal(tsi, centerline_zero)

    # 10. Fisher Transform Crossover (9)
    fisher_k, fisher_d = _calc_fisher(high, low, length=9)
    signals["mom_fisher_cross_9_signal"] = _crossover_signal(fisher_k, fisher_d)

    # 11. Awesome Oscillator (AO) Zero Cross and Saucer Pattern
    ao = ta.momentum.AwesomeOscillatorIndicator(
        high=high, low=low, window1=5, window2=34, fillna=False
    ).awesome_oscillator()
    signals["mom_ao_zero_cross_signal"] = _crossover_signal(ao, centerline_zero)
    signals["mom_ao_saucer_signal"] = _calc_ao_saucer(ao)

    # 12. Ultimate Oscillator Boundary Extremes (7, 14, 28)
    uo = ta.momentum.UltimateOscillator(
        high=high, low=low, close=close, window1=7, window2=14, window3=28, fillna=False
    ).ultimate_oscillator()
    signals["mom_ultimate_osc_signal"] = _bound_signal(uo, 30.0, 70.0, buy_below=True)

    # 13. Chande Momentum Oscillator (CMO) Thresholds (14)
    cmo14 = _calc_cmo(close, length=14)
    signals["mom_cmo_14_signal"] = _bound_signal(cmo14, -50.0, 50.0, buy_below=False)

    # Ensure all columns are present, filled with NONE, and matching index
    for col in MOMENTUM_SIGNAL_COLUMNS:
        if col not in signals.columns:
            signals[col] = SignalState.NONE
        else:
            signals[col] = signals[col].fillna(SignalState.NONE)

    return signals[MOMENTUM_SIGNAL_COLUMNS]
