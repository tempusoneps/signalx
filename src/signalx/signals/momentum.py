from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as pta
import ta

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
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
    "mom_min_max_10_rsi_signal",
    "mom_rsi_divergence_5_signal",
    "mom_connors_rsi_3_2_100_signal",
    "mom_mfi_reversal_20_80_signal",
    "mom_shift_3_bar_signal",
    "mom_return_momentum_5_signal",
    "mom_extreme_move_10_signal",
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


def _calc_streak_series(close: pd.Series) -> pd.Series:
    """Calculate price streak count (+1, +2, ... for up-days, -1, -2, ... for down-days)."""
    diff = close.diff().to_numpy()
    n = len(diff)
    streak = np.zeros(n, dtype=float)
    for i in range(1, n):
        if np.isnan(diff[i]):
            streak[i] = 0.0
        elif diff[i] > 0:
            streak[i] = streak[i - 1] + 1.0 if streak[i - 1] > 0 else 1.0
        elif diff[i] < 0:
            streak[i] = streak[i - 1] - 1.0 if streak[i - 1] < 0 else -1.0
        else:
            streak[i] = 0.0
    return pd.Series(streak, index=close.index, dtype=float)


def _calc_connors_rsi(
    close: pd.Series, n_rsi: int = 3, n_streak: int = 2, n_rank: int = 100
) -> pd.Series:
    """Calculate Connors RSI (Composite RSI of price, streak, and rank)."""
    if len(close) < n_rsi:
        return pd.Series(np.nan, index=close.index, dtype=float)

    rsi_p = ta.momentum.RSIIndicator(close, window=n_rsi, fillna=False).rsi()
    streak = _calc_streak_series(close)
    rsi_s = ta.momentum.RSIIndicator(streak, window=n_streak, fillna=False).rsi()

    pct_chg = close.pct_change()
    # Percent rank over rolling window
    pct_rank = pct_chg.rolling(n_rank, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0, raw=False
    )

    crsi = (rsi_p + rsi_s + pct_rank) / 3.0
    return crsi


def _calc_rsi_divergence(
    high: pd.Series, low: pd.Series, rsi: pd.Series, lookback: int = 5
) -> pd.Series:
    """Calculate 5-bar regular bullish/bearish RSI divergence."""
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    r_arr = rsi.to_numpy(dtype=float, na_value=np.nan)

    l_prev = low.shift(lookback).to_numpy(dtype=float, na_value=np.nan)
    h_prev = high.shift(lookback).to_numpy(dtype=float, na_value=np.nan)
    r_prev = rsi.shift(lookback).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(l_arr) & ~np.isnan(r_arr) & ~np.isnan(l_prev) & ~np.isnan(r_prev)

    bull_div = valid & (l_arr < l_prev) & (r_arr > r_prev)
    bear_div = valid & (h_arr > h_prev) & (r_arr < r_prev)
    hold = valid & ~bull_div & ~bear_div

    conds = [bull_div, bear_div, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=low.index, dtype=str
    )


def _calc_min_max_rsi(close: pd.Series, rsi: pd.Series, lookback: int = 10) -> pd.Series:
    """Calculate 10-period min/max extreme combined with RSI overbought/oversold."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    r = rsi.to_numpy(dtype=float, na_value=np.nan)
    min_c = close.rolling(lookback, min_periods=1).min().to_numpy(dtype=float, na_value=np.nan)
    max_c = close.rolling(lookback, min_periods=1).max().to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(r) & ~np.isnan(min_c) & ~np.isnan(max_c)
    bull = valid & (c <= min_c) & (r < 30.0)
    bear = valid & (c >= max_c) & (r > 70.0)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_mfi_reversal(mfi: pd.Series) -> pd.Series:
    """Calculate MFI exiting overbought/oversold reversal signal."""
    m = mfi.to_numpy(dtype=float, na_value=np.nan)
    m1 = mfi.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(m) & ~np.isnan(m1)
    bull = valid & (m < 20.0) & (m > m1)
    bear = valid & (m > 80.0) & (m < m1)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=mfi.index, dtype=str
    )


def _calc_momentum_shift(close: pd.Series) -> pd.Series:
    """Calculate 3-bar Momentum Shift signal."""
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


def _calc_return_threshold(close: pd.Series, lookback: int, threshold: float) -> pd.Series:
    """Calculate Return threshold crossing signal."""
    ret = close.pct_change(lookback).to_numpy(dtype=float, na_value=np.nan)
    valid = ~np.isnan(ret)
    bull = valid & (ret > threshold)
    bear = valid & (ret < -threshold)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def generate_momentum_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Generate all 30 momentum and oscillator signals from OHLCV dataframe.

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

    with GroupProgressBar(
        "Momentum Signals", total=len(MOMENTUM_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if len(df_norm) == 0:
            pbar.update(len(MOMENTUM_SIGNAL_COLUMNS))
            return pd.DataFrame(
                {col: pd.Series(dtype=str, index=df.index) for col in MOMENTUM_SIGNAL_COLUMNS},
                index=df.index,
            )

        high = df_norm["high"]
        low = df_norm["low"]
        close = df_norm["close"]
        volume = df_norm["volume"]

        signals = pd.DataFrame(index=df_norm.index)

        # 1. RSI Overbought / Oversold and Midline (6)
        rsi14 = ta.momentum.RSIIndicator(close=close, window=14, fillna=False).rsi()
        rsi7 = ta.momentum.RSIIndicator(close=close, window=7, fillna=False).rsi()
        rsi21 = ta.momentum.RSIIndicator(close=close, window=21, fillna=False).rsi()
        rsi28 = ta.momentum.RSIIndicator(close=close, window=28, fillna=False).rsi()

        signals["mom_rsi_ob_os_14_signal"] = _bound_signal(rsi14, 30.0, 70.0, buy_below=True)
        signals["mom_rsi_ob_os_7_signal"] = _bound_signal(rsi7, 20.0, 80.0, buy_below=True)
        signals["mom_rsi_ob_os_21_signal"] = _bound_signal(rsi21, 30.0, 70.0, buy_below=True)
        signals["mom_rsi_ob_os_28_signal"] = _bound_signal(rsi28, 30.0, 70.0, buy_below=True)

        centerline_50 = pd.Series(50.0, index=close.index)
        signals["mom_rsi_50_cross_14_signal"] = _crossover_signal(rsi14, centerline_50)
        signals["mom_rsi_50_cross_21_signal"] = _crossover_signal(rsi21, centerline_50)
        pbar.update(6)

        # 2. Stochastic Oscillator (2)
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
        pbar.update(2)

        # 3. Stochastic RSI (1)
        stoch_rsi_14 = ta.momentum.StochRSIIndicator(
            close=close, window=14, smooth1=3, smooth2=3, fillna=False
        )
        signals["mom_stoch_rsi_cross_14_signal"] = _crossover_signal(
            stoch_rsi_14.stochrsi_k(), stoch_rsi_14.stochrsi_d()
        )
        pbar.update(1)

        # 4. Williams %R (2)
        wr14 = ta.momentum.WilliamsRIndicator(
            high=high, low=low, close=close, lbp=14, fillna=False
        ).williams_r()
        signals["mom_williams_r_14_signal"] = _bound_signal(wr14, -80.0, -20.0, buy_below=True)

        wr28 = ta.momentum.WilliamsRIndicator(
            high=high, low=low, close=close, lbp=28, fillna=False
        ).williams_r()
        signals["mom_williams_r_28_signal"] = _bound_signal(wr28, -80.0, -20.0, buy_below=True)
        pbar.update(2)

        # 5. Commodity Channel Index (CCI) (2)
        cci14 = ta.trend.CCIIndicator(
            high=high, low=low, close=close, window=14, fillna=False
        ).cci()
        signals["mom_cci_100_14_signal"] = _bound_signal(cci14, -100.0, 100.0, buy_below=True)

        cci20 = ta.trend.CCIIndicator(
            high=high, low=low, close=close, window=20, fillna=False
        ).cci()
        signals["mom_cci_200_20_signal"] = _bound_signal(cci20, -200.0, 200.0, buy_below=True)
        pbar.update(2)

        # 6. Rate of Change (ROC) (3)
        centerline_zero = pd.Series(0.0, index=close.index)
        roc5 = ta.momentum.ROCIndicator(close=close, window=5, fillna=False).roc()
        roc10 = ta.momentum.ROCIndicator(close=close, window=10, fillna=False).roc()
        roc20 = ta.momentum.ROCIndicator(close=close, window=20, fillna=False).roc()

        signals["mom_roc_zero_cross_5_signal"] = _crossover_signal(roc5, centerline_zero)
        signals["mom_roc_zero_cross_10_signal"] = _crossover_signal(roc10, centerline_zero)
        signals["mom_roc_zero_cross_20_signal"] = _crossover_signal(roc20, centerline_zero)
        pbar.update(3)

        # 7. Money Flow Index (MFI) Overbought/Oversold (14) (1)
        mfi14 = ta.volume.MFIIndicator(
            high=high, low=low, close=close, volume=volume, window=14, fillna=False
        ).money_flow_index()
        signals["mom_mfi_ob_os_14_signal"] = _bound_signal(mfi14, 20.0, 80.0, buy_below=True)
        pbar.update(1)

        # 8. True Strength Index (TSI) Zero Line Crossover (13, 25) (1)
        tsi = ta.momentum.TSIIndicator(
            close=close, window_slow=25, window_fast=13, fillna=False
        ).tsi()
        signals["mom_tsi_cross_13_25_signal"] = _crossover_signal(tsi, centerline_zero)
        pbar.update(1)

        # 9. Fisher Transform Crossover (9) (1)
        fisher_k, fisher_d = _calc_fisher(high, low, length=9)
        signals["mom_fisher_cross_9_signal"] = _crossover_signal(fisher_k, fisher_d)
        pbar.update(1)

        # 10. Awesome Oscillator (AO) Zero Cross and Saucer Pattern (2)
        ao = ta.momentum.AwesomeOscillatorIndicator(
            high=high, low=low, window1=5, window2=34, fillna=False
        ).awesome_oscillator()
        signals["mom_ao_zero_cross_signal"] = _crossover_signal(ao, centerline_zero)
        signals["mom_ao_saucer_signal"] = _calc_ao_saucer(ao)
        pbar.update(2)

        # 11. Ultimate Oscillator Boundary Extremes (7, 14, 28) (1)
        uo = ta.momentum.UltimateOscillator(
            high=high, low=low, close=close, window1=7, window2=14, window3=28, fillna=False
        ).ultimate_oscillator()
        signals["mom_ultimate_osc_signal"] = _bound_signal(uo, 30.0, 70.0, buy_below=True)
        pbar.update(1)

        # 12. Chande Momentum Oscillator (CMO) Thresholds (14) (1)
        cmo14 = _calc_cmo(close, length=14)
        signals["mom_cmo_14_signal"] = _bound_signal(cmo14, -50.0, 50.0, buy_below=False)
        pbar.update(1)

        # 13. Min Max 10 + RSI Reversal (1)
        signals["mom_min_max_10_rsi_signal"] = _calc_min_max_rsi(close, rsi14, lookback=10)
        pbar.update(1)

        # 14. RSI Regular Divergence (5 bars) (1)
        signals["mom_rsi_divergence_5_signal"] = _calc_rsi_divergence(high, low, rsi14, lookback=5)
        pbar.update(1)

        # 15. Connors RSI (3, 2, 100) (1)
        crsi = _calc_connors_rsi(close, n_rsi=3, n_streak=2, n_rank=100)
        signals["mom_connors_rsi_3_2_100_signal"] = _bound_signal(crsi, 15.0, 85.0, buy_below=True)
        pbar.update(1)

        # 16. MFI Reversal from Bounds (1)
        signals["mom_mfi_reversal_20_80_signal"] = _calc_mfi_reversal(mfi14)
        pbar.update(1)

        # 17. Momentum Shift (3 bars) (1)
        signals["mom_shift_3_bar_signal"] = _calc_momentum_shift(close)
        pbar.update(1)

        # 18. Return Momentum (5 bars > 2%) (1)
        signals["mom_return_momentum_5_signal"] = _calc_return_threshold(
            close, lookback=5, threshold=0.02
        )
        pbar.update(1)

        # 19. Extreme Move (10 bars > 5%) (1)
        signals["mom_extreme_move_10_signal"] = _calc_return_threshold(
            close, lookback=10, threshold=0.05
        )
        pbar.update(1)

        # Ensure all columns are present, filled with NONE, and matching index
        for col in MOMENTUM_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[MOMENTUM_SIGNAL_COLUMNS]
