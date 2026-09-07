from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
from signalx.signals.session_helper import SessionContext, extract_session_context
from signalx.utils import normalize_ohlcv

SMC_SIGNAL_COLUMNS = [
    "smc_fvg_bullish_mitigation_signal",
    "smc_fvg_bearish_mitigation_signal",
    "smc_order_block_retest_signal",
    "smc_break_of_structure_signal",
    "smc_change_of_character_signal",
    "smc_market_structure_break_signal",
    "smc_liquidity_sweep_signal",
    "smc_equal_high_low_sweep_signal",
    "smc_judas_swing_signal",
    "smc_inducement_sweep_signal",
    "smc_pdh_pdl_sweep_signal",
    "smc_morning_midpoint_acceptance_signal",
]


def _calc_fvg_bullish_mitigation(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Bullish Fair Value Gap (FVG) mitigation and breakdown signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    # Bullish FVG created when Low > High[2]
    fvg_created = low > high.shift(2)
    fvg_top = low.where(fvg_created).shift(1).ffill().to_numpy(dtype=float, na_value=np.nan)
    fvg_bottom = (
        high.shift(2).where(fvg_created).shift(1).ffill().to_numpy(dtype=float, na_value=np.nan)
    )

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(fvg_top) & ~np.isnan(fvg_bottom)

    retrace = valid & (l_arr <= fvg_top) & (c >= fvg_bottom) & (c > o)
    breakdown = valid & (c < fvg_bottom)

    condlist = [retrace, breakdown, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_fvg_bearish_mitigation(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Bearish Fair Value Gap (FVG) mitigation and breakout signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    # Bearish FVG created when High < Low[2]
    fvg_created = high < low.shift(2)
    fvg_top = (
        low.shift(2).where(fvg_created).shift(1).ffill().to_numpy(dtype=float, na_value=np.nan)
    )
    fvg_bottom = high.where(fvg_created).shift(1).ffill().to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(fvg_top) & ~np.isnan(fvg_bottom)

    retrace = valid & (h >= fvg_bottom) & (c <= fvg_top) & (c < o)
    breakout = valid & (c > fvg_top)

    condlist = [retrace, breakout, valid]
    choicelist = [SignalState.SELL, SignalState.BUY, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_order_block_retest(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Bullish and Bearish Order Block (OB) formation and retest signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    # Bullish OB: bar i-3 red, bars i-2, i-1, i green
    bull_ob_created = (
        (close.shift(3) < open_p.shift(3))
        & (close.shift(2) > open_p.shift(2))
        & (close.shift(1) > open_p.shift(1))
        & (close > open_p)
    )
    ob_bull_top = (
        open_p.shift(3)
        .where(bull_ob_created)
        .shift(1)
        .ffill()
        .to_numpy(dtype=float, na_value=np.nan)
    )
    ob_bull_bottom = (
        close.shift(3)
        .where(bull_ob_created)
        .shift(1)
        .ffill()
        .to_numpy(dtype=float, na_value=np.nan)
    )

    # Bearish OB: bar i-3 green, bars i-2, i-1, i red
    bear_ob_created = (
        (close.shift(3) > open_p.shift(3))
        & (close.shift(2) < open_p.shift(2))
        & (close.shift(1) < open_p.shift(1))
        & (close < open_p)
    )
    ob_bear_top = (
        close.shift(3)
        .where(bear_ob_created)
        .shift(1)
        .ffill()
        .to_numpy(dtype=float, na_value=np.nan)
    )
    ob_bear_bottom = (
        open_p.shift(3)
        .where(bear_ob_created)
        .shift(1)
        .ffill()
        .to_numpy(dtype=float, na_value=np.nan)
    )

    valid_bull = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(ob_bull_top) & ~np.isnan(ob_bull_bottom)
    valid_bear = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(ob_bear_top) & ~np.isnan(ob_bear_bottom)
    valid = ~np.isnan(o) & ~np.isnan(c)

    bull_retest = valid_bull & (l_arr <= ob_bull_top) & (c >= ob_bull_bottom) & (c > o)
    bear_retest = valid_bear & (h >= ob_bear_bottom) & (c <= ob_bear_top) & (c < o)

    condlist = [bull_retest, bear_retest, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_break_of_structure(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Break of Structure (BOS) aligned with trend filter."""
    n = len(close)
    if n == 0:
        return pd.Series(dtype=str, index=close.index)

    high_10_prev = high.rolling(10).max().shift(1).to_numpy(dtype=float, na_value=np.nan)
    low_10_prev = low.rolling(10).min().shift(1).to_numpy(dtype=float, na_value=np.nan)
    sma20 = close.rolling(20, min_periods=20).mean().to_numpy(dtype=float, na_value=np.nan)
    sma50 = close.rolling(50, min_periods=50).mean().to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(c)
        & ~np.isnan(high_10_prev)
        & ~np.isnan(low_10_prev)
        & ~np.isnan(sma20)
        & ~np.isnan(sma50)
    )

    bull_bos = valid & (c > high_10_prev) & (sma20 > sma50)
    bear_bos = valid & (c < low_10_prev) & (sma20 < sma50)

    condlist = [bull_bos, bear_bos, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_change_of_character(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Change of Character (CHoCH) structural trend reversal signal."""
    n = len(close)
    if n == 0:
        return pd.Series(dtype=str, index=close.index)

    high_5_prev = high.rolling(5).max().shift(1).to_numpy(dtype=float, na_value=np.nan)
    low_5_prev = low.rolling(5).min().shift(1).to_numpy(dtype=float, na_value=np.nan)
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()

    bear_ratio = (sma20 < sma50).rolling(10, min_periods=1).mean().shift(1)
    prev_bear_regime = (bear_ratio >= 0.5).to_numpy(dtype=bool) & ~bear_ratio.isna().to_numpy()

    bull_ratio = (sma20 > sma50).rolling(10, min_periods=1).mean().shift(1)
    prev_bull_regime = (bull_ratio >= 0.5).to_numpy(dtype=bool) & ~bull_ratio.isna().to_numpy()

    c = close.to_numpy(dtype=float, na_value=np.nan)
    sma20_arr = sma20.to_numpy(dtype=float, na_value=np.nan)
    sma50_arr = sma50.to_numpy(dtype=float, na_value=np.nan)
    valid = (
        ~np.isnan(c)
        & ~np.isnan(high_5_prev)
        & ~np.isnan(low_5_prev)
        & ~np.isnan(sma20_arr)
        & ~np.isnan(sma50_arr)
    )

    bull_choch = valid & (c > high_5_prev) & prev_bear_regime
    bear_choch = valid & (c < low_5_prev) & prev_bull_regime

    condlist = [bull_choch, bear_choch, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_market_structure_break(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 10
) -> pd.Series:
    """Calculate Market Structure Break (MSB) signal."""
    recent_high = high.rolling(lookback).max().shift(1).to_numpy(dtype=float, na_value=np.nan)
    recent_low = low.rolling(lookback).min().shift(1).to_numpy(dtype=float, na_value=np.nan)
    recent_high_prev = (
        high.rolling(lookback).max().shift(lookback + 1).to_numpy(dtype=float, na_value=np.nan)
    )
    recent_low_prev = (
        low.rolling(lookback).min().shift(lookback + 1).to_numpy(dtype=float, na_value=np.nan)
    )

    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(c)
        & ~np.isnan(recent_high)
        & ~np.isnan(recent_low)
        & ~np.isnan(recent_high_prev)
        & ~np.isnan(recent_low_prev)
    )
    bull = valid & (c > recent_high) & (recent_low < recent_low_prev)
    bear = valid & (c < recent_low) & (recent_high > recent_high_prev)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_liquidity_sweep(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Liquidity Sweep (sweep of 5-bar high/low with close back inside)."""
    low_5_prev = low.rolling(5).min().shift(1).to_numpy(dtype=float, na_value=np.nan)
    high_5_prev = high.rolling(5).max().shift(1).to_numpy(dtype=float, na_value=np.nan)

    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(low_5_prev) & ~np.isnan(high_5_prev)

    bull_sweep = valid & (l_arr < low_5_prev) & (c > low_5_prev)
    bear_sweep = valid & (h > high_5_prev) & (c < high_5_prev)

    condlist = [bull_sweep, bear_sweep, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_equal_high_low_sweep(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Equal High / Low liquidity sweep signal."""
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(l_prev) & ~np.isnan(h_prev)
    equal_low = valid & (np.abs(l_arr - l_prev) <= (0.001 * c))
    equal_high = valid & (np.abs(h_arr - h_prev) <= (0.001 * c))

    bull_equal = equal_low & (c > l_prev)
    bear_equal = equal_high & (c < h_prev)

    condlist = [bull_equal, bear_equal, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_judas_swing(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Judas Swing (false breakout stop-hunt with reversal close)."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    low_5_prev = low.rolling(5).min().shift(1).to_numpy(dtype=float, na_value=np.nan)
    high_5_prev = high.rolling(5).max().shift(1).to_numpy(dtype=float, na_value=np.nan)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(low_5_prev) & ~np.isnan(high_5_prev)
    midpoint = (h + l_arr) / 2.0

    bull_judas = valid & (l_arr < low_5_prev) & (c > o) & (c > midpoint)
    bear_judas = valid & (h > high_5_prev) & (c < o) & (c < midpoint)

    condlist = [bull_judas, bear_judas, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_inducement_sweep(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Inducement Sweep (minor liquidity sweep with long wick rejection)."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)
    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(o)
        & ~np.isnan(h)
        & ~np.isnan(l_arr)
        & ~np.isnan(c)
        & ~np.isnan(l_prev)
        & ~np.isnan(h_prev)
    )
    rng = np.where(valid, h - l_arr, 0.0)
    lower_wick = np.where(valid, np.minimum(o, c) - l_arr, 0.0)
    upper_wick = np.where(valid, h - np.maximum(o, c), 0.0)

    has_range = valid & (rng > 0.0)
    bull_sweep = has_range & (l_arr < l_prev) & (lower_wick >= 0.50 * rng) & (c > o)
    bear_sweep = has_range & (h > h_prev) & (upper_wick >= 0.50 * rng) & (c < o)

    condlist = [bull_sweep, bear_sweep, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_pdh_pdl_sweep(
    df: pd.DataFrame | None,
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate Prior Day High / Low (PDH/PDL) Liquidity Sweep & Mean-Reversion.

    Prior session PDH = max(High of previous session), PDL = min(Low of previous session).
    Uses session_id strictly with shift(1) across sessions so that for session 0 / warmup
    there is no prior session (PDH/PDL is NaN, producing SignalState.NONE) -> Zero lookahead bias.

    Bullish sweep (sweeps PDL): (low < pdl) & (close > pdl) & (close > open) -> SignalState.BUY.
    Bearish sweep (sweeps PDH): (high > pdh) & (close < pdh) & (close < open) -> SignalState.SELL.
    Otherwise: SignalState.NONE.
    """
    if len(close) == 0:
        return pd.Series(dtype=str, index=close.index)

    if df is None:
        df = pd.DataFrame(
            {"open": open_p, "high": high, "low": low, "close": close},
            index=close.index,
        )

    ctx = extract_session_context(df)

    session_high = high.groupby(ctx.session_id, sort=False).max()
    session_low = low.groupby(ctx.session_id, sort=False).min()

    prior_high = session_high.shift(1)
    prior_low = session_low.shift(1)

    pdh = ctx.session_id.map(prior_high).to_numpy(dtype=float, na_value=np.nan)
    pdl = ctx.session_id.map(prior_low).to_numpy(dtype=float, na_value=np.nan)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    lo = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(lo) & ~np.isnan(c) & ~np.isnan(pdh) & ~np.isnan(pdl)
    )

    buy = valid & (lo < pdl) & (c > pdl) & (c > o)
    sell = valid & (h > pdh) & (c < pdh) & (c < o)

    condlist = [buy, sell]
    choicelist = [SignalState.BUY, SignalState.SELL]
    return pd.Series(
        np.select(condlist, choicelist, default=SignalState.NONE),
        index=close.index,
        dtype=str,
    )


def _calc_morning_midpoint_acceptance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    session_ctx: SessionContext,
) -> pd.Series:
    """Calculate Morning Midpoint Acceptance Signal (SMC012).

    Morning_Mid = (High_<=11:00 + Low_<=11:00) / 2.0
    Accept_long = rolling_mean(Close > Morning_Mid, 4).shift(1)
    Accept_short = rolling_mean(Close < Morning_Mid, 4).shift(1)

    Triggers (evaluated in afternoon window t >= 13:00):
        buy: t >= 13:00 and Accept_long >= 0.50 and Close > High_<=11:00 and RSI_8 > 54
        sell: t >= 13:00 and Accept_short >= 0.50 and Close < Low_<=11:00 and RSI_8 < 46
        hold: t >= 13:00 and ((Close > High_<=11:00) | (Close < Low_<=11:00))
        none: default
    """
    if len(close) == 0:
        return pd.Series(dtype=str, index=close.index)

    is_datetime = pd.api.types.is_string_dtype(session_ctx.session_id) or (
        len(session_ctx.session_id) > 0 and isinstance(session_ctx.session_id.iloc[0], str)
    )

    if is_datetime:
        morning_mask = session_ctx.time_minutes <= 660
        afternoon_mask = session_ctx.time_minutes >= 780
    else:
        morning_mask = session_ctx.bar_in_session <= 25
        afternoon_mask = session_ctx.bar_in_session >= 36

    morning_high_sub = high.where(morning_mask)
    morning_low_sub = low.where(morning_mask)
    morning_high = morning_high_sub.groupby(session_ctx.session_id, sort=False).transform("max")
    morning_low = morning_low_sub.groupby(session_ctx.session_id, sort=False).transform("min")
    morning_mid = (morning_high + morning_low) / 2.0

    accept_long = (close > morning_mid).astype(float).rolling(4, min_periods=1).mean().shift(1)
    accept_short = (close < morning_mid).astype(float).rolling(4, min_periods=1).mean().shift(1)

    try:
        rsi8 = ta.momentum.RSIIndicator(close=close, window=8, fillna=False).rsi()
    except Exception:
        rsi8 = pd.Series(np.nan, index=close.index)

    c = close.to_numpy(dtype=float, na_value=np.nan)
    mh = morning_high.to_numpy(dtype=float, na_value=np.nan)
    ml = morning_low.to_numpy(dtype=float, na_value=np.nan)
    al = accept_long.to_numpy(dtype=float, na_value=np.nan)
    as_ = accept_short.to_numpy(dtype=float, na_value=np.nan)
    r8 = rsi8.to_numpy(dtype=float, na_value=np.nan)
    is_afternoon = afternoon_mask.to_numpy(dtype=bool)

    valid = (
        is_afternoon
        & ~np.isnan(c)
        & ~np.isnan(mh)
        & ~np.isnan(ml)
        & ~np.isnan(al)
        & ~np.isnan(as_)
        & ~np.isnan(r8)
    )

    buy = valid & (al >= 0.50) & (c > mh) & (r8 > 54.0)
    sell = valid & (as_ >= 0.50) & (c < ml) & (r8 < 46.0)
    hold = valid & ~buy & ~sell & ((c > mh) | (c < ml))

    condlist = [buy, sell, hold]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(condlist, choicelist, default=SignalState.NONE),
        index=close.index,
        dtype=str,
    )


def generate_smc_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Generate all 12 Smart Money Concepts (SMC) signals from OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
    show_progress : bool, default False
        Whether to display a real-time progress bar for this signal group.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 12 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)

    with GroupProgressBar(
        "SMC Signals", total=len(SMC_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if len(df_norm) == 0:
            pbar.update(len(SMC_SIGNAL_COLUMNS))
            return pd.DataFrame(
                {col: pd.Series(dtype=str, index=df.index) for col in SMC_SIGNAL_COLUMNS},
                index=df.index,
            )

        open_p = df_norm["open"]
        high = df_norm["high"]
        low = df_norm["low"]
        close = df_norm["close"]
        session_ctx = extract_session_context(df_norm)

        signals = pd.DataFrame(index=df_norm.index)

        # 1. Fair Value Gap (FVG) Bullish Mitigation (1)
        signals["smc_fvg_bullish_mitigation_signal"] = _calc_fvg_bullish_mitigation(
            open_p, high, low, close
        )
        pbar.update(1)

        # 2. Fair Value Gap (FVG) Bearish Mitigation (1)
        signals["smc_fvg_bearish_mitigation_signal"] = _calc_fvg_bearish_mitigation(
            open_p, high, low, close
        )
        pbar.update(1)

        # 3. Order Block Retest (1)
        signals["smc_order_block_retest_signal"] = _calc_order_block_retest(
            open_p, high, low, close
        )
        pbar.update(1)

        # 4. Break of Structure (BOS) (1)
        signals["smc_break_of_structure_signal"] = _calc_break_of_structure(high, low, close)
        pbar.update(1)

        # 5. Change of Character (CHoCH) (1)
        signals["smc_change_of_character_signal"] = _calc_change_of_character(high, low, close)
        pbar.update(1)

        # 6. Market Structure Break (MSB) (1)
        signals["smc_market_structure_break_signal"] = _calc_market_structure_break(
            high, low, close
        )
        pbar.update(1)

        # 7. Liquidity Sweep (1)
        signals["smc_liquidity_sweep_signal"] = _calc_liquidity_sweep(high, low, close)
        pbar.update(1)

        # 8. Equal High/Low Sweep (1)
        signals["smc_equal_high_low_sweep_signal"] = _calc_equal_high_low_sweep(high, low, close)
        pbar.update(1)

        # 9. Judas Swing (1)
        signals["smc_judas_swing_signal"] = _calc_judas_swing(open_p, high, low, close)
        pbar.update(1)

        # 10. Inducement Sweep (1)
        signals["smc_inducement_sweep_signal"] = _calc_inducement_sweep(open_p, high, low, close)
        pbar.update(1)

        # 11. Prior Day High / Low Sweep (1)
        signals["smc_pdh_pdl_sweep_signal"] = _calc_pdh_pdl_sweep(df_norm, open_p, high, low, close)
        pbar.update(1)

        # 12. Morning Midpoint Acceptance (1)
        signals["smc_morning_midpoint_acceptance_signal"] = _calc_morning_midpoint_acceptance(
            high, low, close, session_ctx
        )
        pbar.update(1)

        # Ensure all columns are present, filled with NONE, and match index
        for col in SMC_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[SMC_SIGNAL_COLUMNS]
