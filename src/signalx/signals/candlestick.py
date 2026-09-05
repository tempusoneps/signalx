from __future__ import annotations

import numpy as np
import pandas as pd

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
from signalx.signals.session_helper import extract_session_context
from signalx.utils import normalize_ohlcv

CANDLESTICK_SIGNAL_COLUMNS = [
    "cdl_engulfing_signal",
    "cdl_hammer_star_signal",
    "cdl_pinbar_signal",
    "cdl_marubozu_signal",
    "cdl_harami_signal",
    "cdl_inside_bar_breakout_signal",
    "cdl_outside_bar_signal",
    "cdl_doji_reversal_signal",
    "cdl_three_soldiers_crows_signal",
    "cdl_consecutive_3_signal",
    "cdl_consecutive_5_signal",
    "cdl_morning_evening_star_signal",
    "cdl_piercing_darkcloud_signal",
    "cdl_tweezer_tops_bottoms_signal",
    "cdl_couple_cs_signal",
    "cdl_fakey_pattern_signal",
    "cdl_liquidity_sweep_signal",
    "cdl_equal_high_low_sweep_signal",
    "cdl_gap_up_down_signal",
    "cdl_body_size_expansion_signal",
    "cdl_wick_rejection_signal",
    "cdl_body_direction_signal",
    "cdl_close_strength_signal",
    "cdl_price_rejection_signal",
    "cdl_break_retest_signal",
    "cdl_trend_exhaustion_signal",
    "cdl_final_push_signal",
    "cdl_fvg_bullish_mitigation_signal",
    "cdl_fvg_bearish_mitigation_signal",
    "cdl_order_block_retest_signal",
    "cdl_break_of_structure_signal",
    "cdl_change_of_character_signal",
    "cdl_judas_swing_signal",
    "cdl_inducement_sweep_signal",
    "cdl_thrust_bar_signal",
    "cdl_narrow_range_7_breakout_signal",
    "cdl_wide_range_reversal_signal",
    "cdl_pdh_pdl_sweep_signal",
]


def _calc_engulfing(open_p: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Bullish and Bearish Engulfing 2-bar pattern signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    o_prev = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(o_prev) & ~np.isnan(c_prev)

    bullish_engulf = valid & (c_prev < o_prev) & (c > o) & (o <= c_prev) & (c >= o_prev)
    bearish_engulf = valid & (c_prev > o_prev) & (c < o) & (o >= c_prev) & (c <= o_prev)

    condlist = [
        bullish_engulf,
        bearish_engulf,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_hammer_star(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate Hammer / Inverted Hammer and Shooting Star / Hanging Man signals."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(l_arr) & ~np.isnan(c)

    rng = np.where(valid, h - l_arr, 0.0)
    body = np.where(valid, np.abs(c - o), 0.0)
    upper_wick = np.where(valid, h - np.maximum(o, c), 0.0)
    lower_wick = np.where(valid, np.minimum(o, c) - l_arr, 0.0)

    has_range = valid & (rng > 0.0)

    hammer_rejection = has_range & (lower_wick >= 2.0 * body) & (upper_wick <= 0.25 * rng)
    star_rejection = has_range & (upper_wick >= 2.0 * body) & (lower_wick <= 0.25 * rng)

    condlist = [
        hammer_rejection,
        star_rejection,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_pinbar(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate Pinbar price rejection signal (wick >= 60% of candle range)."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(l_arr) & ~np.isnan(c)

    rng = np.where(valid, h - l_arr, 0.0)
    body = np.where(valid, np.abs(c - o), 0.0)
    upper_wick = np.where(valid, h - np.maximum(o, c), 0.0)
    lower_wick = np.where(valid, np.minimum(o, c) - l_arr, 0.0)

    has_range = valid & (rng > 0.0)

    bullish_pinbar = has_range & (lower_wick >= 0.60 * rng) & (lower_wick >= 2.0 * body)
    bearish_pinbar = has_range & (upper_wick >= 0.60 * rng) & (upper_wick >= 2.0 * body)

    condlist = [
        bullish_pinbar,
        bearish_pinbar,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_marubozu(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate Marubozu strong momentum directional candle signal (body >= 85% range)."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(l_arr) & ~np.isnan(c)

    rng = np.where(valid, h - l_arr, 0.0)
    body = np.where(valid, np.abs(c - o), 0.0)

    has_range = valid & (rng > 0.0)

    bullish_marubozu = has_range & (c > o) & (body >= 0.85 * rng)
    bearish_marubozu = has_range & (c < o) & (body >= 0.85 * rng)

    condlist = [
        bullish_marubozu,
        bearish_marubozu,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_harami(open_p: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Bullish and Bearish Harami inside body pattern signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    o_prev = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(o_prev) & ~np.isnan(c_prev)

    bullish_harami = (
        valid
        & (c_prev < o_prev)
        & (c > o)
        & (o >= c_prev)
        & (c <= o_prev)
        & ((c - o) < (o_prev - c_prev))
    )
    bearish_harami = (
        valid
        & (c_prev > o_prev)
        & (c < o)
        & (o <= c_prev)
        & (c >= o_prev)
        & ((o - c) < (c_prev - o_prev))
    )

    condlist = [
        bullish_harami,
        bearish_harami,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_inside_bar(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate Inside Bar breakout signal with directional close."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(o)
        & ~np.isnan(h)
        & ~np.isnan(l_arr)
        & ~np.isnan(c)
        & ~np.isnan(h_prev)
        & ~np.isnan(l_prev)
    )

    inside = valid & (h <= h_prev) & (l_arr >= l_prev)
    bullish_inside = inside & (c > o)
    bearish_inside = inside & (c < o)

    condlist = [
        bullish_inside,
        bearish_inside,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_outside_bar(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate Outside Bar breakout signal with directional close."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(o)
        & ~np.isnan(h)
        & ~np.isnan(l_arr)
        & ~np.isnan(c)
        & ~np.isnan(h_prev)
        & ~np.isnan(l_prev)
    )

    outside = valid & (h > h_prev) & (l_arr < l_prev)
    bullish_outside = outside & (c > o)
    bearish_outside = outside & (c < o)

    condlist = [
        bullish_outside,
        bearish_outside,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_doji_reversal(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate Dragonfly Doji (bullish) and Gravestone Doji (bearish) reversal signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(l_arr) & ~np.isnan(c)

    rng = np.where(valid, h - l_arr, 0.0)
    body = np.where(valid, np.abs(c - o), 0.0)
    upper_wick = np.where(valid, h - np.maximum(o, c), 0.0)
    lower_wick = np.where(valid, np.minimum(o, c) - l_arr, 0.0)

    has_range = valid & (rng > 0.0)
    tiny_body = has_range & (body <= 0.10 * rng)

    dragonfly = tiny_body & (lower_wick >= 0.60 * rng) & (upper_wick <= 0.15 * rng)
    gravestone = tiny_body & (upper_wick >= 0.60 * rng) & (lower_wick <= 0.15 * rng)

    condlist = [
        dragonfly,
        gravestone,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_three_soldiers_crows(open_p: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Three White Soldiers (bullish) and Three Black Crows (bearish) signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    o1 = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    o2 = open_p.shift(2).to_numpy(dtype=float, na_value=np.nan)
    c2 = close.shift(2).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(o1) & ~np.isnan(c1) & ~np.isnan(o2) & ~np.isnan(c2)
    )

    three_soldiers = (
        valid
        & (c > o)
        & (c1 > o1)
        & (c2 > o2)
        & (c > c1)
        & (c1 > c2)
        & (o >= o1)
        & (o <= c1)
        & (o1 >= o2)
        & (o1 <= c2)
    )
    three_crows = (
        valid
        & (c < o)
        & (c1 < o1)
        & (c2 < o2)
        & (c < c1)
        & (c1 < c2)
        & (o <= o1)
        & (o >= c1)
        & (o1 <= o2)
        & (o1 >= c2)
    )

    condlist = [
        three_soldiers,
        three_crows,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_consecutive_directional(
    open_p: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Calculate consecutive directional bars signal (N consecutive green/red closes)."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid_bar = ~np.isnan(o) & ~np.isnan(c)
    bullish_bar = np.where(valid_bar & (c > o), 1.0, 0.0)
    bearish_bar = np.where(valid_bar & (c < o), 1.0, 0.0)

    bullish_roll = pd.Series(bullish_bar, index=open_p.index).rolling(window).sum()
    bearish_roll = pd.Series(bearish_bar, index=open_p.index).rolling(window).sum()
    valid_roll = pd.Series(valid_bar.astype(float), index=open_p.index).rolling(window).sum()

    valid_window = (valid_roll == float(window)).to_numpy()
    consec_bullish = valid_window & (bullish_roll.to_numpy() == float(window))
    consec_bearish = valid_window & (bearish_roll.to_numpy() == float(window))

    condlist = [
        consec_bullish,
        consec_bearish,
        valid_window,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_morning_evening_star(open_p: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Morning Star (bullish) and Evening Star (bearish) 3-bar reversal signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    o1 = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    o2 = open_p.shift(2).to_numpy(dtype=float, na_value=np.nan)
    c2 = close.shift(2).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(o1) & ~np.isnan(c1) & ~np.isnan(o2) & ~np.isnan(c2)
    )

    body2 = np.where(valid, np.abs(o2 - c2), 0.0)
    body1 = np.where(valid, np.abs(o1 - c1), 0.0)

    morning_star = (
        valid
        & (c2 < o2)
        & (body1 <= 0.5 * body2)
        & (np.maximum(o1, c1) <= o2)
        & (np.minimum(o1, c1) <= c2)
        & (c > o)
        & (c >= (o2 + c2) / 2.0)
    )

    evening_star = (
        valid
        & (c2 > o2)
        & (body1 <= 0.5 * body2)
        & (np.minimum(o1, c1) >= o2)
        & (np.maximum(o1, c1) >= c2)
        & (c < o)
        & (c <= (o2 + c2) / 2.0)
    )

    condlist = [
        morning_star,
        evening_star,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_piercing_darkcloud(open_p: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Piercing Line (bullish) and Dark Cloud Cover (bearish) reversal signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    o_prev = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(o_prev) & ~np.isnan(c_prev)

    piercing_line = (
        valid
        & (c_prev < o_prev)
        & (c > o)
        & (o <= c_prev)
        & (c > (o_prev + c_prev) / 2.0)
        & (c <= o_prev)
    )

    dark_cloud = (
        valid
        & (c_prev > o_prev)
        & (c < o)
        & (o >= c_prev)
        & (c < (o_prev + c_prev) / 2.0)
        & (c >= o_prev)
    )

    condlist = [
        piercing_line,
        dark_cloud,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_tweezer_tops_bottoms(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    rel_tol: float = 0.002,
) -> pd.Series:
    """Calculate Tweezer Bottoms (bullish) and Tweezer Tops (bearish) dual matching shadow reversal signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    o_prev = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(o)
        & ~np.isnan(h)
        & ~np.isnan(l_arr)
        & ~np.isnan(c)
        & ~np.isnan(o_prev)
        & ~np.isnan(h_prev)
        & ~np.isnan(l_prev)
        & ~np.isnan(c_prev)
    )

    matching_lows = np.abs(l_arr - l_prev) <= (rel_tol * np.maximum(l_arr, l_prev))
    matching_highs = np.abs(h - h_prev) <= (rel_tol * np.maximum(h, h_prev))

    tweezer_bottom = valid & (c_prev < o_prev) & (c > o) & matching_lows
    tweezer_top = valid & (c_prev > o_prev) & (c < o) & matching_highs

    condlist = [
        tweezer_bottom,
        tweezer_top,
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_couple_cs(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Couple Candlestick (Green-Green breakout or Red-Red breakdown)."""
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    o_prev = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(o_prev) & ~np.isnan(c_prev)

    # Bullish: Prev green with upper wick, Current green closing at high and exceeding prev high
    cond1_buy = (o_prev < c_prev) & (c_prev <= h_prev - 0.01)
    cond2_buy = (o < c) & (c >= h - 0.01) & (h > h_prev)
    bullish = valid & cond1_buy & cond2_buy

    # Bearish: Prev red with lower wick, Current red closing at low and breaking prev low
    cond1_sell = (o_prev > c_prev) & (c_prev >= l_prev + 0.01)
    cond2_sell = (o > c) & (c <= l_arr + 0.01) & (l_arr < l_prev)
    bearish = valid & cond1_sell & cond2_sell

    condlist = [bullish, bearish, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_fakey_pattern(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Fakey pattern (false 5-bar breakout with strong reversal close)."""
    lowest_5_prev = low.rolling(5).min().shift(1).to_numpy(dtype=float, na_value=np.nan)
    highest_5_prev = high.rolling(5).max().shift(1).to_numpy(dtype=float, na_value=np.nan)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(lowest_5_prev) & ~np.isnan(highest_5_prev)

    bullish_fakey = valid & (l_arr < lowest_5_prev) & (c > o)
    bearish_fakey = valid & (h > highest_5_prev) & (c < o)

    condlist = [bullish_fakey, bearish_fakey, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


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


def _calc_gap_up_down(open_p: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Calculate Opening Gap Up / Down signal."""
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h_prev) & ~np.isnan(l_prev)
    gap_up = valid & (o > h_prev)
    gap_down = valid & (o < l_prev)

    condlist = [gap_up, gap_down, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=open_p.index, dtype=str)


def _calc_body_size_expansion(open_p: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Candle Body Size Expansion (> 2.0x 20-period SMA body)."""
    body_abs = np.abs(close - open_p)
    body_sma20 = body_abs.rolling(20, min_periods=5).mean()

    b_arr = body_abs.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = body_sma20.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    o = open_p.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(b_arr) & ~np.isnan(sma_arr) & ~np.isnan(c) & ~np.isnan(o)
    is_expanded = valid & (b_arr > (2.0 * sma_arr))

    bull_expand = is_expanded & (c > o)
    bear_expand = is_expanded & (c < o)

    condlist = [bull_expand, bear_expand, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_wick_rejection(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Rejection Wick (> 2.0x body) signal."""
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(l_arr) & ~np.isnan(c)
    body_abs = np.abs(c - o)
    upwick = h - np.maximum(c, o)
    lowwick = np.minimum(c, o) - l_arr

    bull_wick = valid & (lowwick > 2.0 * body_abs) & (c > o)
    bear_wick = valid & (upwick > 2.0 * body_abs) & (c < o)

    condlist = [bull_wick, bear_wick, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_body_direction(open_p: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate 2-bar consecutive directional body agreement signal."""
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    o_prev = open_p.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(c) & ~np.isnan(o_prev) & ~np.isnan(c_prev)
    bull_dir = valid & (c > o) & (c_prev > o_prev)
    bear_dir = valid & (c < o) & (c_prev < o_prev)

    condlist = [bull_dir, bear_dir, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_close_strength(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Close Strength relative to bar midpoint and open."""
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(l_arr) & ~np.isnan(c)
    midpoint = (h + l_arr) / 2.0

    bull_str = valid & (c > midpoint) & (c > o)
    bear_str = valid & (c < midpoint) & (c < o)

    condlist = [bull_str, bear_str, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_price_rejection(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Price Rejection (sweep previous extreme but close reversed)."""
    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(l_arr) & ~np.isnan(c) & ~np.isnan(h_prev)
    bull_rej = valid & (l_arr < l_prev) & (c > o)
    bear_rej = valid & (h > h_prev) & (c < o)

    condlist = [bull_rej, bear_rej, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_break_retest(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Break & Retest of 10-period extremes."""
    high_10 = high.rolling(10).max().shift(2).to_numpy(dtype=float, na_value=np.nan)
    low_10 = low.rolling(10).min().shift(2).to_numpy(dtype=float, na_value=np.nan)

    c = close.to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(c_prev) & ~np.isnan(high_10) & ~np.isnan(low_10)
    bull_retest = valid & (c_prev > high_10) & (c < c_prev) & (c >= high_10)
    bear_retest = valid & (c_prev < low_10) & (c > c_prev) & (c <= low_10)

    condlist = [bull_retest, bear_retest, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_trend_exhaustion(close: pd.Series) -> pd.Series:
    """Calculate Trend Exhaustion (counter-trend reaction after 3-bar directional move)."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c2 = close.shift(2).to_numpy(dtype=float, na_value=np.nan)
    c3 = close.shift(3).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(c1) & ~np.isnan(c2) & ~np.isnan(c3)
    bull_exhaust = valid & (c > c1) & (c1 < c2) & (c2 < c3)
    bear_exhaust = valid & (c < c1) & (c1 > c2) & (c2 > c3)

    condlist = [bull_exhaust, bear_exhaust, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_final_push(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Final Push (higher/lower close on diminishing volume)."""
    c = close.to_numpy(dtype=float, na_value=np.nan)
    c1 = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c2 = close.shift(2).to_numpy(dtype=float, na_value=np.nan)

    v = volume.to_numpy(dtype=float, na_value=np.nan)
    v1 = volume.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(c1) & ~np.isnan(c2) & ~np.isnan(v) & ~np.isnan(v1)
    bull_push = valid & (c > c1) & (c1 > c2) & (v < v1)
    bear_push = valid & (c < c1) & (c1 < c2) & (v < v1)

    condlist = [bull_push, bear_push, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


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


def _calc_thrust_bar(
    open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate Thrust Bar strong directional expansion candle signal."""
    n = len(open_p)
    if n == 0:
        return pd.Series(dtype=str, index=open_p.index)

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    body_abs = np.abs(close - open_p)
    body_sma20 = body_abs.rolling(20, min_periods=5).mean().to_numpy(dtype=float, na_value=np.nan)
    b_arr = body_abs.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(o)
        & ~np.isnan(h)
        & ~np.isnan(l_arr)
        & ~np.isnan(c)
        & ~np.isnan(b_arr)
        & ~np.isnan(body_sma20)
    )
    rng = np.where(valid, h - l_arr, 0.0)

    is_thrust = valid & (rng > 0.0) & (b_arr >= 0.75 * rng) & (b_arr >= 1.8 * body_sma20)
    bull_thrust = is_thrust & (c > o)
    bear_thrust = is_thrust & (c < o)

    condlist = [bull_thrust, bear_thrust, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_narrow_range_7_breakout(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Narrow Range 7 (NR7) volatility compression breakout signal."""
    n = len(close)
    if n == 0:
        return pd.Series(dtype=str, index=close.index)

    rng = high - low
    min_rng_7 = rng.rolling(7).min()

    is_nr7_prev = (rng.shift(1) <= min_rng_7.shift(1)).to_numpy(dtype=bool)
    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(h_prev) & ~np.isnan(l_prev)

    bull_break = valid & is_nr7_prev & (c > h_prev)
    bear_break = valid & is_nr7_prev & (c < l_prev)

    condlist = [bull_break, bear_break, valid]
    choicelist = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_wide_range_reversal(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Wide Range Reversal bar (exhaustion spike with strong wick rejection)."""
    n = len(close)
    if n == 0:
        return pd.Series(dtype=str, index=close.index)

    h = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    rng_series = high - low
    rng_sma20 = rng_series.rolling(20, min_periods=5).mean().to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(h) & ~np.isnan(l_arr) & ~np.isnan(c) & ~np.isnan(rng_sma20)
    rng = np.where(valid, h - l_arr, 0.0)

    is_wide = valid & (rng > 0.0) & (rng >= 2.5 * rng_sma20)
    bull_rev = is_wide & (c >= (l_arr + 0.70 * rng))
    bear_rev = is_wide & (c <= (l_arr + 0.30 * rng))

    condlist = [bull_rev, bear_rev, valid]
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


def generate_candlestick_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Generate all 38 candlestick and price action signals from OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
    show_progress : bool, default False
        Whether to display a real-time progress bar for this signal group.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 38 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)

    with GroupProgressBar(
        "Candlestick Signals", total=len(CANDLESTICK_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if len(df_norm) == 0:
            pbar.update(len(CANDLESTICK_SIGNAL_COLUMNS))
            return pd.DataFrame(
                {col: pd.Series(dtype=str, index=df.index) for col in CANDLESTICK_SIGNAL_COLUMNS},
                index=df.index,
            )

        open_p = df_norm["open"]
        high = df_norm["high"]
        low = df_norm["low"]
        close = df_norm["close"]
        volume = df_norm["volume"]

        signals = pd.DataFrame(index=df_norm.index)

        # 1. Bullish and Bearish Engulfing (1)
        signals["cdl_engulfing_signal"] = _calc_engulfing(open_p, close)
        pbar.update(1)

        # 2. Hammer, Inverted Hammer, Shooting Star, Hanging Man (1)
        signals["cdl_hammer_star_signal"] = _calc_hammer_star(open_p, high, low, close)
        pbar.update(1)

        # 3. Pinbar price rejection (1)
        signals["cdl_pinbar_signal"] = _calc_pinbar(open_p, high, low, close)
        pbar.update(1)

        # 4. Marubozu strong momentum directional candle (1)
        signals["cdl_marubozu_signal"] = _calc_marubozu(open_p, high, low, close)
        pbar.update(1)

        # 5. Bullish and Bearish Harami (1)
        signals["cdl_harami_signal"] = _calc_harami(open_p, close)
        pbar.update(1)

        # 6. Inside Bar breakout (1)
        signals["cdl_inside_bar_breakout_signal"] = _calc_inside_bar(open_p, high, low, close)
        pbar.update(1)

        # 7. Outside Bar breakout (1)
        signals["cdl_outside_bar_signal"] = _calc_outside_bar(open_p, high, low, close)
        pbar.update(1)

        # 8. Doji reversal (Dragonfly / Gravestone) (1)
        signals["cdl_doji_reversal_signal"] = _calc_doji_reversal(open_p, high, low, close)
        pbar.update(1)

        # 9. Three White Soldiers / Three Black Crows (1)
        signals["cdl_three_soldiers_crows_signal"] = _calc_three_soldiers_crows(open_p, close)
        pbar.update(1)

        # 10. Three consecutive directional bars (1)
        signals["cdl_consecutive_3_signal"] = _calc_consecutive_directional(open_p, close, window=3)
        pbar.update(1)

        # 11. Five consecutive directional bars (1)
        signals["cdl_consecutive_5_signal"] = _calc_consecutive_directional(open_p, close, window=5)
        pbar.update(1)

        # 12. Morning Star and Evening Star (1)
        signals["cdl_morning_evening_star_signal"] = _calc_morning_evening_star(open_p, close)
        pbar.update(1)

        # 13. Piercing Line and Dark Cloud Cover (1)
        signals["cdl_piercing_darkcloud_signal"] = _calc_piercing_darkcloud(open_p, close)
        pbar.update(1)

        # 14. Tweezer Tops and Bottoms (1)
        signals["cdl_tweezer_tops_bottoms_signal"] = _calc_tweezer_tops_bottoms(
            open_p, high, low, close
        )
        pbar.update(1)

        # 15. Couple Candlestick Pattern (1)
        signals["cdl_couple_cs_signal"] = _calc_couple_cs(open_p, high, low, close)
        pbar.update(1)

        # 16. Fakey Pattern (1)
        signals["cdl_fakey_pattern_signal"] = _calc_fakey_pattern(open_p, high, low, close)
        pbar.update(1)

        # 17. Liquidity Sweep (1)
        signals["cdl_liquidity_sweep_signal"] = _calc_liquidity_sweep(high, low, close)
        pbar.update(1)

        # 18. Equal High / Low Sweep (1)
        signals["cdl_equal_high_low_sweep_signal"] = _calc_equal_high_low_sweep(high, low, close)
        pbar.update(1)

        # 19. Gap Up / Down (1)
        signals["cdl_gap_up_down_signal"] = _calc_gap_up_down(open_p, high, low)
        pbar.update(1)

        # 20. Body Size Expansion (1)
        signals["cdl_body_size_expansion_signal"] = _calc_body_size_expansion(open_p, close)
        pbar.update(1)

        # 21. Wick Rejection (1)
        signals["cdl_wick_rejection_signal"] = _calc_wick_rejection(open_p, high, low, close)
        pbar.update(1)

        # 22. Body Direction Agreement (1)
        signals["cdl_body_direction_signal"] = _calc_body_direction(open_p, close)
        pbar.update(1)

        # 23. Close Strength (1)
        signals["cdl_close_strength_signal"] = _calc_close_strength(open_p, high, low, close)
        pbar.update(1)

        # 24. Price Rejection (1)
        signals["cdl_price_rejection_signal"] = _calc_price_rejection(open_p, high, low, close)
        pbar.update(1)

        # 25. Break and Retest (1)
        signals["cdl_break_retest_signal"] = _calc_break_retest(high, low, close)
        pbar.update(1)

        # 26. Trend Exhaustion (1)
        signals["cdl_trend_exhaustion_signal"] = _calc_trend_exhaustion(close)
        pbar.update(1)

        # 27. Final Push (1)
        signals["cdl_final_push_signal"] = _calc_final_push(close, volume)
        pbar.update(1)

        # 28. Bullish FVG Mitigation (1)
        signals["cdl_fvg_bullish_mitigation_signal"] = _calc_fvg_bullish_mitigation(
            open_p, high, low, close
        )
        pbar.update(1)

        # 29. Bearish FVG Mitigation (1)
        signals["cdl_fvg_bearish_mitigation_signal"] = _calc_fvg_bearish_mitigation(
            open_p, high, low, close
        )
        pbar.update(1)

        # 30. Order Block Retest (1)
        signals["cdl_order_block_retest_signal"] = _calc_order_block_retest(
            open_p, high, low, close
        )
        pbar.update(1)

        # 31. Break of Structure (1)
        signals["cdl_break_of_structure_signal"] = _calc_break_of_structure(high, low, close)
        pbar.update(1)

        # 32. Change of Character (1)
        signals["cdl_change_of_character_signal"] = _calc_change_of_character(high, low, close)
        pbar.update(1)

        # 33. Judas Swing (1)
        signals["cdl_judas_swing_signal"] = _calc_judas_swing(open_p, high, low, close)
        pbar.update(1)

        # 34. Inducement Sweep (1)
        signals["cdl_inducement_sweep_signal"] = _calc_inducement_sweep(open_p, high, low, close)
        pbar.update(1)

        # 35. Thrust Bar (1)
        signals["cdl_thrust_bar_signal"] = _calc_thrust_bar(open_p, high, low, close)
        pbar.update(1)

        # 36. Narrow Range 7 Breakout (1)
        signals["cdl_narrow_range_7_breakout_signal"] = _calc_narrow_range_7_breakout(
            high, low, close
        )
        pbar.update(1)

        # 37. Wide Range Reversal (1)
        signals["cdl_wide_range_reversal_signal"] = _calc_wide_range_reversal(high, low, close)
        pbar.update(1)

        # 38. Prior Day High / Low Sweep (1)
        signals["cdl_pdh_pdl_sweep_signal"] = _calc_pdh_pdl_sweep(df_norm, open_p, high, low, close)
        pbar.update(1)

        # Ensure all columns are present, filled with NONE, and match index
        for col in CANDLESTICK_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[CANDLESTICK_SIGNAL_COLUMNS]
