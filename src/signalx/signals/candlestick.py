from __future__ import annotations

import numpy as np
import pandas as pd

from signalx.constants import SignalState
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


def generate_candlestick_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all 14 candlestick and price action signals from OHLCV dataframe."""
    df_norm = normalize_ohlcv(df)

    if len(df_norm) == 0:
        return pd.DataFrame(
            {col: pd.Series(dtype=str, index=df.index) for col in CANDLESTICK_SIGNAL_COLUMNS},
            index=df.index,
        )

    open_p = df_norm["open"]
    high = df_norm["high"]
    low = df_norm["low"]
    close = df_norm["close"]

    signals = pd.DataFrame(index=df_norm.index)

    # 1. Bullish and Bearish Engulfing
    signals["cdl_engulfing_signal"] = _calc_engulfing(open_p, close)

    # 2. Hammer, Inverted Hammer, Shooting Star, Hanging Man
    signals["cdl_hammer_star_signal"] = _calc_hammer_star(open_p, high, low, close)

    # 3. Pinbar price rejection
    signals["cdl_pinbar_signal"] = _calc_pinbar(open_p, high, low, close)

    # 4. Marubozu strong momentum directional candle
    signals["cdl_marubozu_signal"] = _calc_marubozu(open_p, high, low, close)

    # 5. Bullish and Bearish Harami
    signals["cdl_harami_signal"] = _calc_harami(open_p, close)

    # 6. Inside Bar breakout
    signals["cdl_inside_bar_breakout_signal"] = _calc_inside_bar(open_p, high, low, close)

    # 7. Outside Bar breakout
    signals["cdl_outside_bar_signal"] = _calc_outside_bar(open_p, high, low, close)

    # 8. Doji reversal (Dragonfly / Gravestone)
    signals["cdl_doji_reversal_signal"] = _calc_doji_reversal(open_p, high, low, close)

    # 9. Three White Soldiers / Three Black Crows
    signals["cdl_three_soldiers_crows_signal"] = _calc_three_soldiers_crows(open_p, close)

    # 10. Three consecutive directional bars
    signals["cdl_consecutive_3_signal"] = _calc_consecutive_directional(open_p, close, window=3)

    # 11. Five consecutive directional bars
    signals["cdl_consecutive_5_signal"] = _calc_consecutive_directional(open_p, close, window=5)

    # 12. Morning Star and Evening Star
    signals["cdl_morning_evening_star_signal"] = _calc_morning_evening_star(open_p, close)

    # 13. Piercing Line and Dark Cloud Cover
    signals["cdl_piercing_darkcloud_signal"] = _calc_piercing_darkcloud(open_p, close)

    # 14. Tweezer Tops and Bottoms
    signals["cdl_tweezer_tops_bottoms_signal"] = _calc_tweezer_tops_bottoms(
        open_p, high, low, close
    )

    # Ensure all columns are present, filled with NONE, and match index
    for col in CANDLESTICK_SIGNAL_COLUMNS:
        if col not in signals.columns:
            signals[col] = SignalState.NONE
        else:
            signals[col] = signals[col].fillna(SignalState.NONE)

    return signals[CANDLESTICK_SIGNAL_COLUMNS]
