from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pandas_ta as pta

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
from signalx.utils import normalize_ohlcv

STATISTICAL_SIGNAL_COLUMNS = [
    "stat_price_zscore_10_signal",
    "stat_price_zscore_20_signal",
    "stat_price_zscore_50_signal",
    "stat_price_zscore_100_signal",
    "stat_return_zscore_20_signal",
    "stat_ker_trend_filter_10_signal",
    "stat_ker_trend_filter_20_signal",
    "stat_chop_regime_14_signal",
    "stat_rolling_quantile_extremes_20_signal",
    "stat_linreg_slope_14_signal",
    "stat_linreg_price_cross_30_signal",
    "stat_ma_stretch_zscore_20_signal",
    "stat_hurst_proxy_signal",
    "stat_range_mid_reversion_10_signal",
    "stat_price_acceleration_signal",
    "stat_mean_distance_5pct_signal",
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
    bullish = valid_mask & (fast_arr >= slow_arr)
    bearish = valid_mask & (fast_arr < slow_arr)

    condlist = [
        cross_up,
        cross_down,
        bullish,
        bearish,
        valid_mask,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
        SignalState.HOLD,
        SignalState.NONE,
    ]
    res_arr = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res_arr, index=fast.index, dtype=str)


def _calc_zscore(
    s: pd.Series,
    window: int,
    lower_bound: float = -2.0,
    upper_bound: float = 2.0,
) -> pd.Series:
    """Calculate rolling Z-score signal based on lower and upper standard deviation thresholds."""
    n = len(s)
    if n < window or window <= 0:
        return pd.Series(SignalState.NONE, index=s.index, dtype=str)

    mean = s.rolling(window=window).mean()
    std = s.rolling(window=window).std()

    s_arr = s.to_numpy(dtype=float, na_value=np.nan)
    m_arr = mean.to_numpy(dtype=float, na_value=np.nan)
    sd_arr = std.to_numpy(dtype=float, na_value=np.nan)

    valid_window = ~np.isnan(s_arr) & ~np.isnan(m_arr)
    z = np.zeros(n, dtype=float)
    non_zero_sd = valid_window & ~np.isnan(sd_arr) & (sd_arr > 0)
    z[non_zero_sd] = (s_arr[non_zero_sd] - m_arr[non_zero_sd]) / sd_arr[non_zero_sd]

    condlist = [
        valid_window & (z < lower_bound),
        valid_window & (z > upper_bound),
        valid_window,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=s.index, dtype=str)


def _calc_ker(
    close: pd.Series,
    length: int = 10,
    threshold: float = 0.6,
) -> pd.Series:
    """Calculate Kaufman Efficiency Ratio (KER) trend filter signal."""
    n = len(close)
    if n <= length or length <= 0:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    net_change = close - close.shift(length)
    vol = (close - close.shift(1)).abs()
    sum_vol = vol.rolling(length).sum()

    nc_arr = net_change.to_numpy(dtype=float, na_value=np.nan)
    sv_arr = sum_vol.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(nc_arr) & ~np.isnan(sv_arr) & (sv_arr > 0)
    ker = np.zeros(n, dtype=float)
    np.divide(np.abs(nc_arr), sv_arr, out=ker, where=valid)

    condlist = [
        valid & (ker > threshold) & (nc_arr > 0),
        valid & (ker > threshold) & (nc_arr < 0),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_chop_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Calculate Choppiness Index trending (<38.2) vs consolidating (>61.8) regime."""
    n = len(close)
    if n <= length or length <= 0:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    # Check for completely flat data to avoid warnings
    if (high == low).all():
        valid_mask = pd.Series(True, index=close.index)
        valid_mask.iloc[:length] = False
        return pd.Series(
            np.where(valid_mask, SignalState.HOLD, SignalState.NONE),
            index=close.index,
            dtype=str,
        )

    # Attempt pandas_ta implementation first with warning suppression
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pta.chop(high=high, low=low, close=close, length=length)
        if res is not None and isinstance(res, pd.Series) and not res.empty:
            chop_arr = res.to_numpy(dtype=float, na_value=np.nan)
            valid = ~np.isnan(chop_arr) & ~np.isinf(chop_arr)
            condlist = [
                valid & (chop_arr < 38.2),
                valid & (chop_arr > 61.8),
                valid,
            ]
            choicelist = [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.HOLD,
            ]
            out = np.select(condlist, choicelist, default=SignalState.NONE)
            return pd.Series(out, index=close.index, dtype=str)
    except Exception:
        pass

    # Native Choppiness Index fallback
    prev_c = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_c).abs()
    tr3 = (low - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    sum_tr = tr.rolling(length).sum()
    max_h = high.rolling(length).max()
    min_l = low.rolling(length).min()
    rng = max_h - min_l

    str_arr = sum_tr.to_numpy(dtype=float, na_value=np.nan)
    rng_arr = rng.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(str_arr) & ~np.isnan(rng_arr) & (rng_arr > 0) & (str_arr > 0)
    chop = np.full(n, np.nan, dtype=float)

    if valid.any():
        ratio = str_arr[valid] / rng_arr[valid]
        positive_ratio = ratio > 0
        valid_idx = np.where(valid)[0][positive_ratio]
        if len(valid_idx) > 0:
            chop[valid_idx] = 100.0 * np.log10(ratio[positive_ratio]) / np.log10(float(length))

    valid_final = ~np.isnan(chop) & ~np.isinf(chop)

    condlist = [
        valid_final & (chop < 38.2),
        valid_final & (chop > 61.8),
        valid_final,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    out = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(out, index=close.index, dtype=str)


def _calc_rolling_quantile_extremes(
    close: pd.Series,
    window: int = 20,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> pd.Series:
    """Calculate rolling quantile extreme triggers (5th percentile buy, 95th percentile sell)."""
    n = len(close)
    if n < window or window <= 0:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    q_low = close.rolling(window).quantile(lower_q)
    q_high = close.rolling(window).quantile(upper_q)

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    ql_arr = q_low.to_numpy(dtype=float, na_value=np.nan)
    qh_arr = q_high.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c_arr) & ~np.isnan(ql_arr) & ~np.isnan(qh_arr)

    condlist = [
        valid & (c_arr <= ql_arr),
        valid & (c_arr >= qh_arr),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_linreg_slope(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate rolling linear regression slope direction."""
    n = len(close)
    if n < length or length <= 1:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    # Closed-form weights for linear regression slope: w = (x - mean(x)) / sum((x - mean(x))^2)
    x = np.arange(length, dtype=float)
    x_mean = (length - 1) / 2.0
    x_diff = x - x_mean
    s_xx = np.sum(x_diff**2)
    weights = x_diff / s_xx

    # Vectorized 1D convolution over close price series
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    slopes = np.full(n, np.nan, dtype=float)

    # Compute rolling slope
    for i in range(length - 1, n):
        window = c_arr[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            slopes[i] = np.dot(window, weights)

    valid = ~np.isnan(slopes)
    condlist = [
        valid & (slopes > 0),
        valid & (slopes < 0),
        valid,
    ]
    choicelist = [
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
    ]
    res = np.select(condlist, choicelist, default=SignalState.NONE)
    return pd.Series(res, index=close.index, dtype=str)


def _calc_linreg_price_cross(close: pd.Series, length: int = 30) -> pd.Series:
    """Calculate 30-period linear regression trendline price crossover."""
    n = len(close)
    if n < length or length <= 1:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    x = np.arange(length, dtype=float)
    x_mean = (length - 1) / 2.0
    x_diff = x - x_mean
    s_xx = np.sum(x_diff**2)
    weights = x_diff / s_xx

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    linreg_line = np.full(n, np.nan, dtype=float)

    for i in range(length - 1, n):
        window = c_arr[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            slope = np.dot(window, weights)
            mean_y = np.mean(window)
            # Forecast endpoint at x = length - 1
            linreg_line[i] = mean_y + slope * x_mean

    linreg_s = pd.Series(linreg_line, index=close.index)
    return _crossover_signal(close, linreg_s)


def _calc_ma_stretch(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Moving Average Stretch (Z-Score > 2 or < -2)."""
    mean = close.rolling(window=window, min_periods=window // 2).mean()
    std = close.rolling(window=window, min_periods=window // 2).std()
    z = (close - mean) / std.replace(0, np.nan)

    z_arr = z.to_numpy(dtype=float, na_value=np.nan)
    valid = ~np.isnan(z_arr)
    bull = valid & (z_arr < -2.0)
    bear = valid & (z_arr > 2.0)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_hurst_proxy(close: pd.Series, window: int = 50) -> pd.Series:
    """Calculate rolling Hurst Exponent proxy (< 0.5 mean-reversion, > 0.5 trending)."""
    diff1 = close.diff(1)
    diff2 = close.diff(2)

    var1 = diff1.rolling(window, min_periods=window // 2).var()
    var2 = diff2.rolling(window, min_periods=window // 2).var()

    ratio = var2 / (2.0 * var1 + 1e-9).replace(0, np.nan)
    hurst = 0.5 * np.log2(ratio.clip(lower=1e-4))

    h_arr = hurst.to_numpy(dtype=float, na_value=np.nan)
    valid = ~np.isnan(h_arr)
    bull = valid & (h_arr < 0.5)
    bear = valid & (h_arr > 0.5)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_range_mid_reversion(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10
) -> pd.Series:
    """Calculate Range Midpoint Reversion signal."""
    h_max = high.rolling(window, min_periods=2).max().to_numpy(dtype=float, na_value=np.nan)
    l_min = low.rolling(window, min_periods=2).min().to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(h_max) & ~np.isnan(l_min)
    mid = (h_max + l_min) / 2.0

    bull = valid & (c < mid)
    bear = valid & (c > mid)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_price_acceleration(close: pd.Series) -> pd.Series:
    """Calculate 1-bar Price Change Acceleration."""
    delta = close.diff().to_numpy(dtype=float, na_value=np.nan)
    delta1 = close.diff().shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(delta) & ~np.isnan(delta1)
    bull = valid & (delta > delta1)
    bear = valid & (delta < delta1)
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def _calc_mean_distance(close: pd.Series, window: int = 20, pct: float = 0.05) -> pd.Series:
    """Calculate Mean Distance (> 5% deviation from rolling SMA)."""
    sma = (
        close.rolling(window, min_periods=window // 2).mean().to_numpy(dtype=float, na_value=np.nan)
    )
    c = close.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c) & ~np.isnan(sma)
    bull = valid & (c < (sma * (1.0 - pct)))
    bear = valid & (c > (sma * (1.0 + pct)))
    hold = valid & ~bull & ~bear

    conds = [bull, bear, hold]
    choices = [SignalState.BUY, SignalState.SELL, SignalState.HOLD]
    return pd.Series(
        np.select(conds, choices, default=SignalState.NONE), index=close.index, dtype=str
    )


def generate_statistical_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Generate all 16 standardized statistical & mean reversion signals from normalized OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
    show_progress : bool, default False
        Whether to display a real-time progress bar for this signal group.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 16 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)
    signals = pd.DataFrame(index=df_norm.index)

    with GroupProgressBar(
        "Statistical Signals", total=len(STATISTICAL_SIGNAL_COLUMNS), enabled=show_progress
    ) as pbar:
        if df_norm.empty:
            for col in STATISTICAL_SIGNAL_COLUMNS:
                signals[col] = pd.Series(dtype=str)
            pbar.update(len(STATISTICAL_SIGNAL_COLUMNS))
            return signals

        close = df_norm["close"]
        high = df_norm["high"]
        low = df_norm["low"]

        # 1. Price Z-Scores (10, 20, 50, 100) (4)
        signals["stat_price_zscore_10_signal"] = _calc_zscore(close, window=10)
        signals["stat_price_zscore_20_signal"] = _calc_zscore(close, window=20)
        signals["stat_price_zscore_50_signal"] = _calc_zscore(close, window=50)
        signals["stat_price_zscore_100_signal"] = _calc_zscore(close, window=100)
        pbar.update(4)

        # 2. Return Z-Score (20) (1)
        returns = close.pct_change()
        signals["stat_return_zscore_20_signal"] = _calc_zscore(returns, window=20)
        pbar.update(1)

        # 3. Kaufman Efficiency Ratio Trend Filters (10, 20) (2)
        signals["stat_ker_trend_filter_10_signal"] = _calc_ker(close, length=10, threshold=0.6)
        signals["stat_ker_trend_filter_20_signal"] = _calc_ker(close, length=20, threshold=0.6)
        pbar.update(2)

        # 4. Choppiness Index Regime (14) (1)
        signals["stat_chop_regime_14_signal"] = _calc_chop_regime(
            high=high, low=low, close=close, length=14
        )
        pbar.update(1)

        # 5. Rolling Quantile Extremes (20) (1)
        signals["stat_rolling_quantile_extremes_20_signal"] = _calc_rolling_quantile_extremes(
            close, window=20, lower_q=0.05, upper_q=0.95
        )
        pbar.update(1)

        # 6. Linear Regression Slope (14) (1)
        signals["stat_linreg_slope_14_signal"] = _calc_linreg_slope(close, length=14)
        pbar.update(1)

        # 7. Linear Regression Price Crossover (30) (1)
        signals["stat_linreg_price_cross_30_signal"] = _calc_linreg_price_cross(close, length=30)
        pbar.update(1)

        # 8. MA Stretch Z-Score (20) (1)
        signals["stat_ma_stretch_zscore_20_signal"] = _calc_ma_stretch(close, window=20)
        pbar.update(1)

        # 9. Hurst Exponent Proxy (1)
        signals["stat_hurst_proxy_signal"] = _calc_hurst_proxy(close, window=50)
        pbar.update(1)

        # 10. Range Midpoint Reversion (10) (1)
        signals["stat_range_mid_reversion_10_signal"] = _calc_range_mid_reversion(
            high, low, close, window=10
        )
        pbar.update(1)

        # 11. Price Acceleration (1)
        signals["stat_price_acceleration_signal"] = _calc_price_acceleration(close)
        pbar.update(1)

        # 12. Mean Distance 5% (1)
        signals["stat_mean_distance_5pct_signal"] = _calc_mean_distance(close, window=20, pct=0.05)
        pbar.update(1)

        # Ensure all columns are present, filled with NONE, and matching index
        for col in STATISTICAL_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[STATISTICAL_SIGNAL_COLUMNS]
