from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from signalx.constants import SignalState
from signalx.progress import GroupProgressBar
from signalx.signals.session_helper import extract_session_context
from signalx.utils import normalize_ohlcv

MEAN_REVERSION_SIGNAL_COLUMNS = [
    "mr_connors_rsi2_regime_signal",
    "mr_ou_process_spread_reversion_signal",
    "mr_vwap_distance_zscore_signal",
    "mr_bb_pct_b_hook_reversion_signal",
    "mr_keltner_atr_stretch_reentry_signal",
    "mr_kurtosis_fat_tail_exhaustion_signal",
    "mr_dual_ma_disparity_index_signal",
    "mr_linreg_residual_zscore_signal",
    "mr_wr_cci_double_oversold_signal",
    "mr_session_range_fade_signal",
    "mr_volume_climax_absorption_reversion_signal",
    "mr_multi_period_stretch_consensus_signal",
    "mr_lehmann_short_term_reversal_signal",
    "mr_lo_mackinlay_variance_ratio_signal",
    "mr_ehlers_roofing_filter_reversion_signal",
    "mr_amihud_liquidity_exhaustion_signal",
    "mr_bb_w_bottom_m_top_signal",
]


def _calc_connors_rsi2_regime(close: pd.Series) -> pd.Series:
    """Larry Connors RSI(2) Trend-Filtered Reversion.

    Buy: Close > SMA(200) and RSI(2) < 10.0
    Sell: Close < SMA(200) and RSI(2) > 90.0
    """
    n = len(close)
    if n < 2:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    sma200 = close.rolling(200, min_periods=5).mean()
    rsi2 = ta.momentum.RSIIndicator(close, window=2, fillna=False).rsi()

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = sma200.to_numpy(dtype=float, na_value=np.nan)
    rsi_arr = rsi2.to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(c_arr) & ~np.isnan(sma_arr) & ~np.isnan(rsi_arr)
    buy = valid & (c_arr > sma_arr) & (rsi_arr < 10.0)
    sell = valid & (c_arr < sma_arr) & (rsi_arr > 90.0)
    hold = valid & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_ou_process_spread_reversion(close: pd.Series, window: int = 30) -> pd.Series:
    """Ornstein-Uhlenbeck (OU) Equilibrium Spread Reversion.

    Fits dx_t = theta * (mu - x_t) * dt + sigma * dW_t over rolling window.
    Buy: Standardized Deviation < -2.0 and theta > 0
    Sell: Standardized Deviation > +2.0 and theta > 0
    """
    n = len(close)
    if n < window or window <= 1:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    delta_x = close.diff(1)
    x_lag = close.shift(1)

    cov = delta_x.rolling(window).cov(x_lag)
    var = x_lag.rolling(window).var()
    mean_delta = delta_x.rolling(window).mean()
    mean_lag = x_lag.rolling(window).mean()

    cov_arr = cov.to_numpy(dtype=float, na_value=np.nan)
    var_arr = var.to_numpy(dtype=float, na_value=np.nan)
    md_arr = mean_delta.to_numpy(dtype=float, na_value=np.nan)
    ml_arr = mean_lag.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)

    valid_stat = ~np.isnan(cov_arr) & ~np.isnan(var_arr) & (var_arr > 1e-12)

    b = np.zeros(n, dtype=float)
    np.divide(cov_arr, var_arr, out=b, where=valid_stat)

    theta = -b
    reverting = valid_stat & (theta > 1e-8)

    a = md_arr - b * ml_arr

    mu = np.zeros(n, dtype=float)
    np.divide(a, theta, out=mu, where=reverting)

    rolling_std = close.rolling(window).std().to_numpy(dtype=float, na_value=np.nan)
    valid_std = ~np.isnan(rolling_std) & (rolling_std > 1e-8)

    valid_dev = reverting & valid_std
    dev = np.zeros(n, dtype=float)
    np.divide(c_arr - mu, rolling_std, out=dev, where=valid_dev)

    buy = valid_dev & (dev < -2.0)
    sell = valid_dev & (dev > 2.0)
    hold = valid_stat & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_vwap_distance_zscore(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling VWAP Distance Z-Score Stretch.

    Buy: Distance Z-Score < -2.0
    Sell: Distance Z-Score > +2.0
    """
    n = len(close)
    if n < window or window <= 1:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    tp = (high + low + close) / 3.0
    pv = tp * volume
    sum_pv = pv.rolling(window, min_periods=window).sum()
    sum_vol = volume.rolling(window, min_periods=window).sum()

    pv_arr = sum_pv.to_numpy(dtype=float, na_value=np.nan)
    vol_arr = sum_vol.to_numpy(dtype=float, na_value=np.nan)

    valid_vol = ~np.isnan(pv_arr) & ~np.isnan(vol_arr) & (vol_arr > 1e-12)
    vwap = np.full(n, np.nan, dtype=float)
    np.divide(pv_arr, vol_arr, out=vwap, where=valid_vol)

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    dist = np.full(n, np.nan, dtype=float)
    valid_vwap = valid_vol & ~np.isnan(c_arr)
    dist[valid_vwap] = c_arr[valid_vwap] - vwap[valid_vwap]

    dist_s = pd.Series(dist, index=close.index)
    dist_mean = (
        dist_s.rolling(window, min_periods=window).mean().to_numpy(dtype=float, na_value=np.nan)
    )
    dist_std = (
        dist_s.rolling(window, min_periods=window).std().to_numpy(dtype=float, na_value=np.nan)
    )

    valid_z = ~np.isnan(dist) & ~np.isnan(dist_mean) & ~np.isnan(dist_std) & (dist_std > 1e-12)
    z = np.zeros(n, dtype=float)
    np.divide(dist - dist_mean, dist_std, out=z, where=valid_z)

    buy = valid_z & (z < -2.0)
    sell = valid_z & (z > 2.0)
    hold = valid_z & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_bb_pct_b_hook_reversion(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0,
    open_p: pd.Series | None = None,
) -> pd.Series:
    """Bollinger Bands %B Extreme Hook Reversion.

    Buy: %B_{t-1} < 0.0 and %B_t >= 0.0 with bullish close
    Sell: %B_{t-1} > 1.0 and %B_t <= 1.0 with bearish close
    """
    n = len(close)
    if n < length or length <= 1:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    sma = close.rolling(length, min_periods=length).mean()
    rolling_std = close.rolling(length, min_periods=length).std()

    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    band_width = upper - lower

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    lower_arr = lower.to_numpy(dtype=float, na_value=np.nan)
    bw_arr = band_width.to_numpy(dtype=float, na_value=np.nan)

    valid_bw = ~np.isnan(c_arr) & ~np.isnan(lower_arr) & ~np.isnan(bw_arr) & (bw_arr > 1e-12)
    pct_b = np.full(n, np.nan, dtype=float)
    np.divide(c_arr - lower_arr, bw_arr, out=pct_b, where=valid_bw)

    pct_b_prev = pd.Series(pct_b, index=close.index).shift(1).to_numpy(dtype=float, na_value=np.nan)
    valid_pair = valid_bw & ~np.isnan(pct_b_prev)

    if open_p is not None:
        o_arr = open_p.to_numpy(dtype=float, na_value=np.nan)
        bullish = c_arr > o_arr
        bearish = c_arr < o_arr
    else:
        c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
        bullish = c_arr > c_prev
        bearish = c_arr < c_prev

    buy = valid_pair & (pct_b_prev < 0.0) & (pct_b >= 0.0) & bullish
    sell = valid_pair & (pct_b_prev > 1.0) & (pct_b <= 1.0) & bearish
    hold = valid_bw & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_keltner_atr_stretch_reentry(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 3.0,
) -> pd.Series:
    """Keltner Channel 3-ATR Re-entry Reversal.

    Buy: Low_{t-1} < Lower_{t-1} and Close_t >= Lower_t
    Sell: High_{t-1} > Upper_{t-1} and Close_t <= Upper_t
    """
    n = len(close)
    if n < max(length, 14):
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    c_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - c_prev).abs()
    tr3 = (low - c_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=14, adjust=False).mean()
    ema = close.ewm(span=length, adjust=False).mean()

    lower = ema - mult * atr
    upper = ema + mult * atr

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    lower_arr = lower.to_numpy(dtype=float, na_value=np.nan)
    upper_arr = upper.to_numpy(dtype=float, na_value=np.nan)

    l_prev = low.shift(1).to_numpy(dtype=float, na_value=np.nan)
    h_prev = high.shift(1).to_numpy(dtype=float, na_value=np.nan)
    lower_prev = lower.shift(1).to_numpy(dtype=float, na_value=np.nan)
    upper_prev = upper.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(c_arr)
        & ~np.isnan(h_arr)
        & ~np.isnan(l_arr)
        & ~np.isnan(lower_arr)
        & ~np.isnan(upper_arr)
        & ~np.isnan(l_prev)
        & ~np.isnan(h_prev)
        & ~np.isnan(lower_prev)
        & ~np.isnan(upper_prev)
    )

    buy = valid & (l_prev < lower_prev) & (c_arr >= lower_arr)
    sell = valid & (h_prev > upper_prev) & (c_arr <= upper_arr)
    hold = valid & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_kurtosis_fat_tail_exhaustion(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling Kurtosis Fat-Tail Shock Reversal.

    Excess kurtosis > 3.0 on returns and |Z_ret| > 2.5 with hook.
    """
    n = len(close)
    if n < window or window <= 3:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    returns = close.pct_change()
    kurt = returns.rolling(window, min_periods=window).kurt()
    mean_r = returns.rolling(window, min_periods=window).mean()
    std_r = returns.rolling(window, min_periods=window).std()

    k_arr = kurt.to_numpy(dtype=float, na_value=np.nan)
    r_arr = returns.to_numpy(dtype=float, na_value=np.nan)
    mr_arr = mean_r.to_numpy(dtype=float, na_value=np.nan)
    sr_arr = std_r.to_numpy(dtype=float, na_value=np.nan)

    valid_std = (
        ~np.isnan(r_arr)
        & ~np.isnan(mr_arr)
        & ~np.isnan(sr_arr)
        & (sr_arr > 1e-12)
        & ~np.isnan(k_arr)
    )
    z_ret = np.zeros(n, dtype=float)
    np.divide(r_arr - mr_arr, sr_arr, out=z_ret, where=valid_std)

    z_prev = pd.Series(z_ret, index=close.index).shift(1).to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = valid_std & ~np.isnan(z_prev) & ~np.isnan(c_prev)

    hook_up = (z_ret > z_prev) | (c_arr > c_prev)
    hook_down = (z_ret < z_prev) | (c_arr < c_prev)

    buy = valid & (k_arr > 3.0) & ((z_ret < -2.5) | (z_prev < -2.5)) & hook_up
    sell = valid & (k_arr > 3.0) & ((z_ret > 2.5) | (z_prev > 2.5)) & hook_down
    hold = valid & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_dual_ma_disparity_index(
    close: pd.Series,
    length: int = 20,
    threshold: float = 3.5,
    open_p: pd.Series | None = None,
) -> pd.Series:
    """Disparity Index Stretch Reversion.

    Buy: DI < -threshold with bullish candle
    Sell: DI > +threshold with bearish candle
    """
    n = len(close)
    if n < length or length <= 1:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    sma = close.rolling(length, min_periods=length).mean()
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = sma.to_numpy(dtype=float, na_value=np.nan)

    valid_sma = ~np.isnan(c_arr) & ~np.isnan(sma_arr) & (sma_arr > 1e-12)
    di = np.zeros(n, dtype=float)
    np.divide((c_arr - sma_arr) * 100.0, sma_arr, out=di, where=valid_sma)

    if open_p is not None:
        o_arr = open_p.to_numpy(dtype=float, na_value=np.nan)
        bullish = c_arr > o_arr
        bearish = c_arr < o_arr
    else:
        c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
        bullish = c_arr > c_prev
        bearish = c_arr < c_prev

    buy = valid_sma & (di < -threshold) & bullish
    sell = valid_sma & (di > threshold) & bearish
    hold = valid_sma & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_linreg_residual_zscore(close: pd.Series, length: int = 20) -> pd.Series:
    """Linear Regression Residuals Z-Score Reversion.

    Computes 20-bar linear regression trendline and evaluates Z-score of residuals.
    Buy: Residual Z-Score < -2.0
    Sell: Residual Z-Score > +2.0
    """
    n = len(close)
    if n < length or length <= 2:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    x = np.arange(length, dtype=float)
    x_mean = (length - 1) / 2.0
    x_diff = x - x_mean
    s_xx = np.sum(x_diff**2)
    weights = x_diff / s_xx

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    z_scores = np.full(n, np.nan, dtype=float)
    valid_mask = np.zeros(n, dtype=bool)

    for i in range(length - 1, n):
        window = c_arr[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            slope = np.dot(window, weights)
            mean_y = np.mean(window)
            y_hat = mean_y + slope * x_diff
            residuals = window - y_hat
            sse = np.sum(residuals**2)
            se = np.sqrt(sse / (length - 2))
            if se > 1e-12:
                z_scores[i] = residuals[-1] / se
                valid_mask[i] = True

    buy = valid_mask & (z_scores < -2.0)
    sell = valid_mask & (z_scores > 2.0)
    hold = valid_mask & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_wr_cci_double_oversold(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    wr_len: int = 14,
    cci_len: int = 20,
) -> pd.Series:
    """Williams %R and CCI Confluence Mean Reversion.

    Buy: Williams %R < -85 and CCI < -150 hooking upward
    Sell: Williams %R > -15 and CCI > +150 hooking downward
    """
    n = len(close)
    req_len = max(wr_len, cci_len)
    if n < req_len:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    highest_h = high.rolling(wr_len, min_periods=wr_len).max()
    lowest_l = low.rolling(wr_len, min_periods=wr_len).min()
    hl_diff = highest_h - lowest_l

    hh_arr = highest_h.to_numpy(dtype=float, na_value=np.nan)
    ll_arr = lowest_l.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    hld_arr = hl_diff.to_numpy(dtype=float, na_value=np.nan)

    valid_wr = ~np.isnan(c_arr) & ~np.isnan(hh_arr) & ~np.isnan(ll_arr) & (hld_arr > 1e-12)
    wr = np.zeros(n, dtype=float)
    np.divide(-100.0 * (hh_arr - c_arr), hld_arr, out=wr, where=valid_wr)

    tp = (high + low + close) / 3.0
    tp_sma = tp.rolling(cci_len, min_periods=cci_len).mean()
    tp_md = (tp - tp_sma).abs().rolling(cci_len, min_periods=cci_len).mean()

    tp_arr = tp.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = tp_sma.to_numpy(dtype=float, na_value=np.nan)
    md_arr = tp_md.to_numpy(dtype=float, na_value=np.nan)

    valid_cci = ~np.isnan(tp_arr) & ~np.isnan(sma_arr) & ~np.isnan(md_arr) & (md_arr > 1e-12)
    cci = np.zeros(n, dtype=float)
    np.divide(tp_arr - sma_arr, 0.015 * md_arr, out=cci, where=valid_cci)

    wr_prev = pd.Series(wr, index=close.index).shift(1).to_numpy(dtype=float, na_value=np.nan)
    cci_prev = pd.Series(cci, index=close.index).shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid = valid_wr & valid_cci & ~np.isnan(wr_prev) & ~np.isnan(cci_prev)

    hook_up = (wr > wr_prev) & (cci > cci_prev)
    hook_down = (wr < wr_prev) & (cci < cci_prev)

    buy = (
        valid
        & ((wr < -85.0) | (wr_prev < -85.0))
        & ((cci < -150.0) | (cci_prev < -150.0))
        & hook_up
    )
    sell = (
        valid
        & ((wr > -15.0) | (wr_prev > -15.0))
        & ((cci > 150.0) | (cci_prev > 150.0))
        & hook_down
    )
    hold = valid & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_session_range_fade(
    df: pd.DataFrame,
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Intraday Session Initial Balance (IB 30m) Fade Reversion.

    Detects false extensions beyond the 30-minute Initial Balance range.
    Buy: Wick below IB Low by > 0.2% but close back inside IB (Close > IB_low)
    Sell: Wick above IB High by > 0.2% but close back inside IB (Close < IB_high)
    """
    n = len(close)
    if n == 0:
        return pd.Series(dtype=str, index=close.index)

    ctx = extract_session_context(df)
    ib_mask = ctx.bar_in_session < 6
    ib_high_sub = high.where(ib_mask)
    ib_low_sub = low.where(ib_mask)

    ib_high = ib_high_sub.groupby(ctx.session_id, sort=False).transform("max")
    ib_low = ib_low_sub.groupby(ctx.session_id, sort=False).transform("min")

    c = close.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    lo = low.to_numpy(dtype=float, na_value=np.nan)
    ibh = ib_high.to_numpy(dtype=float, na_value=np.nan)
    ibl = ib_low.to_numpy(dtype=float, na_value=np.nan)

    is_bar_ge_6 = (ctx.bar_in_session >= 6).to_numpy(dtype=bool)

    valid = (
        is_bar_ge_6 & ~np.isnan(c) & ~np.isnan(h) & ~np.isnan(lo) & ~np.isnan(ibh) & ~np.isnan(ibl)
    )

    fade_low = valid & (lo < ibl * 0.998) & (c > ibl)
    fade_high = valid & (h > ibh * 1.002) & (c < ibh)
    hold = valid & ~fade_low & ~fade_high

    return pd.Series(
        np.select(
            [fade_low, fade_high, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_volume_climax_absorption_reversion(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Volume Climax Absorption at Extremes Reversion.

    Buy: Volume spike >= 3x SMA(Volume) at 20-bar Low with lower wick >= 40% and bullish close
    Sell: Volume spike >= 3x SMA(Volume) at 20-bar High with upper wick >= 40% and bearish close
    """
    n = len(close)
    if n < window or window <= 1:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    vol_sma = volume.rolling(window, min_periods=window).mean()
    roll_low = low.rolling(window, min_periods=window).min()
    roll_high = high.rolling(window, min_periods=window).max()

    o = open_p.to_numpy(dtype=float, na_value=np.nan)
    h = high.to_numpy(dtype=float, na_value=np.nan)
    lo = low.to_numpy(dtype=float, na_value=np.nan)
    c = close.to_numpy(dtype=float, na_value=np.nan)
    v = volume.to_numpy(dtype=float, na_value=np.nan)
    vsma = vol_sma.to_numpy(dtype=float, na_value=np.nan)
    rl = roll_low.to_numpy(dtype=float, na_value=np.nan)
    rh = roll_high.to_numpy(dtype=float, na_value=np.nan)

    valid_vol = ~np.isnan(v) & ~np.isnan(vsma) & (vsma > 1e-12)
    vol_spike = valid_vol & (v >= 3.0 * vsma)

    candle_range = h - lo
    valid_range = ~np.isnan(candle_range) & (candle_range > 1e-12)

    lower_wick = np.minimum(o, c) - lo
    upper_wick = h - np.maximum(o, c)

    lower_wick_pct = np.zeros(n, dtype=float)
    upper_wick_pct = np.zeros(n, dtype=float)
    np.divide(lower_wick, candle_range, out=lower_wick_pct, where=valid_range)
    np.divide(upper_wick, candle_range, out=upper_wick_pct, where=valid_range)

    valid = valid_vol & valid_range & ~np.isnan(rl) & ~np.isnan(rh)

    at_low = valid & (lo <= rl + 1e-8)
    at_high = valid & (h >= rh - 1e-8)

    buy = vol_spike & at_low & (lower_wick_pct >= 0.40) & (c >= o)
    sell = vol_spike & at_high & (upper_wick_pct >= 0.40) & (c <= o)
    hold = valid & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_multi_period_stretch_consensus(close: pd.Series) -> pd.Series:
    """Multi-Period Deviation Consensus Reversion.

    Buy: Price simultaneously < mean_k - 1.8 * std_k for k in {10, 20, 50}
    Sell: Price simultaneously > mean_k + 1.8 * std_k for k in {10, 20, 50}
    """
    n = len(close)
    if n < 50:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    m10 = close.rolling(10, min_periods=10).mean()
    s10 = close.rolling(10, min_periods=10).std()
    m20 = close.rolling(20, min_periods=20).mean()
    s20 = close.rolling(20, min_periods=20).std()
    m50 = close.rolling(50, min_periods=50).mean()
    s50 = close.rolling(50, min_periods=50).std()

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    m10_a = m10.to_numpy(dtype=float, na_value=np.nan)
    s10_a = s10.to_numpy(dtype=float, na_value=np.nan)
    m20_a = m20.to_numpy(dtype=float, na_value=np.nan)
    s20_a = s20.to_numpy(dtype=float, na_value=np.nan)
    m50_a = m50.to_numpy(dtype=float, na_value=np.nan)
    s50_a = s50.to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(c_arr)
        & ~np.isnan(m10_a)
        & ~np.isnan(s10_a)
        & (s10_a > 1e-12)
        & ~np.isnan(m20_a)
        & ~np.isnan(s20_a)
        & (s20_a > 1e-12)
        & ~np.isnan(m50_a)
        & ~np.isnan(s50_a)
        & (s50_a > 1e-12)
    )

    buy = (
        valid
        & (c_arr < m10_a - 1.8 * s10_a)
        & (c_arr < m20_a - 1.8 * s20_a)
        & (c_arr < m50_a - 1.8 * s50_a)
    )
    sell = (
        valid
        & (c_arr > m10_a + 1.8 * s10_a)
        & (c_arr > m20_a + 1.8 * s20_a)
        & (c_arr > m50_a + 1.8 * s50_a)
    )
    hold = valid & ~buy & ~sell

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_lehmann_short_term_reversal(
    open_p: pd.Series,
    close: pd.Series,
    return_window: int = 3,
    z_window: int = 20,
    z_thresh: float = 2.2,
) -> pd.Series:
    """Lehmann (1990) Short-Term Return Reversal (MR013).

    Measures rolling standardized return shock over short horizon.
    Buy: Z_ret < -z_thresh and Close > Open (bounce off oversold shock)
    Sell: Z_ret > +z_thresh and Close < Open (rejection off overbought shock)
    Hold: Mean-reverting continuation (-z_thresh <= Z_ret < -0.8 or 0.8 < Z_ret <= z_thresh)
    """
    n = len(close)
    if n < z_window + return_window:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    c_lag = close.shift(return_window)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    o_arr = open_p.to_numpy(dtype=float, na_value=np.nan)
    clag_arr = c_lag.to_numpy(dtype=float, na_value=np.nan)

    valid_c = ~np.isnan(c_arr) & ~np.isnan(clag_arr) & (clag_arr > 0)
    r3 = np.full(n, np.nan, dtype=float)
    np.divide(c_arr - clag_arr, clag_arr, out=r3, where=valid_c)

    r3_s = pd.Series(r3, index=close.index)
    roll_mean = r3_s.rolling(z_window).mean().to_numpy(dtype=float, na_value=np.nan)
    roll_std = r3_s.rolling(z_window).std().to_numpy(dtype=float, na_value=np.nan)

    valid_z = ~np.isnan(r3) & ~np.isnan(roll_mean) & ~np.isnan(roll_std) & (roll_std > 1e-12)
    z_ret = np.zeros(n, dtype=float)
    np.divide(r3 - roll_mean, roll_std, out=z_ret, where=valid_z)

    buy = valid_z & (z_ret < -z_thresh) & (c_arr > o_arr)
    sell = valid_z & (z_ret > z_thresh) & (c_arr < o_arr)
    hold = (
        valid_z
        & ~buy
        & ~sell
        & (((z_ret >= -z_thresh) & (z_ret < -0.8)) | ((z_ret <= z_thresh) & (z_ret > 0.8)))
    )

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_lo_mackinlay_variance_ratio(
    close: pd.Series,
    q: int = 4,
    window: int = 30,
    vr_thresh: float = 0.75,
    z_thresh: float = 1.8,
) -> pd.Series:
    """Lo & MacKinlay (1988) Variance Ratio Mean Reversion (MR014).

    VR(q) = Var(r_q) / (q * Var(r_1)).
    When VR < vr_thresh, price is statistically anti-persistent (mean-reverting).
    Buy: VR < vr_thresh and Price Z-Score < -z_thresh and Close > Close.shift(1)
    Sell: VR < vr_thresh and Price Z-Score > +z_thresh and Close < Close.shift(1)
    Hold: VR < vr_thresh while |Price Z-Score| > 0.5
    """
    n = len(close)
    if n < window + q:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    log_c = np.log(close.to_numpy(dtype=float, na_value=np.nan))
    r1 = pd.Series(np.diff(log_c, prepend=np.nan), index=close.index)
    rq = pd.Series(log_c - np.roll(log_c, q), index=close.index)
    rq.iloc[:q] = np.nan

    var1 = r1.rolling(window).var()
    varq = rq.rolling(window).var()

    var1_arr = var1.to_numpy(dtype=float, na_value=np.nan)
    varq_arr = varq.to_numpy(dtype=float, na_value=np.nan)

    valid_vr = ~np.isnan(var1_arr) & ~np.isnan(varq_arr) & (var1_arr > 1e-14)
    vr = np.full(n, np.nan, dtype=float)
    np.divide(varq_arr, q * var1_arr, out=vr, where=valid_vr)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    sma_arr = sma20.to_numpy(dtype=float, na_value=np.nan)
    std_arr = std20.to_numpy(dtype=float, na_value=np.nan)

    valid_z = ~np.isnan(c_arr) & ~np.isnan(sma_arr) & ~np.isnan(std_arr) & (std_arr > 1e-12)
    z_price = np.zeros(n, dtype=float)
    np.divide(c_arr - sma_arr, std_arr, out=z_price, where=valid_z)

    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)
    valid = valid_vr & valid_z & ~np.isnan(c_prev)
    mr_regime = valid & (vr < vr_thresh)

    buy = mr_regime & (z_price < -z_thresh) & (c_arr > c_prev)
    sell = mr_regime & (z_price > z_thresh) & (c_arr < c_prev)
    hold = mr_regime & ~buy & ~sell & (np.abs(z_price) > 0.5)

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_ehlers_roofing_filter_reversion(
    close: pd.Series,
    hp_period: int = 48,
    ss_period: int = 10,
    std_window: int = 30,
) -> pd.Series:
    """John Ehlers (2013) Roofing Filter Reversion (MR015).

    Two-pole High-Pass Filter removes macro trend, Two-pole SuperSmoother Filter
    removes high-frequency tick noise.
    Buy: Standardized Roofing Filter hooks up from < -1.8
    Sell: Standardized Roofing Filter hooks down from > +1.8
    Hold: Mean-reverting path back toward zero
    """
    n = len(close)
    if n < hp_period + std_window:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    c = close.to_numpy(dtype=float, na_value=np.nan)

    # 1. Two-pole High-Pass Filter
    hp = np.zeros(n, dtype=float)
    rad_hp = np.sqrt(2.0) * np.pi / hp_period
    alpha1 = (np.cos(rad_hp) + np.sin(rad_hp) - 1.0) / np.cos(rad_hp)
    c1 = (1.0 - alpha1 / 2.0) ** 2
    c2 = 2.0 * (1.0 - alpha1)
    c3 = -((1.0 - alpha1) ** 2)

    for i in range(2, n):
        hp[i] = c1 * (c[i] - 2.0 * c[i - 1] + c[i - 2]) + c2 * hp[i - 1] + c3 * hp[i - 2]

    # 2. Two-pole SuperSmoother Filter on HP
    filt = np.zeros(n, dtype=float)
    rad_ss = np.sqrt(2.0) * np.pi / ss_period
    a1 = np.exp(-rad_ss)
    b1 = 2.0 * a1 * np.cos(rad_ss)
    coef2 = b1
    coef3 = -(a1**2)
    coef1 = 1.0 - coef2 - coef3

    for i in range(2, n):
        filt[i] = coef1 * (hp[i] + hp[i - 1]) / 2.0 + coef2 * filt[i - 1] + coef3 * filt[i - 2]

    # 3. Standardize by rolling std
    filt_s = pd.Series(filt, index=close.index)
    roll_std = filt_s.rolling(std_window).std().to_numpy(dtype=float, na_value=np.nan)

    valid = ~np.isnan(roll_std) & (roll_std > 1e-12)
    z_roof = np.zeros(n, dtype=float)
    np.divide(filt, roll_std, out=z_roof, where=valid)

    z_prev = np.roll(z_roof, 1)
    z_prev[0] = 0.0

    buy = valid & (z_prev < -1.8) & (z_roof >= -1.8)
    sell = valid & (z_prev > 1.8) & (z_roof <= 1.8)
    hold = (
        valid
        & ~buy
        & ~sell
        & (((z_roof < -0.3) & (z_roof >= -1.8)) | ((z_roof > 0.3) & (z_roof <= 1.8)))
    )

    return pd.Series(
        np.select(
            [buy, sell, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_amihud_liquidity_exhaustion(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    z_window: int = 20,
    extreme_window: int = 15,
    z_thresh: float = 2.0,
) -> pd.Series:
    """Amihud (2002) Illiquidity Shock & Exhaustion Reversion (MR016).

    Measures sudden spikes in ILLIQ = |r_t| / (Volume_t * Close_t) at rolling price extremes
    accompanied by long rejection wicks.
    Buy: ILLIQ Z-score > z_thresh at extreme Low with lower wick >= 35% and Close > Open
    Sell: ILLIQ Z-score > z_thresh at extreme High with upper wick >= 35% and Close < Open
    """
    n = len(close)
    if n < max(z_window, extreme_window) + 2:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    o_arr = open_p.to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    v_arr = volume.to_numpy(dtype=float, na_value=np.nan)
    c_prev = close.shift(1).to_numpy(dtype=float, na_value=np.nan)

    valid_ret = ~np.isnan(c_arr) & ~np.isnan(c_prev) & (c_prev > 0)
    abs_ret = np.zeros(n, dtype=float)
    np.divide(np.abs(c_arr - c_prev), c_prev, out=abs_ret, where=valid_ret)

    dvol = c_arr * v_arr
    valid_illiq = valid_ret & ~np.isnan(v_arr) & (dvol > 0)
    illiq = np.zeros(n, dtype=float)
    np.divide(abs_ret * 1e6, dvol, out=illiq, where=valid_illiq)

    illiq_s = pd.Series(illiq, index=close.index)
    mean_illiq = illiq_s.rolling(z_window).mean().to_numpy(dtype=float, na_value=np.nan)
    std_illiq = illiq_s.rolling(z_window).std().to_numpy(dtype=float, na_value=np.nan)

    valid_z = valid_illiq & ~np.isnan(mean_illiq) & ~np.isnan(std_illiq) & (std_illiq > 1e-14)
    z_illiq = np.zeros(n, dtype=float)
    np.divide(illiq - mean_illiq, std_illiq, out=z_illiq, where=valid_z)

    rng = np.maximum(h_arr - l_arr, 1e-12)
    body_low = np.minimum(o_arr, c_arr)
    body_high = np.maximum(o_arr, c_arr)
    lower_wick_pct = (body_low - l_arr) / rng
    upper_wick_pct = (h_arr - body_high) / rng

    low_roll = low.shift(1).rolling(extreme_window).min().to_numpy(dtype=float, na_value=np.nan)
    high_roll = high.shift(1).rolling(extreme_window).max().to_numpy(dtype=float, na_value=np.nan)

    is_low_extreme = l_arr <= low_roll * 1.002
    is_high_extreme = h_arr >= high_roll * 0.998

    buy = (
        valid_z & (z_illiq > z_thresh) & is_low_extreme & (lower_wick_pct >= 0.35) & (c_arr > o_arr)
    )
    sell = (
        valid_z
        & (z_illiq > z_thresh)
        & is_high_extreme
        & (upper_wick_pct >= 0.35)
        & (c_arr < o_arr)
    )

    return pd.Series(
        np.select(
            [buy, sell],
            [SignalState.BUY, SignalState.SELL],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def _calc_bb_w_bottom_m_top(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_window: int = 20,
    lookback: int = 10,
) -> pd.Series:
    """John Bollinger (2001) W-Bottom & M-Top Structural Reversion (MR017).

    W-Bottom: Prior low broke below lower BB, current low retests support but stays
              inside lower BB, confirming buying support.
    M-Top: Prior high broke above upper BB, current high retests resistance but stays
           inside upper BB, confirming selling resistance.
    """
    n = len(close)
    if n < bb_window + lookback:
        return pd.Series(SignalState.NONE, index=close.index, dtype=str)

    sma = close.rolling(bb_window).mean()
    std = close.rolling(bb_window).std()
    bb_upper = (sma + 2.0 * std).to_numpy(dtype=float, na_value=np.nan)
    bb_lower = (sma - 2.0 * std).to_numpy(dtype=float, na_value=np.nan)
    sma_arr = sma.to_numpy(dtype=float, na_value=np.nan)

    l_arr = low.to_numpy(dtype=float, na_value=np.nan)
    h_arr = high.to_numpy(dtype=float, na_value=np.nan)
    c_arr = close.to_numpy(dtype=float, na_value=np.nan)
    o_arr = open_p.to_numpy(dtype=float, na_value=np.nan)

    penetrated_lower = (low < (sma - 2.0 * std)).astype(float)
    penetrated_upper = (high > (sma + 2.0 * std)).astype(float)

    prior_pen_lower = (
        penetrated_lower.shift(1).rolling(lookback).max().to_numpy(dtype=float, na_value=np.nan)
        > 0.5
    )
    prior_pen_upper = (
        penetrated_upper.shift(1).rolling(lookback).max().to_numpy(dtype=float, na_value=np.nan)
        > 0.5
    )

    prior_min_low = low.shift(1).rolling(lookback).min().to_numpy(dtype=float, na_value=np.nan)
    prior_max_high = high.shift(1).rolling(lookback).max().to_numpy(dtype=float, na_value=np.nan)

    valid = (
        ~np.isnan(bb_lower)
        & ~np.isnan(bb_upper)
        & ~np.isnan(prior_min_low)
        & ~np.isnan(prior_max_high)
    )

    w_bottom = (
        valid
        & prior_pen_lower
        & (l_arr <= prior_min_low * 1.015)
        & (l_arr >= bb_lower)
        & (c_arr > o_arr)
    )

    m_top = (
        valid
        & prior_pen_upper
        & (h_arr >= prior_max_high * 0.985)
        & (h_arr <= bb_upper)
        & (c_arr < o_arr)
    )

    hold = (
        valid
        & ~w_bottom
        & ~m_top
        & (((c_arr < sma_arr) & (c_arr > bb_lower)) | ((c_arr > sma_arr) & (c_arr < bb_upper)))
    )

    return pd.Series(
        np.select(
            [w_bottom, m_top, hold],
            [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            default=SignalState.NONE,
        ),
        index=close.index,
        dtype=str,
    )


def generate_mean_reversion_signals(
    df: pd.DataFrame,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Generate all 12 Mean Reversion trading signals from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing OHLCV price series.
    show_progress : bool, optional
        Whether to display a real-time progress bar, by default False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 17 columns ending with '_signal', with values in
        ['buy', 'sell', 'hold', 'none'] and index matching the input df.
    """
    df_norm = normalize_ohlcv(df)
    signals = pd.DataFrame(index=df_norm.index)

    with GroupProgressBar(
        "Mean Reversion Signals",
        total=len(MEAN_REVERSION_SIGNAL_COLUMNS),
        enabled=show_progress,
    ) as pbar:
        if df_norm.empty:
            for col in MEAN_REVERSION_SIGNAL_COLUMNS:
                signals[col] = pd.Series(dtype=str)
            pbar.update(len(MEAN_REVERSION_SIGNAL_COLUMNS))
            return signals

        open_p = df_norm["open"]
        high = df_norm["high"]
        low = df_norm["low"]
        close = df_norm["close"]
        volume = df_norm["volume"]

        # 1. Connors RSI(2) Trend-Filtered Reversion (MR001)
        signals["mr_connors_rsi2_regime_signal"] = _calc_connors_rsi2_regime(close)
        pbar.update(1)

        # 2. OU Process Spread Reversion (MR002)
        signals["mr_ou_process_spread_reversion_signal"] = _calc_ou_process_spread_reversion(
            close, window=30
        )
        pbar.update(1)

        # 3. Rolling VWAP Distance Z-Score (MR003)
        signals["mr_vwap_distance_zscore_signal"] = _calc_vwap_distance_zscore(
            high=high, low=low, close=close, volume=volume, window=20
        )
        pbar.update(1)

        # 4. Bollinger Bands %B Hook Reversion (MR004)
        signals["mr_bb_pct_b_hook_reversion_signal"] = _calc_bb_pct_b_hook_reversion(
            close=close, length=20, std=2.0, open_p=open_p
        )
        pbar.update(1)

        # 5. Keltner Channel 3-ATR Re-entry Reversal (MR005)
        signals["mr_keltner_atr_stretch_reentry_signal"] = _calc_keltner_atr_stretch_reentry(
            high=high, low=low, close=close, length=20, mult=3.0
        )
        pbar.update(1)

        # 6. Kurtosis Fat-Tail Exhaustion (MR006)
        signals["mr_kurtosis_fat_tail_exhaustion_signal"] = _calc_kurtosis_fat_tail_exhaustion(
            close=close, window=20
        )
        pbar.update(1)

        # 7. Disparity Index Stretch (MR007)
        signals["mr_dual_ma_disparity_index_signal"] = _calc_dual_ma_disparity_index(
            close=close, length=20, threshold=3.5, open_p=open_p
        )
        pbar.update(1)

        # 8. Linear Regression Residuals Z-Score (MR008)
        signals["mr_linreg_residual_zscore_signal"] = _calc_linreg_residual_zscore(
            close=close, length=20
        )
        pbar.update(1)

        # 9. Williams %R & CCI Confluence (MR009)
        signals["mr_wr_cci_double_oversold_signal"] = _calc_wr_cci_double_oversold(
            high=high, low=low, close=close, wr_len=14, cci_len=20
        )
        pbar.update(1)

        # 10. Session Range Fade (MR010)
        signals["mr_session_range_fade_signal"] = _calc_session_range_fade(
            df=df_norm, open_p=open_p, high=high, low=low, close=close
        )
        pbar.update(1)

        # 11. Volume Climax Absorption Reversion (MR011)
        signals["mr_volume_climax_absorption_reversion_signal"] = (
            _calc_volume_climax_absorption_reversion(
                open_p=open_p, high=high, low=low, close=close, volume=volume, window=20
            )
        )
        pbar.update(1)

        # 12. Multi-Period Stretch Consensus (MR012)
        signals["mr_multi_period_stretch_consensus_signal"] = _calc_multi_period_stretch_consensus(
            close=close
        )
        pbar.update(1)

        # 13. Lehmann Short-Term Return Reversal (MR013)
        signals["mr_lehmann_short_term_reversal_signal"] = _calc_lehmann_short_term_reversal(
            open_p=open_p, close=close
        )
        pbar.update(1)

        # 14. Lo & MacKinlay Variance Ratio (MR014)
        signals["mr_lo_mackinlay_variance_ratio_signal"] = _calc_lo_mackinlay_variance_ratio(
            close=close
        )
        pbar.update(1)

        # 15. Ehlers Roofing Filter Reversion (MR015)
        signals["mr_ehlers_roofing_filter_reversion_signal"] = (
            _calc_ehlers_roofing_filter_reversion(close=close)
        )
        pbar.update(1)

        # 16. Amihud Liquidity Exhaustion (MR016)
        signals["mr_amihud_liquidity_exhaustion_signal"] = _calc_amihud_liquidity_exhaustion(
            open_p=open_p, high=high, low=low, close=close, volume=volume
        )
        pbar.update(1)

        # 17. Bollinger Bands W-Bottom / M-Top (MR017)
        signals["mr_bb_w_bottom_m_top_signal"] = _calc_bb_w_bottom_m_top(
            open_p=open_p, high=high, low=low, close=close
        )
        pbar.update(1)

        for col in MEAN_REVERSION_SIGNAL_COLUMNS:
            if col not in signals.columns:
                signals[col] = SignalState.NONE
            else:
                signals[col] = signals[col].fillna(SignalState.NONE)

        return signals[MEAN_REVERSION_SIGNAL_COLUMNS]
