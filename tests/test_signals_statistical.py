from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.statistical import (
    STATISTICAL_SIGNAL_COLUMNS,
    _calc_chop_regime,
    _calc_fractal_dimension_index,
    _calc_ker,
    _calc_linreg_price_cross,
    _calc_linreg_slope,
    _calc_rolling_half_life_reversion,
    _calc_rolling_quantile_extremes,
    _calc_rolling_skewness_reversal,
    _calc_variance_ratio_test,
    _calc_zscore,
    generate_statistical_signals,
)


def make_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataset with varied statistical regimes for testing."""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.02

    # Add trending regimes (low chop, high KER) and sideways noisy regimes (high chop, low KER)
    if n >= 150:
        # Strong steady trend: bars 30 to 70
        returns[30:70] = 0.015 + np.abs(np.random.randn(40)) * 0.005
        # Mean reverting / sideways chop: bars 80 to 120
        returns[80:120] = np.sin(np.linspace(0, 8 * np.pi, 40)) * 0.02

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.random.uniform(0.005, 0.03, n))
    low = close * (1.0 - np.random.uniform(0.005, 0.03, n))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.randint(1000, 50000, n).astype(float)

    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


EXPECTED_STATISTICAL_SIGNALS = STATISTICAL_SIGNAL_COLUMNS


def test_statistical_signals_all_20_columns_present():
    """Verify generate_statistical_signals produces exactly 20 statistical signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_statistical_signals(df)

    assert len(EXPECTED_STATISTICAL_SIGNALS) == 20
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 20
    assert list(res.index) == list(df.index)

    for col in EXPECTED_STATISTICAL_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"


def test_statistical_signals_all_states_valid():
    """Verify that every value in every signal column is strictly within ALL_SIGNAL_STATES and has no NaNs."""
    df = make_synthetic_ohlcv(250)
    res = generate_statistical_signals(df)

    for col in res.columns:
        assert not res[col].isna().any(), f"Column {col} contains unexpected NaN values"
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), (
            f"Column {col} contains invalid states: {unique_vals - ALL_SIGNAL_STATES}"
        )


def test_statistical_signals_state_occurrences():
    """Verify that BUY, SELL, HOLD, and NONE states occur in the output across signals."""
    df = make_synthetic_ohlcv(300)
    res = generate_statistical_signals(df)

    all_values = set()
    for col in res.columns:
        all_values.update(res[col].unique())

    assert SignalState.BUY in all_values
    assert SignalState.SELL in all_values
    assert SignalState.HOLD in all_values
    assert SignalState.NONE in all_values


def test_statistical_signals_short_dataframe():
    """Verify graceful execution without exceptions when given a short dataframe (warmup edge case)."""
    df_short = make_synthetic_ohlcv(10)
    res = generate_statistical_signals(df_short)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert len(res.columns) == 20

    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_statistical_signals_empty_dataframe():
    """Verify handling of empty dataframe."""
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_statistical_signals(df_empty)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert len(res.columns) == 20
    for col in res.columns:
        assert col.endswith("_signal")


def test_statistical_signals_normalization():
    """Verify that uppercase column names are properly normalized."""
    df = make_synthetic_ohlcv(50)
    df_upper = df.rename(
        columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "volume": "VOLUME"}
    )
    res = generate_statistical_signals(df_upper)

    assert len(res) == 50
    assert len(res.columns) == 20


def test_statistical_signals_missing_columns():
    """Verify that missing OHLCV columns raise ValueError."""
    df_invalid = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        generate_statistical_signals(df_invalid)


def test_statistical_signals_flat_data():
    """Verify flat / constant prices handle zero standard deviation without errors or NaNs."""
    df_flat = pd.DataFrame(
        {
            "open": [100.0] * 50,
            "high": [100.0] * 50,
            "low": [100.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000.0] * 50,
        }
    )
    res = generate_statistical_signals(df_flat)
    assert len(res) == 50
    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_statistical_signals_datetime_index_preserved():
    """Verify that DatetimeIndex is preserved in output."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = make_synthetic_ohlcv(100)
    df.index = dates
    res = generate_statistical_signals(df)

    assert isinstance(res.index, pd.DatetimeIndex)
    assert (res.index == dates).all()


def test_zscore_calculation_direct():
    """Verify direct Z-score helper function logic."""
    # Test oversold and overbought price spikes
    prices = [10.0] * 20 + [5.0] + [10.0] * 20 + [25.0]
    s = pd.Series(prices)
    sig = _calc_zscore(s, window=10, lower_bound=-2.0, upper_bound=2.0)

    # During warmup (< 10 bars)
    assert sig.iloc[0] == SignalState.NONE
    # Oversold drop at index 20 should produce BUY
    assert sig.iloc[20] == SignalState.BUY
    # Overbought spike at index 41 should produce SELL
    assert sig.iloc[41] == SignalState.SELL
    # Normal flat region after warmup should be HOLD
    assert sig.iloc[15] == SignalState.HOLD


def test_ker_trend_filter_direct():
    """Verify Kaufman Efficiency Ratio (KER) calculation and directional filtering."""
    # Perfect linear trend (efficiency = 1.0)
    uptrend = pd.Series(np.linspace(10, 50, 30))
    downtrend = pd.Series(np.linspace(50, 10, 30))
    choppy = pd.Series([10.0, 12.0, 9.0, 11.0, 10.0] * 6)

    ker_up = _calc_ker(uptrend, length=10, threshold=0.6)
    ker_down = _calc_ker(downtrend, length=10, threshold=0.6)
    ker_chop = _calc_ker(choppy, length=10, threshold=0.6)

    assert ker_up.iloc[25] == SignalState.BUY
    assert ker_down.iloc[25] == SignalState.SELL
    assert ker_chop.iloc[25] == SignalState.HOLD


def test_chop_regime_direct():
    """Verify Choppiness Index trending (<38.2) vs consolidating (>61.8) regimes."""
    # Strong persistent uptrend
    n = 60
    close_trend = pd.Series(np.linspace(100, 200, n))
    high_trend = close_trend + 1.0
    low_trend = close_trend - 1.0
    chop_trend = _calc_chop_regime(high_trend, low_trend, close_trend, length=14)

    # Choppy sideways oscillation
    osc = np.array([100.0, 105.0, 95.0, 104.0, 96.0] * 12)
    close_chop = pd.Series(osc)
    high_chop = close_chop + 2.0
    low_chop = close_chop - 2.0
    chop_side = _calc_chop_regime(high_chop, low_chop, close_chop, length=14)

    # Trend should produce BUY (< 38.2)
    assert SignalState.BUY in chop_trend.values
    # Chop should produce SELL (> 61.8)
    assert SignalState.SELL in chop_side.values


def test_rolling_quantile_extremes_direct():
    """Verify 20-period rolling quantile extremes (5th and 95th percentile triggers)."""
    # 20 flat numbers followed by extreme drop and extreme rise
    vals = [100.0] * 19 + [50.0] + [100.0] * 19 + [200.0]
    s = pd.Series(vals)
    sig = _calc_rolling_quantile_extremes(s, window=20, lower_q=0.05, upper_q=0.95)

    assert sig.iloc[19] == SignalState.BUY
    assert sig.iloc[39] == SignalState.SELL


def test_linreg_slope_and_cross_direct():
    """Verify linear regression slope and price cross helpers."""
    uptrend = pd.Series(np.linspace(10, 50, 40))
    downtrend = pd.Series(np.linspace(50, 10, 40))

    slope_up = _calc_linreg_slope(uptrend, length=14)
    slope_down = _calc_linreg_slope(downtrend, length=14)

    assert slope_up.iloc[20] == SignalState.BUY
    assert slope_down.iloc[20] == SignalState.SELL

    # Crossover test: series below linreg then sharp jump above
    crossover_data = [100.0] * 32 + [90.0, 115.0, 120.0]
    cross_sig = _calc_linreg_price_cross(pd.Series(crossover_data), length=30)
    assert SignalState.BUY in cross_sig.values
    assert cross_sig.iloc[-1] in [SignalState.BUY, SignalState.HOLD]


def test_fractal_dimension_index_direct():
    """Verify 30-period Fractal Dimension Index trending vs consolidating regimes with SMA20."""
    # Strong persistent uptrend: FDI < 1.45 and Close > SMA20 -> BUY
    n = 60
    close_up = pd.Series(np.linspace(100, 200, n))
    high_up = close_up + 0.5
    low_up = close_up - 0.5
    fdi_up = _calc_fractal_dimension_index(high_up, low_up, close_up, length=30, threshold=1.45)

    # Strong persistent downtrend: FDI < 1.45 and Close < SMA20 -> SELL
    close_down = pd.Series(np.linspace(200, 100, n))
    high_down = close_down + 0.5
    low_down = close_down - 0.5
    fdi_down = _calc_fractal_dimension_index(
        high_down, low_down, close_down, length=30, threshold=1.45
    )

    # High chop / noise: FDI > 1.45 -> HOLD
    np.random.seed(42)
    noise_close = pd.Series(100.0 + np.random.uniform(-5, 5, n))
    noise_high = noise_close + np.random.uniform(1, 4, n)
    noise_low = noise_close - np.random.uniform(1, 4, n)
    fdi_chop = _calc_fractal_dimension_index(
        noise_high, noise_low, noise_close, length=30, threshold=1.45
    )

    assert fdi_up.iloc[45] == SignalState.BUY
    assert fdi_down.iloc[45] == SignalState.SELL
    assert SignalState.HOLD in fdi_chop.iloc[35:].values


def test_rolling_half_life_reversion_direct():
    """Verify Ornstein-Uhlenbeck Half-Life mean reversion regime and Z-score triggers."""
    np.random.seed(42)
    n = 150
    # Mean-reverting AR(1) OU process: lambda ~ -0.08 -> half_life ~ 8.66 in [3, 15]
    prices = np.zeros(n)
    prices[0] = 100.0
    for t in range(1, n):
        prices[t] = prices[t - 1] - 0.08 * (prices[t - 1] - 100.0) + np.random.randn() * 1.5

    close_s = pd.Series(prices)
    hl_sig = _calc_rolling_half_life_reversion(
        close_s, window=30, hl_min=3.0, hl_max=15.0, z_thresh=1.5
    )

    assert SignalState.BUY in hl_sig.values
    assert SignalState.SELL in hl_sig.values
    assert SignalState.HOLD in hl_sig.values


def test_variance_ratio_test_direct():
    """Verify Lo-MacKinlay Variance Ratio test trending structure (VR > 1.25) with ROC5."""
    np.random.seed(42)
    n = 100
    e = np.random.randn(n) * 0.01

    # Autocorrelated positive returns -> VR > 1.25, ROC5 > 0 -> BUY
    r_up = np.zeros(n)
    for t in range(1, n):
        r_up[t] = 0.6 * r_up[t - 1] + e[t] + 0.01
    close_up = pd.Series(100.0 * np.exp(np.cumsum(r_up)))
    vr_up = _calc_variance_ratio_test(close_up, q=5, window=30, threshold=1.25)

    # Autocorrelated negative returns -> VR > 1.25, ROC5 < 0 -> SELL
    r_down = np.zeros(n)
    for t in range(1, n):
        r_down[t] = 0.6 * r_down[t - 1] + e[t] - 0.01
    close_down = pd.Series(100.0 * np.exp(np.cumsum(r_down)))
    vr_down = _calc_variance_ratio_test(close_down, q=5, window=30, threshold=1.25)

    assert SignalState.BUY in vr_up.values
    assert SignalState.SELL in vr_down.values


def test_rolling_skewness_reversal_direct():
    """Verify 20-period return skewness reversal triggers on panic/euphoria exhaustion."""
    np.random.seed(42)
    # Series with negative skew shock followed by price bounce
    ret_neg = [0.001] * 40
    ret_neg[25] = -0.10  # Extreme negative return shock
    ret_neg[26] = 0.02  # Price turns up (Close > Close[1]) -> BUY
    close_panic = pd.Series(100.0 * np.exp(np.cumsum(ret_neg)))

    skew_sig_buy = _calc_rolling_skewness_reversal(close_panic, window=20, skew_thresh=1.50)
    assert skew_sig_buy.iloc[26] == SignalState.BUY

    # Series with positive skew shock followed by price downward turn
    ret_pos = [0.001] * 40
    ret_pos[25] = 0.10  # Extreme positive return shock
    ret_pos[26] = -0.02  # Price turns down (Close < Close[1]) -> SELL
    close_euphoria = pd.Series(100.0 * np.exp(np.cumsum(ret_pos)))

    skew_sig_sell = _calc_rolling_skewness_reversal(close_euphoria, window=20, skew_thresh=1.50)
    assert skew_sig_sell.iloc[26] == SignalState.SELL


def test_statistical_helpers_edge_cases():
    """Verify helpers handle empty series and window > length gracefully."""
    empty_s = pd.Series(dtype=float)
    assert len(_calc_zscore(empty_s, 10, -2.0, 2.0)) == 0
    assert len(_calc_ker(empty_s, 10, 0.6)) == 0
    assert len(_calc_chop_regime(empty_s, empty_s, empty_s, 14)) == 0
    assert len(_calc_rolling_quantile_extremes(empty_s, 20, 0.05, 0.95)) == 0
    assert len(_calc_linreg_slope(empty_s, 14)) == 0
    assert len(_calc_linreg_price_cross(empty_s, 30)) == 0
    assert len(_calc_fractal_dimension_index(empty_s, empty_s, empty_s, 30)) == 0
    assert len(_calc_rolling_half_life_reversion(empty_s, 30)) == 0
    assert len(_calc_variance_ratio_test(empty_s, 5, 30)) == 0
    assert len(_calc_rolling_skewness_reversal(empty_s, 20)) == 0


def test_signals_package_export():
    """Verify generate_statistical_signals is properly exported in signalx.signals."""
    import signalx.signals

    assert hasattr(signalx.signals, "generate_statistical_signals")
    assert callable(signalx.signals.generate_statistical_signals)
