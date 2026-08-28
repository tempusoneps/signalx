from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.statistical import (
    STATISTICAL_SIGNAL_COLUMNS,
    _calc_chop_regime,
    _calc_ker,
    _calc_linreg_price_cross,
    _calc_linreg_slope,
    _calc_rolling_quantile_extremes,
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


def test_statistical_signals_all_11_columns_present():
    """Verify generate_statistical_signals produces exactly the expected statistical signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_statistical_signals(df)

    assert len(EXPECTED_STATISTICAL_SIGNALS) == 16
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 16
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
    assert len(res.columns) == 16

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
    assert len(res.columns) == 16
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
    assert len(res.columns) == 16


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


def test_statistical_helpers_edge_cases():
    """Verify helpers handle empty series and window > length gracefully."""
    empty_s = pd.Series(dtype=float)
    assert len(_calc_zscore(empty_s, 10, -2.0, 2.0)) == 0
    assert len(_calc_ker(empty_s, 10, 0.6)) == 0
    assert len(_calc_chop_regime(empty_s, empty_s, empty_s, 14)) == 0
    assert len(_calc_rolling_quantile_extremes(empty_s, 20, 0.05, 0.95)) == 0
    assert len(_calc_linreg_slope(empty_s, 14)) == 0
    assert len(_calc_linreg_price_cross(empty_s, 30)) == 0


def test_signals_package_export():
    """Verify generate_statistical_signals is properly exported in signalx.signals."""
    import signalx.signals

    assert hasattr(signalx.signals, "generate_statistical_signals")
    assert callable(signalx.signals.generate_statistical_signals)
