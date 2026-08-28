from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.trend import TREND_SIGNAL_COLUMNS, generate_trend_signals


def make_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataset for fast, isolated unit testing."""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.02
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


EXPECTED_TREND_SIGNALS = TREND_SIGNAL_COLUMNS


def test_trend_signals_all_30_columns_present():
    """Verify generate_trend_signals produces exactly the expected trend signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_trend_signals(df)

    assert len(EXPECTED_TREND_SIGNALS) == 41
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 41
    assert list(res.index) == list(df.index)

    for col in EXPECTED_TREND_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"


def test_trend_signals_all_states_valid():
    """Verify that every value in every signal column is strictly within ALL_SIGNAL_STATES and has no NaNs."""
    df = make_synthetic_ohlcv(250)
    res = generate_trend_signals(df)

    for col in res.columns:
        assert not res[col].isna().any(), f"Column {col} contains unexpected NaN values"
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), (
            f"Column {col} contains invalid states: {unique_vals - ALL_SIGNAL_STATES}"
        )


def test_trend_signals_state_occurrences():
    """Verify that BUY, SELL, HOLD, and NONE states actually occur in the output across signals."""
    df = make_synthetic_ohlcv(300)
    res = generate_trend_signals(df)

    all_values = set()
    for col in res.columns:
        all_values.update(res[col].unique())

    assert SignalState.BUY in all_values
    assert SignalState.SELL in all_values
    assert SignalState.HOLD in all_values
    assert SignalState.NONE in all_values


def test_trend_signals_short_dataframe():
    """Verify graceful execution without exceptions when given a short dataframe (warmup edge case)."""
    df_short = make_synthetic_ohlcv(10)
    res = generate_trend_signals(df_short)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert len(res.columns) == 41

    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_trend_signals_empty_dataframe():
    """Verify handling of empty dataframe."""
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_trend_signals(df_empty)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert len(res.columns) == 41
    for col in res.columns:
        assert col.endswith("_signal")


def test_trend_signals_normalization():
    """Verify that uppercase column names are properly normalized."""
    df = make_synthetic_ohlcv(50)
    df_upper = df.rename(
        columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "volume": "VOLUME"}
    )
    res = generate_trend_signals(df_upper)

    assert len(res) == 50
    assert len(res.columns) == 41


def test_trend_signals_missing_columns():
    """Verify that missing OHLCV columns raise ValueError."""
    df_invalid = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        generate_trend_signals(df_invalid)


def test_trend_signals_datetime_index_preserved():
    """Verify that custom DatetimeIndex is preserved across all output columns."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = make_synthetic_ohlcv(100)
    df.index = dates
    res = generate_trend_signals(df)

    assert isinstance(res.index, pd.DatetimeIndex)
    assert res.index.equals(dates)


def test_trend_signals_bullish_and_bearish_trends():
    """Verify trend signals respond intuitively to clear unidirectional price movements."""
    n = 100
    # Clear bullish trend
    bull_close = np.linspace(100.0, 200.0, n)
    df_bull = pd.DataFrame(
        {
            "open": bull_close - 0.5,
            "high": bull_close + 1.0,
            "low": bull_close - 1.0,
            "close": bull_close,
            "volume": np.full(n, 1000.0),
        }
    )
    res_bull = generate_trend_signals(df_bull)
    assert (res_bull["trend_price_above_sma20_signal"].iloc[30:] == SignalState.BUY).all()
    assert (res_bull["trend_price_above_ema50_signal"].iloc[60:] == SignalState.BUY).all()

    # Clear bearish trend
    bear_close = np.linspace(200.0, 100.0, n)
    df_bear = pd.DataFrame(
        {
            "open": bear_close + 0.5,
            "high": bear_close + 1.0,
            "low": bear_close - 1.0,
            "close": bear_close,
            "volume": np.full(n, 1000.0),
        }
    )
    res_bear = generate_trend_signals(df_bear)
    assert (res_bear["trend_price_above_sma20_signal"].iloc[30:] == SignalState.SELL).all()
    assert (res_bear["trend_price_above_ema50_signal"].iloc[60:] == SignalState.SELL).all()
