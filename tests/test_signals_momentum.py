from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.momentum import (
    MOMENTUM_SIGNAL_COLUMNS,
    _bound_signal,
    _calc_ao_saucer,
    _calc_cmo,
    _calc_fisher,
    _crossover_signal,
    generate_momentum_signals,
)


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


EXPECTED_MOMENTUM_SIGNALS = MOMENTUM_SIGNAL_COLUMNS


def test_momentum_signals_all_23_columns_present():
    """Verify generate_momentum_signals produces exactly the expected momentum signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_momentum_signals(df)

    assert len(EXPECTED_MOMENTUM_SIGNALS) == 30
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 30
    assert list(res.index) == list(df.index)

    for col in EXPECTED_MOMENTUM_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"


def test_momentum_signals_all_states_valid():
    """Verify that every value in every signal column is strictly within ALL_SIGNAL_STATES and has no NaNs."""
    df = make_synthetic_ohlcv(250)
    res = generate_momentum_signals(df)

    for col in res.columns:
        assert not res[col].isna().any(), f"Column {col} contains unexpected NaN values"
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), (
            f"Column {col} contains invalid states: {unique_vals - ALL_SIGNAL_STATES}"
        )


def test_momentum_signals_state_occurrences():
    """Verify that BUY, SELL, HOLD, and NONE states actually occur in the output across signals."""
    df = make_synthetic_ohlcv(300)
    res = generate_momentum_signals(df)

    all_values = set()
    for col in res.columns:
        all_values.update(res[col].unique())

    assert SignalState.BUY in all_values
    assert SignalState.SELL in all_values
    assert SignalState.HOLD in all_values
    assert SignalState.NONE in all_values


def test_momentum_signals_short_dataframe():
    """Verify graceful execution without exceptions when given a short dataframe (warmup edge case)."""
    df_short = make_synthetic_ohlcv(10)
    res = generate_momentum_signals(df_short)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert len(res.columns) == 30

    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_momentum_signals_empty_dataframe():
    """Verify handling of empty dataframe."""
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_momentum_signals(df_empty)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert len(res.columns) == 30
    for col in res.columns:
        assert col.endswith("_signal")


def test_momentum_signals_normalization():
    """Verify that uppercase column names are properly normalized."""
    df = make_synthetic_ohlcv(50)
    df_upper = df.rename(
        columns={
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
            "volume": "VOLUME",
        }
    )
    res = generate_momentum_signals(df_upper)

    assert len(res) == 50
    assert len(res.columns) == 30


def test_momentum_signals_missing_columns():
    """Verify that missing OHLCV columns raise ValueError."""
    df_invalid = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        generate_momentum_signals(df_invalid)


def test_momentum_signals_datetime_index_preserved():
    """Verify that custom DatetimeIndex is preserved across all output columns."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = make_synthetic_ohlcv(100)
    df.index = dates
    res = generate_momentum_signals(df)

    assert isinstance(res.index, pd.DatetimeIndex)
    assert res.index.equals(dates)


def test_momentum_signals_bullish_and_bearish_trends():
    """Verify momentum signals respond intuitively to clear unidirectional price movements."""
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
    res_bull = generate_momentum_signals(df_bull)
    assert (res_bull["mom_rsi_50_cross_14_signal"].iloc[30:] == SignalState.HOLD).all()
    assert (res_bull["mom_roc_zero_cross_5_signal"].iloc[30:] == SignalState.HOLD).all()
    assert (res_bull["mom_roc_zero_cross_10_signal"].iloc[30:] == SignalState.HOLD).all()

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
    res_bear = generate_momentum_signals(df_bear)
    assert (res_bear["mom_rsi_50_cross_14_signal"].iloc[30:] == SignalState.NONE).all()
    assert (res_bear["mom_roc_zero_cross_5_signal"].iloc[30:] == SignalState.NONE).all()


def test_crossover_signal_edge_cases():
    """Verify _crossover_signal helper on empty and all-nan inputs."""
    empty = pd.Series([], dtype=float)
    assert _crossover_signal(empty, empty).empty

    nan_s = pd.Series([np.nan, np.nan], dtype=float)
    res = _crossover_signal(nan_s, nan_s)
    assert (res == SignalState.NONE).all()


def test_bound_signal_direct():
    """Verify _bound_signal helper with buy_below=True and False."""
    vals = pd.Series([10.0, 50.0, 90.0, np.nan])
    res_buy_below = _bound_signal(vals, 20.0, 80.0, buy_below=True)
    assert list(res_buy_below) == [
        SignalState.BUY,
        SignalState.HOLD,
        SignalState.SELL,
        SignalState.NONE,
    ]

    res_breakout = _bound_signal(vals, 20.0, 80.0, buy_below=False)
    assert list(res_breakout) == [
        SignalState.SELL,
        SignalState.HOLD,
        SignalState.BUY,
        SignalState.NONE,
    ]


def test_fisher_and_cmo_edge_cases():
    """Verify _calc_fisher and _calc_cmo on small datasets and zero lengths."""
    high = pd.Series([10.0, 11.0, 12.0])
    low = pd.Series([9.0, 10.0, 11.0])
    close = pd.Series([9.5, 10.5, 11.5])

    f, s = _calc_fisher(high, low, length=9)
    assert f.isna().all()
    assert s.isna().all()

    cmo = _calc_cmo(close, length=14)
    assert isinstance(cmo, pd.Series)

    f0, s0 = _calc_fisher(high, low, length=0)
    assert f0.isna().all()


def test_ao_saucer_patterns():
    """Verify Awesome Oscillator Saucer bullish and bearish recognition."""
    # Bullish saucer: 3 bars > 0, bar1 > bar2, bar3 > bar2
    ao_bull = pd.Series([1.0, 2.0, 1.5, 1.8, 1.9])
    res_bull = _calc_ao_saucer(ao_bull)
    assert res_bull.iloc[3] == SignalState.BUY

    # Bearish saucer: 3 bars < 0, bar1 < bar2, bar3 < bar2
    ao_bear = pd.Series([-1.0, -2.0, -1.5, -1.8, -1.9])
    res_bear = _calc_ao_saucer(ao_bear)
    assert res_bear.iloc[3] == SignalState.SELL
