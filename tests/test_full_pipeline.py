from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import signalx
from signalx.constants import ALL_SIGNAL_STATES
from signalx.metadata import (
    SIGNAL_CATALOG,
    SIGNAL_CODE_CATALOG,
    to_code_names,
    to_semantic_names,
)
from signalx.signals import (
    generate_candlestick_signals,
    generate_composite_signals,
    generate_momentum_signals,
    generate_statistical_signals,
    generate_trend_signals,
    generate_volatility_signals,
    generate_volume_signals,
    run_all_signal_generators,
)


def make_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataset for fast, isolated unit testing."""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.02

    if n >= 150:
        # Bullish regime
        returns[30:60] = 0.02 + np.abs(np.random.randn(30)) * 0.005
        # Bearish regime
        returns[80:110] = -0.02 - np.abs(np.random.randn(30)) * 0.005

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


def test_full_pipeline_produces_all_signals_default_code():
    """Test end-to-end signal generation produces all coded signal columns by default."""
    df = make_synthetic_ohlcv(150)
    res = signalx.generate_signals(df)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 150

    # Ensure original OHLCV columns are present
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in res.columns

    signal_cols = [c for c in res.columns if c.endswith("_signal")]
    assert len(signal_cols) == 256
    assert len(signal_cols) == len(SIGNAL_CODE_CATALOG)
    assert set(signal_cols) == set(SIGNAL_CODE_CATALOG.keys())

    # Ensure all values belong to valid states
    for col in signal_cols:
        unique_vals = set(res[col].dropna().unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), f"Invalid state in {col}: {unique_vals}"


def test_full_pipeline_naming_semantic():
    """Test generate_signals with naming='semantic' produces semantic columns."""
    df = make_synthetic_ohlcv(150)
    res = signalx.generate_signals(df, naming="semantic")

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 150

    signal_cols = [c for c in res.columns if c.endswith("_signal")]
    assert len(signal_cols) == 256
    assert len(signal_cols) == len(SIGNAL_CATALOG)
    assert set(signal_cols) == set(SIGNAL_CATALOG.keys())


def test_full_pipeline_naming_invalid_raises():
    """Test that invalid naming parameter raises ValueError."""
    df = make_synthetic_ohlcv(50)
    with pytest.raises(ValueError, match="Invalid naming convention"):
        signalx.generate_signals(df, naming="invalid")

    with pytest.raises(ValueError, match="Invalid naming convention"):
        run_all_signal_generators(df, naming="unknown")


def test_full_pipeline_to_code_and_semantic_roundtrip():
    """Test conversion roundtrips between coded and semantic DataFrames."""
    df = make_synthetic_ohlcv(100)
    coded = signalx.generate_signals(df, naming="code")
    semantic = signalx.generate_signals(df, naming="semantic")

    # to_semantic_names(coded) should equal semantic
    converted_semantic = to_semantic_names(coded)
    pd.testing.assert_frame_equal(converted_semantic, semantic, check_like=True)

    # to_code_names(semantic) should equal coded
    converted_code = to_code_names(semantic)
    pd.testing.assert_frame_equal(converted_code, coded, check_like=True)


def test_full_pipeline_sorted_signal_columns():
    """Test that run_all_signal_generators returns alphabetically sorted columns."""
    df = make_synthetic_ohlcv(100)
    signals_code = run_all_signal_generators(df, naming="code")

    assert isinstance(signals_code, pd.DataFrame)
    assert len(signals_code) == 100
    assert len(signals_code.columns) == len(SIGNAL_CODE_CATALOG)
    assert list(signals_code.columns) == sorted(signals_code.columns)
    assert set(signals_code.columns) == set(SIGNAL_CODE_CATALOG.keys())
    assert all(c.endswith("_signal") for c in signals_code.columns)

    signals_semantic = run_all_signal_generators(df, naming="semantic")
    assert isinstance(signals_semantic, pd.DataFrame)
    assert len(signals_semantic) == 100
    assert len(signals_semantic.columns) == len(SIGNAL_CATALOG)
    assert list(signals_semantic.columns) == sorted(signals_semantic.columns)
    assert set(signals_semantic.columns) == set(SIGNAL_CATALOG.keys())


def test_full_pipeline_drop_ohlcv():
    """Test drop_ohlcv=True vs drop_ohlcv=False with both coded and semantic naming."""
    df = make_synthetic_ohlcv(100)

    # 1. drop_ohlcv=False (default)
    res_default = signalx.generate_signals(df, drop_ohlcv=False)
    assert len(res_default.columns) == 5 + len(SIGNAL_CODE_CATALOG)
    assert "close" in res_default.columns

    # 2. drop_ohlcv=True without date column (code naming)
    res_dropped_code = signalx.generate_signals(df, drop_ohlcv=True, naming="code")
    assert "close" not in res_dropped_code.columns
    assert "open" not in res_dropped_code.columns
    assert len(res_dropped_code.columns) == len(SIGNAL_CODE_CATALOG)
    assert set(res_dropped_code.columns) == set(SIGNAL_CODE_CATALOG.keys())

    # 3. drop_ohlcv=True without date column (semantic naming)
    res_dropped_semantic = signalx.generate_signals(df, drop_ohlcv=True, naming="semantic")
    assert "close" not in res_dropped_semantic.columns
    assert "open" not in res_dropped_semantic.columns
    assert len(res_dropped_semantic.columns) == len(SIGNAL_CATALOG)
    assert set(res_dropped_semantic.columns) == set(SIGNAL_CATALOG.keys())


def test_full_pipeline_with_date_columns():
    """Test drop_ohlcv=True preserves date/datetime/timestamp/time columns."""
    df = make_synthetic_ohlcv(100)
    df["Date"] = pd.date_range("2025-01-01", periods=100, freq="D")
    df["extra_metadata"] = "market_open"

    # drop_ohlcv=False should keep everything
    res_false = signalx.generate_signals(df, drop_ohlcv=False)
    assert "Date" in res_false.columns
    assert "extra_metadata" in res_false.columns
    assert "close" in res_false.columns
    assert len([c for c in res_false.columns if c.endswith("_signal")]) == len(SIGNAL_CODE_CATALOG)

    # drop_ohlcv=True should preserve Date column but drop OHLCV and extra non-date columns
    res_true = signalx.generate_signals(df, drop_ohlcv=True)
    assert "Date" in res_true.columns
    assert "close" not in res_true.columns
    assert "open" not in res_true.columns
    assert "extra_metadata" not in res_true.columns
    signal_cols = [c for c in res_true.columns if c.endswith("_signal")]
    assert len(signal_cols) == len(SIGNAL_CODE_CATALOG)
    assert len(res_true.columns) == 1 + len(SIGNAL_CODE_CATALOG)


def test_full_pipeline_datetime_index_preserved():
    """Test DataFrame with DatetimeIndex retains its index identically."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    df = make_synthetic_ohlcv(120)
    df.index = dates

    res = signalx.generate_signals(df)
    assert isinstance(res.index, pd.DatetimeIndex)
    pd.testing.assert_index_equal(res.index, df.index)

    res_dropped = signalx.generate_signals(df, drop_ohlcv=True)
    assert isinstance(res_dropped.index, pd.DatetimeIndex)
    pd.testing.assert_index_equal(res_dropped.index, df.index)


def test_full_pipeline_custom_index_preserved():
    """Test custom string index preservation."""
    df = make_synthetic_ohlcv(50)
    df.index = [f"bar_{i:04d}" for i in range(50)]

    res = signalx.generate_signals(df)
    assert list(res.index) == list(df.index)


def test_full_pipeline_large_dataset_performance():
    """Test execution on 1,000+ rows dataset for performance and correctness."""
    df = make_synthetic_ohlcv(1200)

    start_time = time.time()
    res = signalx.generate_signals(df)
    elapsed = time.time() - start_time

    assert len(res) == 1200
    assert len([c for c in res.columns if c.endswith("_signal")]) == len(SIGNAL_CODE_CATALOG)
    # Execution should be rapid (under 5 seconds)
    assert elapsed < 5.0, f"Pipeline took too long: {elapsed:.2f}s"


def test_full_pipeline_short_dataframe():
    """Test short dataframe produces expected shape and valid states."""
    df = make_synthetic_ohlcv(10)
    res = signalx.generate_signals(df)
    assert len(res) == 10
    signal_cols = [c for c in res.columns if c.endswith("_signal")]
    assert len(signal_cols) == len(SIGNAL_CODE_CATALOG)
    for col in signal_cols:
        assert set(res[col].unique()).issubset(ALL_SIGNAL_STATES)


def test_full_pipeline_empty_dataframe():
    """Test empty dataframe produces empty dataframe with correct columns."""
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = signalx.generate_signals(df)
    assert len(res) == 0
    signal_cols = [c for c in res.columns if c.endswith("_signal")]
    assert len(signal_cols) == len(SIGNAL_CODE_CATALOG)


def test_full_pipeline_missing_columns_validation():
    """Test that missing required OHLCV columns raises ValueError."""
    df = pd.DataFrame({"open": [1.0, 2.0], "close": [1.5, 2.5]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        signalx.generate_signals(df)


def test_full_pipeline_uppercase_column_normalization():
    """Test uppercase column normalization in pipeline."""
    df = make_synthetic_ohlcv(60)
    df.columns = [c.upper() for c in df.columns]
    res = signalx.generate_signals(df)
    assert len(res) == 60
    assert len([c for c in res.columns if c.endswith("_signal")]) == len(SIGNAL_CODE_CATALOG)


def test_signalx_package_exports():
    """Test top-level and submodule package exports."""
    assert hasattr(signalx, "generate_signals")
    assert hasattr(signalx, "SignalState")
    assert hasattr(signalx, "ALL_SIGNAL_STATES")
    assert hasattr(signalx, "to_code_names")
    assert hasattr(signalx, "to_semantic_names")
    assert hasattr(signalx, "get_code_to_name_map")
    assert hasattr(signalx, "get_name_to_code_map")
    assert hasattr(signalx, "SIGNAL_CATALOG")
    assert hasattr(signalx, "SIGNAL_CODE_CATALOG")
    assert hasattr(signalx, "get_signal_by_code")
    assert hasattr(signalx, "get_signal_by_name")
    assert hasattr(signalx, "get_signal_metadata")
    assert signalx.__version__ == "0.2.0"

    assert callable(signalx.to_code_names)
    assert callable(signalx.to_semantic_names)
    assert callable(signalx.get_code_to_name_map)
    assert callable(signalx.get_name_to_code_map)
    assert callable(signalx.get_signal_by_code)
    assert callable(signalx.get_signal_by_name)
    assert callable(signalx.get_signal_metadata)

    assert callable(run_all_signal_generators)
    assert callable(generate_trend_signals)
    assert callable(generate_momentum_signals)
    assert callable(generate_volatility_signals)
    assert callable(generate_volume_signals)
    assert callable(generate_candlestick_signals)
    assert callable(generate_statistical_signals)
    assert callable(generate_composite_signals)


def test_full_pipeline_with_progress_bar():
    """Test pipeline execution with show_progress=True."""
    df = make_synthetic_ohlcv(50)
    res = signalx.generate_signals(df, show_progress=True)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 50
    signal_cols = [c for c in res.columns if c.endswith("_signal")]
    assert len(signal_cols) == len(SIGNAL_CODE_CATALOG)
