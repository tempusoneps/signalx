from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signalx.utils import (
    REQUIRED_OHLCV_COLS,
    compute_signal_stats,
    load_dataframe,
    normalize_ohlcv,
    save_dataframe,
)


def test_required_ohlcv_cols_constant():
    assert REQUIRED_OHLCV_COLS == ["open", "high", "low", "close", "volume"]


def test_normalize_ohlcv_case_insensitive_and_strip():
    df = pd.DataFrame(
        {
            " Date ": pd.date_range("2026-01-01", periods=5, freq="5min"),
            "OPEN": ["100.0", "101.5", 102.0, 103.0, 104.0],
            "High": [105.0, 106.0, 107.0, 108.0, 109.0],
            " LOW ": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "Volume": [1000, 1500, 1200, 1800, 2000],
            "Ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        }
    )
    original_cols = list(df.columns)
    normalized = normalize_ohlcv(df)

    # Immutability of input
    assert list(df.columns) == original_cols

    # Normalized column names
    assert list(normalized.columns) == ["date", "open", "high", "low", "close", "volume", "ticker"]

    # Converted numeric types
    for col in REQUIRED_OHLCV_COLS:
        assert np.issubdtype(normalized[col].dtype, np.floating)

    assert normalized["open"].iloc[0] == 100.0
    assert normalized["ticker"].iloc[0] == "AAPL"


def test_normalize_ohlcv_missing_columns():
    df = pd.DataFrame({"close": [1, 2, 3], "open": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        normalize_ohlcv(df)


def test_normalize_ohlcv_coercion_non_numeric():
    df = pd.DataFrame(
        {
            "open": ["invalid", 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [104.0, 105.0],
            "volume": [1000, 1500],
        }
    )
    normalized = normalize_ohlcv(df)
    assert np.isnan(normalized["open"].iloc[0])
    assert normalized["open"].iloc[1] == 101.0


def test_load_dataframe_not_found():
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_dataframe("non_existent_file_path_12345.parquet")


def test_load_dataframe_unsupported_format(tmp_path: Path):
    bad_file = tmp_path / "test.json"
    bad_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataframe(bad_file)


def test_save_and_load_dataframe_roundtrip_csv_and_parquet(tmp_path: Path):
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [104.0, 105.0],
            "volume": [1000.0, 1500.0],
        }
    )

    # CSV in nested dir
    csv_path = tmp_path / "nested" / "dir" / "data.csv"
    res_path = save_dataframe(df, csv_path)
    assert res_path == csv_path
    assert csv_path.exists()
    df_loaded_csv = load_dataframe(csv_path)
    pd.testing.assert_frame_equal(df_loaded_csv, df)

    # TXT format loading
    txt_path = tmp_path / "data.txt"
    df.to_csv(txt_path, index=False)
    df_loaded_txt = load_dataframe(txt_path)
    pd.testing.assert_frame_equal(df_loaded_txt, df)

    # Parquet in nested dir
    parquet_path = tmp_path / "nested2" / "data.parquet"
    save_dataframe(df, parquet_path)
    assert parquet_path.exists()
    df_loaded_parquet = load_dataframe(parquet_path)
    pd.testing.assert_frame_equal(df_loaded_parquet, df)


def test_save_dataframe_unsupported_format(tmp_path: Path):
    df = pd.DataFrame({"close": [1.0, 2.0]})
    bad_target = tmp_path / "data.xml"
    with pytest.raises(ValueError, match="Unsupported file format"):
        save_dataframe(df, bad_target)


def test_compute_signal_stats():
    df = pd.DataFrame(
        {
            "trend_sma_cross_signal": ["buy", "buy", "sell", "hold", "none"],
            "momentum_rsi_signal": ["buy", "buy", "buy", "buy", "sell"],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )
    stats = compute_signal_stats(df)

    assert "trend_sma_cross_signal" in stats
    assert "momentum_rsi_signal" in stats
    assert "close" not in stats

    assert stats["trend_sma_cross_signal"]["total"] == 5.0
    assert stats["trend_sma_cross_signal"]["buy_pct"] == 40.0
    assert stats["trend_sma_cross_signal"]["sell_pct"] == 20.0
    assert stats["trend_sma_cross_signal"]["hold_pct"] == 20.0
    assert stats["trend_sma_cross_signal"]["none_pct"] == 20.0

    assert stats["momentum_rsi_signal"]["total"] == 5.0
    assert stats["momentum_rsi_signal"]["buy_pct"] == 80.0
    assert stats["momentum_rsi_signal"]["sell_pct"] == 20.0
    assert stats["momentum_rsi_signal"]["hold_pct"] == 0.0
    assert stats["momentum_rsi_signal"]["none_pct"] == 0.0


def test_compute_signal_stats_empty_or_no_signals():
    df_no_signals = pd.DataFrame({"close": [100.0, 101.0]})
    assert compute_signal_stats(df_no_signals) == {}

    df_empty = pd.DataFrame(columns=["trend_signal"])
    assert compute_signal_stats(df_empty) == {
        "trend_signal": {
            "total": 0.0,
            "buy_pct": 0.0,
            "sell_pct": 0.0,
            "hold_pct": 0.0,
            "none_pct": 0.0,
        }
    }
