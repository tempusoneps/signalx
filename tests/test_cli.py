from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signalx import __version__
from signalx.cli import main
from signalx.metadata import VALID_CATEGORIES


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100, 1000, size=n)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def sample_csv_file(synthetic_ohlcv_df: pd.DataFrame, tmp_path: Path) -> Path:
    p = tmp_path / "sample_ohlcv.csv"
    synthetic_ohlcv_df.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_parquet_file(synthetic_ohlcv_df: pd.DataFrame, tmp_path: Path) -> Path:
    p = tmp_path / "sample_ohlcv.parquet"
    synthetic_ohlcv_df.to_parquet(p, index=False)
    return p


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert f"signalx {__version__}" in captured.out or f"signalx {__version__}" in captured.err


def test_cli_help_flag(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "generate" in captured.out
    assert "inspect" in captured.out
    assert "stats" in captured.out
    assert "list" in captured.out


def test_cli_no_args_shows_help(capsys: pytest.CaptureFixture[str]):
    main([])
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "generate" in captured.out


def test_cli_inspect_csv(sample_csv_file: Path, capsys: pytest.CaptureFixture[str]):
    main(["inspect", str(sample_csv_file)])
    captured = capsys.readouterr()
    assert "Total rows: 100" in captured.out
    assert "Date span: 2026-01-01" in captured.out
    assert "open" in captured.out
    assert "high" in captured.out
    assert "low" in captured.out
    assert "close" in captured.out
    assert "volume" in captured.out
    assert "Mean" in captured.out
    assert "Std" in captured.out
    assert "Min" in captured.out
    assert "Max" in captured.out


def test_cli_inspect_parquet_datetime_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    n = 30
    dates = pd.date_range("2026-02-01", periods=n, freq="1D")
    df = pd.DataFrame(
        {
            "open": np.linspace(10, 20, n),
            "high": np.linspace(11, 21, n),
            "low": np.linspace(9, 19, n),
            "close": np.linspace(10.5, 20.5, n),
            "volume": np.full(n, 500.0),
        },
        index=dates,
    )
    pq_path = tmp_path / "indexed.parquet"
    df.to_parquet(pq_path)

    main(["inspect", str(pq_path)])
    captured = capsys.readouterr()
    assert "Total rows: 30" in captured.out
    assert "Date span: 2026-02-01" in captured.out
    assert "Index" in captured.out or "2026-03-02" in captured.out


def test_cli_inspect_no_date_col(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    n = 20
    df = pd.DataFrame(
        {
            "open": np.linspace(10, 20, n),
            "high": np.linspace(11, 21, n),
            "low": np.linspace(9, 19, n),
            "close": np.linspace(10.5, 20.5, n),
            "volume": np.full(n, 500.0),
        }
    )
    csv_path = tmp_path / "nodate.csv"
    df.to_csv(csv_path, index=False)

    main(["inspect", str(csv_path)])
    captured = capsys.readouterr()
    assert "Total rows: 20" in captured.out
    assert "Date span: N/A" in captured.out


def test_cli_inspect_missing_columns(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"open": [1, 2], "close": [2, 3]}).to_csv(bad_csv, index=False)

    with pytest.raises(SystemExit) as exc_info:
        main(["inspect", str(bad_csv)])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Missing required OHLCV column" in captured.err


def test_cli_inspect_nonexistent_file(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        main(["inspect", "nonexistent_ohlcv.csv"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "File not found" in captured.err


def test_cli_generate_default_output(
    sample_csv_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.chdir(tmp_path)
    main(["generate", str(sample_csv_file)])
    captured = capsys.readouterr()
    assert "Successfully generated signals" in captured.out

    expected_output = tmp_path / f"datasets/{sample_csv_file.stem}_signals.parquet"
    assert expected_output.exists()
    df_out = pd.read_parquet(expected_output)
    assert len(df_out) == 100
    assert "trend_sma_cross_5_20_signal" in df_out.columns
    assert "open" in df_out.columns


def test_cli_generate_custom_output_and_drop_ohlcv(
    sample_parquet_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    custom_out = tmp_path / "output_signals.csv"
    main(["generate", str(sample_parquet_file), "-o", str(custom_out), "--drop-ohlcv"])
    captured = capsys.readouterr()
    assert "Successfully generated signals" in captured.out

    assert custom_out.exists()
    df_out = pd.read_csv(custom_out)
    assert len(df_out) == 100
    assert "trend_sma_cross_5_20_signal" in df_out.columns
    assert "open" not in df_out.columns
    assert "volume" not in df_out.columns
    assert "timestamp" in df_out.columns


def test_cli_generate_stats_report(
    sample_csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out_file = tmp_path / "out.parquet"
    main(["generate", str(sample_csv_file), "-o", str(out_file), "--stats-report"])
    captured = capsys.readouterr()
    assert "Successfully generated signals" in captured.out
    assert "Signal Distribution Summary:" in captured.out
    assert "trend_sma_cross_5_20_signal" in captured.out
    assert "Buy %" in captured.out
    assert "Sell %" in captured.out


def test_cli_stats_table(sample_csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    sig_path = tmp_path / "generated_signals.parquet"
    main(["generate", str(sample_csv_file), "-o", str(sig_path)])
    capsys.readouterr()  # clear buffer

    main(["stats", str(sig_path)])
    captured = capsys.readouterr()
    assert "Signal Statistics for" in captured.out
    assert "trend_sma_cross_5_20_signal" in captured.out
    assert "Buy %" in captured.out
    assert "Sell %" in captured.out
    assert "Hold %" in captured.out
    assert "None %" in captured.out


def test_cli_stats_json(sample_csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    sig_path = tmp_path / "generated_signals.parquet"
    main(["generate", str(sample_csv_file), "-o", str(sig_path)])
    capsys.readouterr()  # clear buffer

    main(["stats", str(sig_path), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    assert "trend_sma_cross_5_20_signal" in data
    assert "buy_pct" in data["trend_sma_cross_5_20_signal"]
    assert "sell_pct" in data["trend_sma_cross_5_20_signal"]
    assert "hold_pct" in data["trend_sma_cross_5_20_signal"]
    assert "none_pct" in data["trend_sma_cross_5_20_signal"]


def test_cli_stats_no_signals(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    no_sig_path = tmp_path / "no_sig.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(no_sig_path, index=False)

    main(["stats", str(no_sig_path)])
    captured = capsys.readouterr()
    assert "No signal columns found (*_signal)." in captured.out


def test_cli_stats_nonexistent_file(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        main(["stats", "nonexistent.parquet"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err


def test_cli_list_all(capsys: pytest.CaptureFixture[str]):
    main(["list"])
    captured = capsys.readouterr()
    for cat in VALID_CATEGORIES:
        assert f"Category: {cat.upper()}" in captured.out
    assert "trend_sma_cross_5_20_signal" in captured.out
    assert "Description:" in captured.out
    assert "Buy Trigger:" in captured.out
    assert "Sell Trigger:" in captured.out


def test_cli_list_category_filter(capsys: pytest.CaptureFixture[str]):
    main(["list", "--category", "momentum"])
    captured = capsys.readouterr()
    assert "Category: MOMENTUM" in captured.out
    assert "Category: TREND" not in captured.out
    assert "Category: VOLATILITY" not in captured.out
    assert "mom_rsi_ob_os_14_signal" in captured.out


def test_cli_list_invalid_category(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        main(["list", "--category", "nonexistent_category"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Invalid category" in captured.err


def test_cli_invalid_subcommand(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        main(["unknown_command"])
    assert exc_info.value.code == 2


def test_cli_missing_required_args(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        main(["generate"])
    assert exc_info.value.code == 2


def test_cli_main_module_execution():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "signalx.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"signalx {__version__}" in result.stdout or f"signalx {__version__}" in result.stderr


def test_cli_generate_no_progress_flag(
    sample_parquet_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out_file = tmp_path / "out_no_progress.parquet"
    main(["generate", str(sample_parquet_file), "-o", str(out_file), "--no-progress"])
    assert out_file.exists()
    captured = capsys.readouterr()
    assert "Successfully generated signals" in captured.out
