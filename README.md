# SignalX: Automated Quantitative Trading Signal Generation Library

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org)
[![Signals: 114](https://img.shields.io/badge/signals-114-brightgreen.svg)](#signals-catalog-overview)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

`signalx` is a modern, production-grade Python library designed to automatically extract **114 standardized trading signals** from any OHLCV (Open, High, Low, Close, Volume) dataset. 

Whether you are conducting quantitative market research, engineering features for machine learning models, or building algorithmic trading systems, `signalx` delivers a uniform, leak-free, zero-configuration signal generation pipeline.

---

## Key Features

- **114 Standardized Signals across 7 Families**: Covers Trend, Momentum & Oscillators, Volatility & Breakouts, Volume Dynamics, Candlestick Formations, Statistical Metrics, and Multi-Indicator Composite Ensembles.
- **Strict 4-State String Representation**: Every single signal value strictly resolves to one of four canonical states: `"buy"`, `"sell"`, `"hold"`, or `"none"`. No inconsistent booleans, integers, or float scales.
- **Deterministic Column Naming**: 100% of generated signal column names end with the suffix `_signal` (e.g. `trend_macd_cross_signal`, `mom_rsi_ob_os_14_signal`, `comp_master_ensemble_signal`), allowing instant regex matching and clean DataFrame partitioning.
- **Zero Future Leakage**: Every calculation strictly adheres to causality — computations at bar $t$ use only past and current information ($\le t$).
- **High-Performance Vectorization**: Built on top of `pandas`, `numpy`, `pyarrow`, `ta`, `pandas_ta`, `scipy`, and `statsmodels`.
- **Integrated CLI Application**: Full-featured command-line tool with subcommands `generate`, `inspect`, `stats`, and `list`.
- **AI Agent Synchronization**: AI documentation framework in `docs/.ai/` auto-compiles directly into `AGENTS.md`, `GEMINI.md`, and `CLAUDE.md`.

---

## Installation

Install `signalx` using `uv` or `pip`:

```bash
# Using uv
uv add signalx

# Or using pip
pip install signalx
```

---

## Quickstart

### Python API

```python
import pandas as pd
import signalx

# 1. Load any OHLCV DataFrame (Parquet or CSV)
df = pd.read_parquet("datasets/sample_ohlcv.parquet")

# 2. Extract all 114 standardized trading signals in a single call
signals_df = signalx.generate_signals(df)

# 3. Filter and inspect the generated signal columns
signal_cols = [c for c in signals_df.columns if c.endswith("_signal")]
print(f"Successfully generated {len(signal_cols)} signal columns!")

# Inspect signals for the latest 5 bars
print(signals_df[["Close"] + signal_cols[:5]].tail())
```

### Command Line Interface (CLI)

```bash
# Generate signals and print signal state distribution stats
uv run signalx generate datasets/sample_ohlcv.parquet -o datasets/sample_signals.parquet --stats-report

# Inspect dataset columns and summary statistics
uv run signalx inspect datasets/sample_ohlcv.parquet

# Calculate buy/sell/hold/none state distribution for all signals
uv run signalx stats datasets/sample_signals.parquet

# List all 114 available signals
uv run signalx list
```

---

## Signals Catalog Overview

`signalx` provides 114 production-ready trading signals partitioned across 7 analytical families:

| Category | Signals Count | Primary Analytical Focus | Example Signals |
| :--- | :--- | :--- | :--- |
| **Trend** | 30 | Directional moving average crossovers, MACD variants, SuperTrend, Parabolic SAR, Aroon, ADX/DMI, Ichimoku Cloud | `trend_sma_cross_5_20_signal`, `trend_macd_cross_signal`, `trend_supertrend_10_3_signal`, `trend_ichimoku_cloud_breakout_signal` |
| **Momentum** | 23 | Oscillators, overbought/oversold boundaries, and speed of price change | `mom_rsi_ob_os_14_signal`, `mom_stoch_kd_cross_14_3_3_signal`, `mom_cci_100_14_signal`, `mom_ao_saucer_signal` |
| **Volatility** | 17 | Bollinger Bands, Donchian channels, Keltner channels, TTM Squeeze, and ATR Trailing Stops | `vol_bb_breakout_20_20_signal`, `vol_donchian_breakout_20_signal`, `vol_ttm_squeeze_signal`, `vol_atr_trailing_stop_2x_signal` |
| **Volume** | 12 | Volume dynamics, flow accumulation/distribution, VWAP crossovers, and Volume Spikes | `volume_obv_ema_cross_20_signal`, `volume_cmf_zero_cross_20_signal`, `volume_vwap_cross_20_signal`, `volume_spike_direction_20_signal` |
| **Candlestick** | 14 | Price action geometry, rejection wicks, and single/multi-bar reversal formations | `cdl_engulfing_signal`, `cdl_pinbar_signal`, `cdl_hammer_star_signal`, `cdl_morning_evening_star_signal` |
| **Statistical** | 11 | Rolling Z-scores, linear regression slope/crossings, and market efficiency filters | `stat_price_zscore_20_signal`, `stat_ker_trend_filter_10_signal`, `stat_chop_regime_14_signal`, `stat_linreg_slope_14_signal` |
| **Composite** | 7 | Category consensus voting, trend/momentum confluence, and multi-indicator ensembles | `comp_master_ensemble_signal`, `comp_trend_consensus_signal`, `comp_trend_momentum_align_signal`, `comp_breakout_volume_confirmed_signal` |
| **Total** | **114** | **Full Quantitative Feature Suite** | |

For the complete catalog with exact buy and sell trigger conditions, see [docs/.ai/SIGNALS_CATALOG.md](docs/.ai/SIGNALS_CATALOG.md).

---

## 4-State Signal Contract

All signal columns strictly return values from `signalx.constants.SignalState`:

```python
from signalx.constants import SignalState

print(SignalState.BUY)  # "buy"  -> Long entry / Bullish momentum / Upside breakout
print(SignalState.SELL)  # "sell" -> Short entry / Bearish momentum / Downside breakdown
print(SignalState.HOLD)  # "hold" -> Maintaining position / Established trend continuation
print(SignalState.NONE)  # "none" -> Neutral / Indeterminate / Warmup phase
```

---

## Python API Reference

### `signalx.generate_signals(df: pd.DataFrame, drop_ohlcv: bool = False, show_progress: bool = False) -> pd.DataFrame`
The primary pipeline execution function. Normalizes input columns, executes all 7 signal category generators, and compiles the result.

- `df`: Input `pandas.DataFrame` with Open, High, Low, Close, and Volume columns (case-insensitive).
- `drop_ohlcv`: When `False` (default), returns the original DataFrame concatenated with the 114 signal columns. When `True`, returns only date/datetime columns and signal columns.
- `show_progress`: When `True`, displays real-time per-group progress bars in the terminal. Default is `False`.

```python
import signalx
import pandas as pd

df = pd.read_parquet("datasets/sample_ohlcv.parquet")

# Retain original OHLCV columns + 114 signals
full_df = signalx.generate_signals(df)

# Return only 114 signals + Date
signals_only_df = signalx.generate_signals(df, drop_ohlcv=True)
```

### Signal Metadata Queries
```python
from signalx.metadata import (
    SIGNAL_CATALOG,
    get_signal_metadata,
    get_signals_by_category,
    list_categories,
)

# List all categories
categories = list_categories()

# Retrieve all signals in the Volatility family
vol_signals = get_signals_by_category("volatility")

# Lookup metadata for a specific signal
meta = get_signal_metadata("vol_ttm_squeeze_signal")
print(meta.name)  # "vol_ttm_squeeze_signal"
print(meta.description)  # "TTM Squeeze breakout..."
print(meta.buy_trigger)  # "Squeeze fires and price is above 20-SMA baseline"
print(meta.sell_trigger)  # "Squeeze fires and price is below 20-SMA baseline"
```

### Distribution Statistics
```python
from signalx.utils import compute_signal_stats

stats = compute_signal_stats(signals_df)
print(stats["comp_master_ensemble_signal"])
# Output: {'buy_pct': 12.4, 'sell_pct': 10.8, 'hold_pct': 0.0, 'none_pct': 76.8}
```

---

## CLI Reference

`signalx` includes a CLI accessible directly from your terminal:

```bash
# General help
uv run signalx --help
```

### 1. `generate`
Generates signals for an input dataset.
```bash
uv run signalx generate <input_path> [-o <output_path>] [--drop-ohlcv] [--stats-report]
```
- `<input_path>`: Path to input CSV or Parquet file.
- `-o`, `--output`: Custom output path (defaults to `datasets/<name>_signals.parquet`).
- `--drop-ohlcv`: Output only the signal columns.
- `--stats-report`: Print JSON signal state distribution summary.

### 2. `inspect`
Inspects and validates an OHLCV dataset.
```bash
uv run signalx inspect datasets/sample_ohlcv.parquet
```

### 3. `stats`
Calculates frequency metrics for all `*_signal` columns.
```bash
# Formatted table
uv run signalx stats datasets/sample_signals.parquet

# JSON output
uv run signalx stats datasets/sample_signals.parquet --json
```

### 4. `list`
Lists registered signals and descriptions.
```bash
# List all 114 signals
uv run signalx list

# Filter by category
uv run signalx list --category composite
```

---

## Machine Learning & Backtesting Integration

`signalx` output columns can be directly utilized in downstream quantitative workflows:

```python
import signalx
import pandas as pd

# Load dataset and extract features
df = pd.read_parquet("datasets/sample_ohlcv.parquet")
signals_df = signalx.generate_signals(df)

# Filter all signal columns
signal_cols = [c for c in signals_df.columns if c.endswith("_signal")]

# One-hot encode signals for ML classification / regression models
ml_features = pd.get_dummies(signals_df[signal_cols], prefix=signal_cols)

# Map string states to numeric directional scores (-1, 0, 1)
numeric_scores = signals_df[signal_cols].replace(
    {
        "buy": 1,
        "sell": -1,
        "hold": 0,
        "none": 0,
    }
)

# Composite long score
total_bullish_score = (numeric_scores == 1).sum(axis=1)
```

---

## Development & AI Documentation

```bash
# Run the test suite (175 tests in < 10s)
uv run pytest -v

# Run code linter
uv run ruff check .

# Generate sample synthetic datasets
uv run python scripts/prepare_sample_dataset.py

# Recompile AI Agent guides (AGENTS.md, GEMINI.md, CLAUDE.md)
bash scripts/generate_agents_markdown.sh
```

---

## License

This project is licensed under the MIT License.
