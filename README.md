# SignalX: Automated Quantitative Trading Signal Generation Library

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org)
[![Signals: 230](https://img.shields.io/badge/signals-230-brightgreen.svg)](#signals-catalog-overview)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

`signalx` is a modern, production-grade Python library designed to automatically extract **230 standardized trading signals** from any OHLCV (Open, High, Low, Close, Volume) dataset. 

Whether you are conducting quantitative market research, engineering features for machine learning models, or building algorithmic trading systems, `signalx` delivers a uniform, leak-free, zero-configuration signal generation pipeline.

---

## Key Features

- **230 Standardized Signals across 7 Families**: Covers Trend (53), Momentum & Oscillators (38), Volatility & Breakouts (42), Volume Dynamics (28), Candlestick Formations (37), Statistical Metrics (20), and Multi-Indicator Composite Ensembles (12).
- **Strict 4-State String Representation**: Every single signal value strictly resolves to one of four canonical states: `"buy"`, `"sell"`, `"hold"`, or `"none"`. No inconsistent booleans, integers, or float scales.
- **Deterministic Column Naming**: 100% of generated signal column names end with the suffix `_signal`. Defaults to compact coded identifiers (e.g. `TRD001_signal`, `MOM001_signal`, `CMP003_signal`), with full support for verbose semantic names (e.g. `trend_sma_cross_5_20_signal`) and zero-cost bidirectional column conversion.
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

# 2. Extract all 230 standardized trading signals in coded format (default)
signals_df = signalx.generate_signals(df)

# 3. Filter and inspect the generated signal columns (TRD001_signal, MOM001_signal, etc.)
signal_cols = [c for c in signals_df.columns if c.endswith("_signal")]
print(f"Successfully generated {len(signal_cols)} coded signal columns!")
print(signals_df[["Close"] + signal_cols[:5]].tail())

# 4. Optional: Generate semantic descriptive column names
semantic_df = signalx.generate_signals(df, naming="semantic")

# 5. Seamlessly convert between coded and semantic column formats
from signalx import to_code_names, to_semantic_names

semantic_from_coded = to_semantic_names(signals_df)  # TRD001_signal -> trend_sma_cross_5_20_signal
coded_from_semantic = to_code_names(semantic_df)  # trend_sma_cross_5_20_signal -> TRD001_signal
```

### Command Line Interface (CLI)

```bash
# Generate coded signals and print distribution statistics
uv run signalx generate datasets/sample_ohlcv.parquet -o datasets/sample_signals.parquet --stats-report

# Generate signals with semantic naming format
uv run signalx generate datasets/sample_ohlcv.parquet --naming semantic

# Inspect dataset columns and summary statistics
uv run signalx inspect datasets/sample_ohlcv.parquet

# Calculate buy/sell/hold/none state distribution for all signals
uv run signalx stats datasets/sample_signals.parquet

# List all 230 available signals with codes and descriptions
uv run signalx list
```

---

## Signals Catalog Overview

`signalx` provides 230 production-ready trading signals partitioned across 7 analytical families:

| Category | Signals Count | Code Prefix | Primary Analytical Focus | Example Signals |
| :--- | :--- | :--- | :--- | :--- |
| **Trend** | 53 | `TRD` | Directional moving average crossovers, MACD variants, SuperTrend, Parabolic SAR, Aroon, ADX/DMI, Ichimoku Cloud, TRIX, KAMA, TMA | `TRD001_signal` (`trend_sma_cross_5_20_signal`), `TRD017_signal` (`trend_macd_cross_signal`), `TRD022_signal` (`trend_supertrend_10_3_signal`) |
| **Momentum** | 38 | `MOM` | Oscillators, overbought/oversold boundaries, Connors RSI, RSI divergence, MFI reversals | `MOM001_signal` (`mom_rsi_ob_os_14_signal`), `MOM007_signal` (`mom_stoch_kd_cross_14_3_3_signal`), `MOM012_signal` (`mom_cci_100_14_signal`) |
| **Volatility** | 42 | `VOL` | Bollinger Bands, Donchian channels, Keltner channels, TTM Squeeze, ATR Trailing Stops, LinReg channels, Envelopes | `VOL001_signal` (`vol_bb_breakout_20_20_signal`), `VOL008_signal` (`vol_donchian_breakout_20_signal`), `VOL012_signal` (`vol_ttm_squeeze_signal`) |
| **Volume** | 28 | `VLM` | Volume dynamics, flow accumulation/distribution, VWAP crossovers, Volume Spikes, VSA, VPT divergence | `VLM001_signal` (`volume_obv_ema_cross_20_signal`), `VLM002_signal` (`volume_cmf_zero_cross_20_signal`), `VLM004_signal` (`volume_vwap_cross_20_signal`) |
| **Candlestick** | 37 | `CDL` | Price action geometry, rejection wicks, single/multi-bar reversal formations, couple patterns, liquidity sweeps | `CDL001_signal` (`cdl_engulfing_signal`), `CDL003_signal` (`cdl_pinbar_signal`), `CDL002_signal` (`cdl_hammer_star_signal`) |
| **Statistical** | 20 | `STA` | Rolling Z-scores, linear regression slope/crossings, market efficiency filters, MA stretch Z-score, Hurst proxy | `STA002_signal` (`stat_price_zscore_20_signal`), `STA006_signal` (`stat_ker_trend_filter_10_signal`), `STA008_signal` (`stat_chop_regime_14_signal`) |
| **Composite** | 12 | `CMP` | Category consensus voting, trend/momentum confluence, multi-indicator ensembles, MACD+Candlestick confluence | `CMP003_signal` (`comp_master_ensemble_signal`), `CMP001_signal` (`comp_trend_consensus_signal`), `CMP005_signal` (`comp_trend_momentum_align_signal`) |
| **Total** | **230** | | **Full Quantitative Feature Suite** | |

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

### `signalx.generate_signals(df: pd.DataFrame, drop_ohlcv: bool = False, show_progress: bool = False, naming: Literal["code", "semantic"] = "code") -> pd.DataFrame`
The primary pipeline execution function. Normalizes input columns, executes all 7 signal category generators, and compiles the result.

- `df`: Input `pandas.DataFrame` with Open, High, Low, Close, and Volume columns (case-insensitive).
- `drop_ohlcv`: When `False` (default), returns the original DataFrame concatenated with the 230 signal columns. When `True`, returns only date/datetime columns and signal columns.
- `show_progress`: When `True`, displays real-time per-group progress bars in the terminal. Default is `False`.
- `naming`: Output column naming format. `"code"` (default) generates compact coded columns (`TRD001_signal` ... `CMP012_signal`), `"semantic"` generates descriptive column names (`trend_sma_cross_5_20_signal` ...).

```python
import signalx
import pandas as pd

df = pd.read_parquet("datasets/sample_ohlcv.parquet")

# Retain original OHLCV columns + 230 coded signals
full_df = signalx.generate_signals(df)

# Return only 230 coded signals + Date
signals_only_df = signalx.generate_signals(df, drop_ohlcv=True)

# Generate with semantic column naming
semantic_df = signalx.generate_signals(df, naming="semantic")
```

### Bidirectional Column Conversion Utilities
```python
from signalx import (
    get_code_to_name_map,
    get_name_to_code_map,
    to_code_names,
    to_semantic_names,
)

# Convert DataFrame columns
semantic_df = to_semantic_names(coded_df)
coded_df = to_code_names(semantic_df)

# Get mapping dictionaries
code_map = get_code_to_name_map()  # {"TRD001_signal": "trend_sma_cross_5_20_signal", ...}
name_map = get_name_to_code_map()  # {"trend_sma_cross_5_20_signal": "TRD001_signal", ...}
```

### Signal Metadata Queries
```python
from signalx.metadata import (
    SIGNAL_CATALOG,
    SIGNAL_CODE_CATALOG,
    get_signal_by_code,
    get_signal_by_name,
    get_signal_metadata,
    get_signals_by_category,
    list_categories,
)

# List all categories
categories = list_categories()

# Retrieve all signals in the Volatility family
vol_signals = get_signals_by_category("volatility")

# Lookup metadata for a specific signal by code or semantic name
meta = get_signal_metadata("VOL012_signal")  # or get_signal_metadata("vol_ttm_squeeze_signal")
print(meta.code)  # "VOL012_signal"
print(meta.name)  # "vol_ttm_squeeze_signal"
print(
    meta.description
)  # "TTM Squeeze breakout: Bollinger Bands contract inside Keltner Channels then expand"
print(meta.buy_trigger)  # "Squeeze fires and price is above 20-SMA baseline"
print(meta.sell_trigger)  # "Squeeze fires and price is below 20-SMA baseline"
```

### Distribution Statistics
```python
from signalx.utils import compute_signal_stats

stats = compute_signal_stats(signals_df)
print(stats["CMP003_signal"])
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
uv run signalx generate <input_path> [-o <output_path>] [--drop-ohlcv] [--stats-report] [--no-progress] [--naming {code,semantic}]
```
- `<input_path>`: Path to input CSV or Parquet file.
- `-o`, `--output`: Custom output path (defaults to `datasets/<name>_signals.parquet`).
- `--drop-ohlcv`: Output only the signal columns.
- `--stats-report`: Print JSON signal state distribution summary.
- `--naming`: Output column format (`code` [default] or `semantic`).

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
Lists registered signals, codes, and descriptions.
```bash
# List all 230 signals
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
# Run the full test suite
uv run pytest -v

# Run code linter
uv run ruff check .

# Format code
uv run ruff format .

# Generate sample synthetic datasets
uv run python scripts/prepare_sample_dataset.py

# Recompile AI Agent guides (AGENTS.md, GEMINI.md, CLAUDE.md)
bash scripts/generate_agents_markdown.sh
```

---

## License

This project is licensed under the MIT License.
