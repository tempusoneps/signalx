# SignalX Usage Guide & Examples

This document demonstrates comprehensive usage patterns for both the Python API and the Command Line Interface (CLI).

---

## 1. Python API Usage

### Quickstart
```python
import pandas as pd
import signalx

# 1. Load your OHLCV data
df = pd.read_parquet("datasets/sample_ohlcv.parquet")

# 2. Extract all 114 standardized signals
signals_df = signalx.generate_signals(df)

# 3. View extracted signal columns
signal_cols = [col for col in signals_df.columns if col.endswith("_signal")]
print(f"Generated {len(signal_cols)} signal columns.")
print(signals_df[signal_cols].head())
```

---

### Filtering and Working with Signal States

All signals return one of four strings: `"buy"`, `"sell"`, `"hold"`, `"none"`.

```python
from signalx.constants import SignalState

# Find all bars where Master Ensemble emitted a BUY signal
ensemble_buys = signals_df[signals_df["comp_master_ensemble_signal"] == SignalState.BUY]
print(f"Ensemble Buy triggers: {len(ensemble_buys)}")

# Find bars where RSI oversold is triggered
rsi_buys = signals_df[signals_df["mom_rsi_ob_os_14_signal"] == SignalState.BUY]

# Filter signals by category using column prefixes
trend_signals = signals_df.filter(regex=r"^trend_.*_signal$")
volatility_signals = signals_df.filter(regex=r"^vol_.*_signal$")
composite_signals = signals_df.filter(regex=r"^comp_.*_signal$")
```

---

### Signal Distribution & Statistics

```python
from signalx.utils import compute_signal_stats

# Compute frequency breakdown of buy/sell/hold/none across all signals
stats = compute_signal_stats(signals_df)

for sig_name, stat in list(stats.items())[:5]:
    print(
        f"{sig_name}: Buy={stat['buy_pct']}% | Sell={stat['sell_pct']}% | Hold={stat['hold_pct']}%"
    )
```

---

### Querying Signal Metadata

```python
from signalx.metadata import (
    SIGNAL_CATALOG,
    get_signal_metadata,
    get_signals_by_category,
    list_categories,
)

# List all available categories
print("Categories:", list_categories())

# Inspect all Trend signals
trend_meta = get_signals_by_category("trend")
print(f"Total Trend Signals: {len(trend_meta)}")

# Look up a specific signal's trigger logic
meta = get_signal_metadata("trend_golden_cross_50_200_signal")
if meta:
    print(f"Name: {meta.name}")
    print(f"Description: {meta.description}")
    print(f"Buy Trigger: {meta.buy_trigger}")
    print(f"Sell Trigger: {meta.sell_trigger}")
```

---

### Machine Learning Feature Engineering Integration

Converting categorical string signals into one-hot encoded or numeric features:

```python
import pandas as pd
import signalx

df = pd.read_parquet("datasets/sample_ohlcv.parquet")
signals_df = signalx.generate_signals(df, drop_ohlcv=False)

# One-hot encode all signal columns
signal_cols = [c for c in signals_df.columns if c.endswith("_signal")]
encoded_features = pd.get_dummies(signals_df[signal_cols], prefix=signal_cols)

# Or map states to numeric values (-1, 0, 1)
state_mapping = {
    "buy": 1,
    "sell": -1,
    "hold": 0,
    "none": 0,
}
numeric_signals = signals_df[signal_cols].replace(state_mapping)
```

---

## 2. Command Line Interface (CLI) Usage

### Generating Signals from File
```bash
# Generate signals and output to default path datasets/<name>_signals.parquet
uv run signalx generate datasets/sample_ohlcv.parquet

# Specify custom output destination and print distribution statistics
uv run signalx generate datasets/sample_ohlcv.parquet \
  -o datasets/sample_signals.parquet \
  --stats-report

# Output only the 114 signal columns (drop OHLCV price columns)
uv run signalx generate datasets/sample_ohlcv.csv \
  -o datasets/signals_only.parquet \
  --drop-ohlcv
```

### Inspecting Datasets
```bash
# Validate column schemas and inspect price distributions
uv run signalx inspect datasets/sample_ohlcv.parquet
```

### Analyzing Signal Frequency
```bash
# Print formatted human-readable signal distribution table
uv run signalx stats datasets/sample_signals.parquet

# Print JSON formatted distribution
uv run signalx stats datasets/sample_signals.parquet --json
```

### Listing Signal Catalog
```bash
# List all 114 signals grouped by category
uv run signalx list

# Filter listing to a specific category
uv run signalx list --category composite
```
