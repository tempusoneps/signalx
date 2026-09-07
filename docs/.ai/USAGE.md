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

# 2. Extract all 264 standardized signals in coded format (default)
signals_df = signalx.generate_signals(df, show_progress=True)

# 3. View extracted signal columns (e.g. TRD001_signal, MOM001_signal, CMP003_signal)
signal_cols = [col for col in signals_df.columns if col.endswith("_signal")]
print(f"Generated {len(signal_cols)} coded signal columns.")
print(signals_df[signal_cols].head())

# Alternatively, extract signals with descriptive semantic names
semantic_df = signalx.generate_signals(df, naming="semantic")
print(semantic_df.filter(regex=r"^trend_.*_signal$").head())
```

---

### Bidirectional Column Renaming Helpers

You can seamlessly convert DataFrame column names between coded and semantic formats without recalculating indicators:

```python
from signalx import (
    get_code_to_name_map,
    get_name_to_code_map,
    to_code_names,
    to_semantic_names,
)

# Convert coded signal columns to semantic names
semantic_df = to_semantic_names(signals_df)
# TRD001_signal -> trend_sma_cross_5_20_signal

# Convert semantic signal columns back to coded format
coded_df = to_code_names(semantic_df)
# trend_sma_cross_5_20_signal -> TRD001_signal

# Access complete mappings directly
code_to_name = get_code_to_name_map()
print(f"TRD004_signal -> {code_to_name['TRD004_signal']}")

name_to_code = get_name_to_code_map()
print(f"trend_golden_cross_50_200_signal -> {name_to_code['trend_golden_cross_50_200_signal']}")
```

---

### Filtering and Working with Signal States

All signals return one of four strings: `"buy"`, `"sell"`, `"hold"`, `"none"`.

```python
from signalx.constants import SignalState

# Find all bars where Master Ensemble (CMP003_signal) emitted a BUY signal
ensemble_buys = signals_df[signals_df["CMP003_signal"] == SignalState.BUY]
print(f"Ensemble Buy triggers: {len(ensemble_buys)}")

# Find bars where RSI 14 (MOM001_signal) oversold is triggered
rsi_buys = signals_df[signals_df["MOM001_signal"] == SignalState.BUY]

# Filter signals by category using column prefixes (coded format)
trend_signals = signals_df.filter(regex=r"^TRD\d{3}_signal$")
volatility_signals = signals_df.filter(regex=r"^VOL\d{3}_signal$")
smc_signals = signals_df.filter(regex=r"^SMC\d{3}_signal$")
composite_signals = signals_df.filter(regex=r"^CMP\d{3}_signal$")
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
    SIGNAL_CODE_CATALOG,
    get_signal_by_code,
    get_signal_by_name,
    get_signal_metadata,
    get_signals_by_category,
    list_categories,
)

# List all available categories
print("Categories:", list_categories())

# Inspect all Trend signals
trend_meta = get_signals_by_category("trend")
print(f"Total Trend Signals: {len(trend_meta)}")

# Look up a specific signal by code or semantic name
meta = get_signal_metadata(
    "TRD004_signal"
)  # or get_signal_metadata("trend_golden_cross_50_200_signal")
if meta:
    print(f"Code: {meta.code}")
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
# Generate coded signals (default) and output to default path datasets/<name>_signals.parquet
uv run signalx generate datasets/sample_ohlcv.parquet

# Generate signals with semantic naming
uv run signalx generate datasets/sample_ohlcv.parquet --naming semantic

# Specify custom output destination and print distribution statistics
uv run signalx generate datasets/sample_ohlcv.parquet \
  -o datasets/sample_signals.parquet \
  --stats-report

# Output only the 264 signal columns (drop OHLCV price columns)
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
# List all 264 signals with codes, names, and descriptions
uv run signalx list

# Filter listing to a specific category
uv run signalx list --category mean_reversion
```
