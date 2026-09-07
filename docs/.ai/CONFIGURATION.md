# SignalX Configuration & Data Schemas

This document defines the configuration options, parameters, and input/output data schemas supported by `signalx`.

---

## 1. Input DataFrame Schema Requirements

`signalx` accepts any standard `pandas.DataFrame` representing time-series bar data. Input data is automatically normalized by `signalx.utils.normalize_ohlcv`.

### Required Columns (Case-Insensitive)
| Canonical Name | Accepted Aliases | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `open` | `Open`, `OPEN`, `o`, `O` | `float64` | Bar open price |
| `high` | `High`, `HIGH`, `h`, `H` | `float64` | Bar high price |
| `low` | `Low`, `LOW`, `l`, `L` | `float64` | Bar low price |
| `close` | `Close`, `CLOSE`, `c`, `C` | `float64` | Bar close price |
| `volume` | `Volume`, `VOLUME`, `vol`, `Vol`, `VOL`, `v`, `V` | `float64` / `int64` | Bar traded volume |

### Optional Columns
| Column Pattern | Description | Handling |
| :--- | :--- | :--- |
| `date`, `datetime`, `timestamp`, `time` | Bar timestamp | Preserved in output when `drop_ohlcv=True` or `drop_ohlcv=False`. |
| Any other custom column | User features (e.g. `symbol`, `ticker`) | Retained in output DataFrame when `drop_ohlcv=False`. |

### Validation Invariants
- `high` must be $\ge$ `low` across all valid rows.
- Data with $< 1$ row raises `ValueError("Input DataFrame is empty")`.
- Missing required OHLCV columns raise `ValueError("Missing required OHLCV column: ...")`.

---

## 2. Python API Parameters

### `signalx.generate_signals(df, drop_ohlcv=False, show_progress=False, naming="code")`
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `df` | `pandas.DataFrame` | *Required* | Raw OHLCV DataFrame. |
| `drop_ohlcv` | `bool` | `False` | When `True`, returns only the 251 signal columns (and timestamp column if present). When `False`, returns original columns concatenated with the 251 signal columns. |
| `show_progress` | `bool` | `False` | When `True`, displays real-time multi-progress bars per signal category in terminal. |
| `naming` | `Literal["code", "semantic"]` | `"code"` | Output column naming format. `"code"` produces compact standardized identifiers (e.g. `TRD001_signal`), while `"semantic"` produces descriptive identifiers (e.g. `trend_sma_cross_5_20_signal`). |

**Return Value**: `pandas.DataFrame` containing all 251 signal columns with string states (`"buy"`, `"sell"`, `"hold"`, `"none"`).

---

## 3. CLI Subcommand Options

`signalx` CLI is invoked via `signalx <command> [options]`.

### Subcommand: `generate`
Extracts all 251 trading signals from an input dataset file.
```bash
signalx generate <input_path> [-o <output_path>] [--drop-ohlcv] [--stats-report] [--no-progress] [--naming {code,semantic}]
```
| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `input_path` | | `Path` (Positional) | *Required* | Path to input CSV or Parquet file. |
| `--output` | `-o` | `Path` | `datasets/<stem>_signals.parquet` | Destination path for output dataset (Parquet or CSV). |
| `--drop-ohlcv` | | `flag` | `False` | Drop input OHLCV columns from the output file. |
| `--stats-report` | | `flag` | `False` | Print JSON signal state distribution summary to stdout. |
| `--no-progress` | | `flag` | `False` | Disable real-time per-group progress bars. |
| `--naming` | | `str` (`code`, `semantic`) | `code` | Output column naming format: `code` (e.g. `TRD001_signal`) or `semantic` (e.g. `trend_sma_cross_5_20_signal`). |

### Subcommand: `inspect`
Validates and displays summary statistics of an OHLCV dataset.
```bash
signalx inspect <input_path>
```
| Flag | Type | Description |
| :--- | :--- | :--- |
| `input_path` | `Path` (Positional) | Path to input CSV or Parquet file. |

### Subcommand: `stats`
Computes state frequency percentages across all `*_signal` columns.
```bash
signalx stats <signals_path> [--json]
```
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `signals_path` | `Path` (Positional) | *Required* | Path to dataset containing `*_signal` columns. |
| `--json` | `flag` | `False` | Print results in JSON format instead of human-readable table. |

### Subcommand: `list`
Lists registered signals, codes, and their descriptions.
```bash
signalx list [--category <category>]
```
| Flag | Type | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `--category` | `str` | `trend`, `momentum`, `volatility`, `volume`, `candlestick`, `smc`, `mean_reversion`, `statistical`, `composite` | Filter signal listing by category. |

---

## 4. Supported File Formats
- **Parquet (`.parquet`, `.pq`)**: Recommended for high performance and strict typing via `pyarrow`.
- **CSV (`.csv`)**: Supported with automatic header detection and float parsing.
