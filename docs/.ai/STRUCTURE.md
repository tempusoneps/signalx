# SignalX Repository Structure & File Responsibilities

This document provides a comprehensive breakdown of the directory layout and the single responsibility of each module in `signalx`.

```
signalx/
├── .superpowers/                 # Superpowers agent plan and design files
│   └── sdd/2026-08-28-signalx-library/
├── datasets/                     # Test datasets and sample OHLCV files
│   ├── sample_ohlcv.csv
│   ├── sample_ohlcv.parquet
│   └── sample_signals.parquet
├── docs/                         # Documentation root
│   ├── .ai/                      # AI Agent documentation source files
│   │   ├── AI_AGENT_GUIDELINE.md # Coding standards, commands, test performance rules
│   │   ├── CONFIGURATION.md      # CLI options, dataframe schemas, parameters
│   │   ├── RULE.md               # Core repository invariants and constraints
│   │   ├── SIGNALS_CATALOG.md    # Catalog of all 172 signals across 7 categories
│   │   ├── STRUCTURE.md          # Repository layout and module responsibilities
│   │   └── USAGE.md              # Python API & CLI usage examples
│   ├── superpowers/              # Spec and implementation plan documentation
│   └── README.md                 # Master documentation hub
├── scripts/                      # Developer and build automation scripts
│   ├── generate_agents_markdown.sh # Compiles docs/.ai/ into AGENTS.md, GEMINI.md, CLAUDE.md
│   ├── get_vn30f1m_5m_datasets.sh  # Downloads real VN30F1M 5m dataset from repository
│   ├── prepare_sample_dataset.py   # Generates realistic synthetic OHLCV datasets
│   └── run_signals.sh              # CLI runner script for signal extraction
├── src/                          # Main Python source package
│   └── signalx/
│       ├── __init__.py           # Library entrypoint, exports SignalState, generate_signals
│       ├── cli.py                # Command-line interface (generate, inspect, stats, list)
│       ├── constants.py          # Canonical signal states (buy, sell, hold, none)
│       ├── core.py               # Orchestration pipeline (generate_signals)
│       ├── metadata.py           # Signal catalog, category registries, lookup helpers
│       ├── utils.py              # OHLCV validation, column normalization, I/O, stats
│       └── signals/              # Category signal generator modules
│           ├── __init__.py       # Dispatches all category generators
│           ├── candlestick.py    # Candlestick geometry and price action patterns (27 signals)
│           ├── composite.py      # Consensus and ensemble voting signals (8 signals)
│           ├── momentum.py       # Oscillators and momentum indicators (30 signals)
│           ├── statistical.py    # Z-scores, linear regression, efficiency (16 signals)
│           ├── trend.py          # Moving averages, MACD, SuperTrend, ADX (41 signals)
│           ├── volatility.py     # Bollinger Bands, Donchian, Keltner, ATR (32 signals)
│           └── volume.py         # OBV, CMF, VWAP, Volume Spikes (18 signals)
├── tests/                        # Comprehensive unit and integration test suite
│   ├── test_cli.py               # Tests for CLI subcommands and flag parsing
│   ├── test_constants.py         # Tests for SignalState and ALL_SIGNAL_STATES
│   ├── test_data_validation.py   # Tests for normalize_ohlcv and load/save helpers
│   ├── test_full_pipeline.py     # End-to-end pipeline and integration tests
│   ├── test_metadata.py          # Tests for metadata registration and queries
│   ├── test_signals_candlestick.py # Tests for candlestick signals
│   ├── test_signals_composite.py   # Tests for composite signals
│   ├── test_signals_momentum.py    # Tests for momentum signals
│   ├── test_signals_statistical.py # Tests for statistical signals
│   ├── test_signals_trend.py       # Tests for trend signals
│   ├── test_signals_volatility.py  # Tests for volatility signals
│   └── test_signals_volume.py      # Tests for volume signals
├── AGENTS.md                     # Compiled agent instructions (auto-generated)
├── CLAUDE.md                     # Compiled Claude instructions (auto-generated)
├── GEMINI.md                     # Compiled Gemini instructions (auto-generated)
├── pyproject.toml                # Project configuration, dependencies, build settings
└── README.md                     # User-facing library overview and documentation
```

---

## Detailed Module Responsibilities

### Core Library (`src/signalx/`)
- `constants.py`: Holds `SignalState` class with strings `"buy"`, `"sell"`, `"hold"`, `"none"` and `ALL_SIGNAL_STATES` frozenset.
- `utils.py`: Provides input data sanitation (`normalize_ohlcv`), case-insensitive column aliasing, DataFrame I/O (`load_dataframe`, `save_dataframe`), and frequency metrics (`compute_signal_stats`).
- `metadata.py`: Implements `SignalMetadata` data structures, master registries `SIGNAL_CATALOG` and `SIGNAL_CODE_CATALOG` containing all 172 signals with trigger rules, lookup functions (`get_signal_by_code`, `get_signal_by_name`, `get_signal_metadata`), and bidirectional DataFrame column renaming utilities (`to_code_names`, `to_semantic_names`, `get_code_to_name_map`, `get_name_to_code_map`).
- `core.py`: Exposes `generate_signals(df, drop_ohlcv=False, show_progress=False, naming="code")` which coordinates normalization, dispatches category signal extractors, applies requested naming convention (`"code"` or `"semantic"`), and aggregates results.
- `cli.py`: Implements the `signalx` command-line executable using `argparse` (subcommands: `generate`, `inspect`, `stats`, `list`).

### Signal Generators (`src/signalx/signals/`)
- `trend.py`: Moving average crossovers (SMA, EMA, DEMA, TEMA, HMA, VWMA), MACD variants, SuperTrend, Parabolic SAR, Aroon, ADX/DMI, Ichimoku Cloud, Vortex, TRIX, KAMA, TMA, MSB, MA Alignment, Pullback, Micro Trend/Reversal, TII.
- `momentum.py`: RSI multi-period, Stochastics, StochRSI, Williams %R, CCI, ROC, MFI, TSI, Fisher Transform, Awesome Oscillator, Ultimate Oscillator, CMO, Connors RSI, RSI Divergence, MFI Reversal, Momentum Shift.
- `volatility.py`: Bollinger Bands breakouts/bounces, %B reversals, Donchian Channels, Keltner Channels, TTM Squeeze, Bandwidth Expansion, ATR Trailing Stops, Chaikin Volatility, Historical Volatility Ratio, BB Rejection, Compression Breakouts, LinReg Channels, Envelopes.
- `volume.py`: On-Balance Volume (OBV), Chaikin Money Flow (CMF), Rolling VWAP crossovers & standard deviation bands, Volume Spikes with directional candles, PVT, ADL, Force Index, Ease of Movement (EOM), VSA Confirmation, VPT Divergence, Volume Trends.
- `candlestick.py`: Engulfing, Hammer, Inverted Hammer, Shooting Star, Hanging Man, Pinbar, Marubozu, Harami, Inside Bar, Outside Bar, Doji, Three White Soldiers / Black Crows, Consecutive 3/5, Morning/Evening Star, Piercing Line / Dark Cloud, Tweezer Tops/Bottoms, Couple Candlestick, Fakey Pattern, Liquidity Sweeps, Gap Up/Down.
- `statistical.py`: Rolling Price Z-Scores, Rolling Return Z-Scores, Kaufman Efficiency Ratio (KER), Choppiness Index, Rolling Quantile Extremes, Linear Regression Slope & Price Cross, MA Stretch Z-Score, Hurst Proxy, Range Mid Reversion, Price Acceleration.
- `composite.py`: Family consensus signals (Trend, Momentum, MA), Master Ensemble (weighted multi-indicator), Trend-Momentum Alignment, Breakout + Volume confirmation, Multi-oscillator mean reversion confluence, MACD Hist + Candlestick confluence.
- `__init__.py`: Aggregates all category functions into `run_all_signal_generators(df)`.
