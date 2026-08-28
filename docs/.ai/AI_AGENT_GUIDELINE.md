# SignalX AI Agent Guidelines

## 1. Overview & Mission
`signalx` is a quantitative finance and algorithmic trading Python library designed to extract 100+ standardized trading signal columns from any generic OHLCV (Open, High, Low, Close, Volume) dataset.

The core mission of `signalx` is to provide quantitative analysts, machine learning pipelines, and algorithmic trading systems with clean, leak-free, standardized technical and statistical features.

---

## 2. Core Architectural Principles

1. **Standardized State Contract**:
   Every signal column produces values strictly from `signalx.constants.SignalState`:
   - `"buy"`: Long entry / bullish momentum / breakout.
   - `"sell"`: Short entry / bearish momentum / breakdown.
   - `"hold"`: Continuing in established trend / holding position.
   - `"none"`: Neutral / sideways / indeterminate / warmup period.

2. **Column Naming Postfix**:
   100% of generated signal columns must end with `_signal`.
   Naming schema: `<category>_<indicator_name>_<params>_signal`.

3. **Zero Future Leakage (No Lookahead Bias)**:
   Calculations at bar $t$ must strictly depend only on information available at or before bar $t$. Shift operations or forward-looking rollings are strictly forbidden in signal calculation logic.

4. **Single Unified Pipeline**:
   `signalx.generate_signals(df)` executes all 7 signal categories (`trend`, `momentum`, `volatility`, `volume`, `candlestick`, `statistical`, `composite`) with zero configuration required.

5. **Sub-second Test Suite Execution**:
   All unit and integration tests must run against lightweight synthetic datasets (< 500 rows) to keep the entire test suite fast and deterministic.

---

## 3. Development Workflow & Commands

All development tasks should use `uv` for dependency management, testing, and linting.

### Environment & Dependencies
```bash
# Sync dependencies
uv sync

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=src/signalx -v

# Run code linter and formatting checks
uv run ruff check .
uv run ruff format --check .
```

### CLI Execution
```bash
# Generate trading signals
uv run signalx generate datasets/sample_ohlcv.parquet -o datasets/sample_signals.parquet --stats-report

# Inspect an OHLCV dataset
uv run signalx inspect datasets/sample_ohlcv.parquet

# Calculate signal state distributions
uv run signalx stats datasets/sample_signals.parquet

# List all available signals or filter by category
uv run signalx list --category trend
```

### AI Documentation Compilation
Whenever documentation in `docs/.ai/` is updated, run:
```bash
bash scripts/generate_agents_markdown.sh
```
This re-compiles `AGENTS.md`, `GEMINI.md`, and `CLAUDE.md`.

---

## 4. Coding Standards & Conventions

- **Python Version**: Python >= 3.12.
- **Typing**: Strict type annotations on all function signatures (`from __future__ import annotations`).
- **DataFrames**: Use `pandas.DataFrame` and `pandas.Series` vectorized operations. Avoid slow row-by-row `iterrows()` iterations.
- **Missing Data & Warmup**: First $N$ warmup bars before an indicator converges must produce `"none"` state (or appropriate fallback), never `NaN` or unhandled exceptions.
- **Normalization**: Always pass raw DataFrames through `normalize_ohlcv(df)` before running indicator calculations.
