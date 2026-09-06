# SignalX Architecture Rules & Invariants

This document establishes the mandatory architectural rules and invariant constraints of the `signalx` repository. All contributors and AI agents must strictly adhere to these rules.

---

## Rule 1: Suffix & Naming Rule (`_signal`)
- **Requirement**: 100% of generated signal column names must end with the postfix `_signal`.
- **Formats**:
  - **Coded Format (Default)**: `<CODE>_signal` where `<CODE>` is a 3-letter uppercase category code followed by a 3-digit zero-padded index (e.g., `TRD001_signal`, `MOM001_signal`, `CMP008_signal`).
  - **Semantic Format**: `<category>_<indicator_name>_<params>_signal` (e.g., `trend_sma_cross_5_20_signal`, `comp_master_ensemble_signal`).
- **Rationale**: Enables seamless regular expression matching (e.g., `df.filter(regex=r"_signal$")` or `df.filter(regex=r"^TRD\d{3}_signal$")`), prevents name collisions with raw price/indicator columns, and allows deterministic pipeline operations.

---

## Rule 2: 4-State String Representation Contract
- **Requirement**: Every signal value in every signal column must strictly be one of the four canonical lowercase string constants defined in `signalx.constants.SignalState`:
  - `"buy"` (Long Entry / Bullish Momentum / Positive Breakout)
  - `"sell"` (Short Entry / Bearish Momentum / Negative Breakdown)
  - `"hold"` (Consolidation / Maintaining Position / Continuation)
  - `"none"` (Neutral / Indeterminate / Warmup Phase)
- **Prohibited**: Numeric outputs (`1`, `-1`, `0`), boolean flags (`True`, `False`), capitalized strings (`"BUY"`, `"SELL"`), or `NaN` / `None` values in output signal columns.

---

## Rule 3: Zero Future Leakage (Lookahead Protection)
- **Requirement**: At any time step $t$, signal calculations must strictly use data from index $\le t$.
- **Prohibited**:
  - No negative shifts (`df['close'].shift(-1)`).
  - No centered rolling windows (`df['close'].rolling(20, center=True)`).
  - No global dataset-wide scaling (e.g. global mean/std normalization across entire time series) without expanding or rolling windows.

---

## Rule 4: Unified Single-Pipeline Execution
- **Requirement**: Calling `signalx.generate_signals(df)` must run all 8 signal categories (`trend`, `momentum`, `volatility`, `volume`, `candlestick`, `smc`, `statistical`, `composite`) without requiring manual multi-stage wiring.
- **Behavior**:
  - `naming="code"` (default): Returns compact coded column names (`TRD001_signal` ... `CMP008_signal`).
  - `naming="semantic"`: Returns verbose descriptive column names (`trend_sma_cross_5_20_signal` ...).
  - `drop_ohlcv=False` (default): Appends all signal columns to the original DataFrame while preserving the original index and columns.
  - `drop_ohlcv=True`: Returns only the signal columns (along with Date/Datetime index/columns if present).

---

## Rule 5: Unit Test Speed & Synthetic Fixtures
- **Requirement**: Unit tests must execute in sub-second timeframes per test module.
- **Implementation**:
  - All test fixtures must use synthetic, deterministically generated OHLCV DataFrames with length $< 500$ bars.
  - Tests must never make network requests, download remote files, or depend on multi-megabyte external historical files.

---

## Rule 6: Type Safety, Linting & Python 3.12+ Standards
- **Requirement**:
  - Strict type annotations on all public and internal interfaces.
  - 100% compliance with Ruff rules (`ruff check .` with zero errors).
  - Use Python 3.12+ features (modern union syntax `int | None`, standard generics `list[str]`, etc.).

---

## Rule 7: Strict Category Partitioning
- **Requirement**: Every signal must belong to exactly one of the 8 valid categories:
  1. `trend` (Trend-following & Moving Averages) (52 signals)
  2. `momentum` (Oscillators & Speed of Price Change) (39 signals)
  3. `volatility` (Bands, Envelopes & Breakouts) (44 signals)
  4. `volume` (Volume Dynamics & Flow Accumulation) (32 signals)
  5. `candlestick` (Price Action Geometry & Multi-Bar Formations) (28 signals)
  6. `smc` (Smart Money Concepts & Structural Order Flow) (11 signals)
  7. `statistical` (Z-Scores, Regressions & Regime Filters) (20 signals)
  8. `composite` (Consensus, Confluence & Ensemble Voting) (13 signals)
