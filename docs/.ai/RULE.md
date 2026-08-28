# SignalX Architecture Rules & Invariants

This document establishes the mandatory architectural rules and invariant constraints of the `signalx` repository. All contributors and AI agents must strictly adhere to these rules.

---

## Rule 1: Suffix Rule (`_signal`)
- **Requirement**: 100% of generated signal column names must end with the postfix `_signal`.
- **Format**: `<category>_<indicator_name>_<params>_signal`
- **Rationale**: Enables seamless regular expression matching (e.g., `df.filter(regex=r"_signal$")`), prevents name collisions with raw price/indicator columns, and allows deterministic pipeline operations.

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
- **Requirement**: Calling `signalx.generate_signals(df)` must run all 7 signal categories (`trend`, `momentum`, `volatility`, `volume`, `candlestick`, `statistical`, `composite`) without requiring manual multi-stage wiring.
- **Behavior**:
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
- **Requirement**: Every signal must belong to exactly one of the 7 valid categories:
  1. `trend` (Trend-following & Moving Averages)
  2. `momentum` (Oscillators & Speed of Price Change)
  3. `volatility` (Bands, Envelopes & Breakouts)
  4. `volume` (Volume Dynamics & Flow Accumulation)
  5. `candlestick` (Price Action Geometry & Multi-Bar Formations)
  6. `statistical` (Z-Scores, Regressions & Regime Filters)
  7. `composite` (Consensus, Confluence & Ensemble Voting)
