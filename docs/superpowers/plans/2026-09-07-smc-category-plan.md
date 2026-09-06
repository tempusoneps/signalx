# SMC (Smart Money Concepts) Category Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `smc` as the official 8th signal category in `signalx`, migrating 11 existing SMC signals into `src/signalx/signals/smc.py`, updating pipeline orchestration, metadata, CLI, test suites, and documentation.

**Architecture:** 
- Extract 10 signals from `candlestick.py` and 1 signal from `trend.py` into `src/signalx/signals/smc.py`.
- Map new coded identifiers `SMC001_signal` ... `SMC011_signal` and new semantic identifiers `smc_*_signal`.
- Update `candlestick.py` (38 -> 28 signals) and `trend.py` (53 -> 52 signals), keeping remaining numeric codes unchanged.
- Update `core.py` pipeline orchestration and `metadata.py` to support 8 categories.
- Update `composite.py` to reference `smc_*` / `SMC011_signal`.
- Add comprehensive `tests/test_signals_smc.py` and update documentation across `docs/.ai/`.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, ruff, uv.

## Global Constraints

- 100% of signal values must strictly adhere to `SignalState` strings: `"buy"`, `"sell"`, `"hold"`, `"none"`.
- 100% of generated signal column names must end with `_signal`.
- Zero future leakage (no negative shift, no lookahead bias).
- Unit tests must execute in sub-second times with synthetic fixtures (< 500 rows).
- Full compliance with `ruff check .` and `ruff format --check .`.
- Git commit message format: `<branch>(<next_tag|next_version>): <short summary in lowercase>`.

---

### Task 1: Create `src/signalx/signals/smc.py` and `tests/test_signals_smc.py`

**Files:**
- Create: `src/signalx/signals/smc.py`
- Create: `tests/test_signals_smc.py`

**Interfaces:**
- Produces: `generate_smc_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame`
- Produces: `SMC_SIGNAL_COLUMNS` (list of 11 semantic signal column names)

- [ ] **Step 1: Write the failing tests in `tests/test_signals_smc.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.smc import (
    SMC_SIGNAL_COLUMNS,
    generate_smc_signals,
    _calc_fvg_bullish_mitigation,
    _calc_fvg_bearish_mitigation,
    _calc_order_block_retest,
    _calc_break_of_structure,
    _calc_change_of_character,
    _calc_market_structure_break,
    _calc_liquidity_sweep,
    _calc_equal_high_low_sweep,
    _calc_judas_swing,
    _calc_inducement_sweep,
    _calc_pdh_pdl_sweep,
)


def make_synthetic_ohlcv(n: int = 150, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.015
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.random.uniform(0.002, 0.015, n))
    low = close * (1.0 - np.random.uniform(0.002, 0.015, n))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_smc_signals_all_11_columns_present():
    df = make_synthetic_ohlcv(100)
    res = generate_smc_signals(df)
    assert len(res.columns) == 11
    assert list(res.columns) == SMC_SIGNAL_COLUMNS


def test_smc_signals_all_states_valid():
    df = make_synthetic_ohlcv(100)
    res = generate_smc_signals(df)
    for col in res.columns:
        assert set(res[col].unique()).issubset(ALL_SIGNAL_STATES)


def test_smc_signals_empty_and_short_dataframe():
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_smc_signals(empty_df)
    assert len(res) == 0
    assert len(res.columns) == 11

    short_df = make_synthetic_ohlcv(3)
    res_short = generate_smc_signals(short_df)
    assert len(res_short) == 3
    assert len(res_short.columns) == 11


def test_smc_fvg_mitigation_direct():
    df = make_synthetic_ohlcv(60)
    sig_bull = _calc_fvg_bullish_mitigation(df["open"], df["high"], df["low"], df["close"])
    sig_bear = _calc_fvg_bearish_mitigation(df["open"], df["high"], df["low"], df["close"])
    assert set(sig_bull.unique()).issubset(ALL_SIGNAL_STATES)
    assert set(sig_bear.unique()).issubset(ALL_SIGNAL_STATES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_signals_smc.py -v`
Expected: ModuleNotFoundError or ImportError for `signalx.signals.smc`.

- [ ] **Step 3: Implement `src/signalx/signals/smc.py`**

Migrate the 11 functions from `candlestick.py` and `trend.py`:
- `_calc_fvg_bullish_mitigation`
- `_calc_fvg_bearish_mitigation`
- `_calc_order_block_retest`
- `_calc_break_of_structure`
- `_calc_change_of_character`
- `_calc_market_structure_break`
- `_calc_liquidity_sweep`
- `_calc_equal_high_low_sweep`
- `_calc_judas_swing`
- `_calc_inducement_sweep`
- `_calc_pdh_pdl_sweep`
Implement `generate_smc_signals(df, show_progress=False) -> pd.DataFrame` with `tqdm` progress support.
Define `SMC_SIGNAL_COLUMNS = [...]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_signals_smc.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/signalx/signals/smc.py tests/test_signals_smc.py
git commit -m "develop(v0.1.0): implement smc signal generator module and test suite"
```

---

### Task 2: Remove Migrated Signals from `candlestick.py` and `trend.py`

**Files:**
- Modify: `src/signalx/signals/candlestick.py`
- Modify: `src/signalx/signals/trend.py`
- Modify: `tests/test_signals_candlestick.py`
- Modify: `tests/test_signals_trend.py`

- [ ] **Step 1: Update `candlestick.py`**
  - Remove the 10 migrated functions.
  - Remove the 10 column names from `CANDLESTICK_SIGNAL_COLUMNS` (down to 28 columns).
  - Remove calls inside `generate_candlestick_signals`.

- [ ] **Step 2: Update `trend.py`**
  - Remove `_calc_market_structure_break`.
  - Remove `trend_market_structure_break_signal` from `TREND_SIGNAL_COLUMNS` (down to 52 columns).
  - Remove call inside `generate_trend_signals`.

- [ ] **Step 3: Update `tests/test_signals_candlestick.py` and `tests/test_signals_trend.py`**
  - Update expected column count in `test_signals_candlestick.py` from 38 to 28.
  - Update expected column count in `test_signals_trend.py` from 53 to 52.

- [ ] **Step 4: Run test suites**

Run: `uv run pytest tests/test_signals_candlestick.py tests/test_signals_trend.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/signalx/signals/candlestick.py src/signalx/signals/trend.py tests/test_signals_candlestick.py tests/test_signals_trend.py
git commit -m "develop(v0.1.0): remove migrated smc signals from candlestick and trend modules"
```

---

### Task 3: Update `composite.py`, `signals/__init__.py`, and `core.py`

**Files:**
- Modify: `src/signalx/signals/composite.py`
- Modify: `src/signalx/signals/__init__.py`
- Modify: `src/signalx/core.py`
- Modify: `tests/test_signals_composite.py`

- [ ] **Step 1: Update `composite.py`**
  - In `_calc_smc_trend_volume_confluence`: update `smc_keywords = ["fvg", "order_block", "liquidity_sweep", "market_structure_break"]` and check for `smc_` or `SMC0`.
  - In `_calc_intraday_confluence`: update lookup to `_get_series("smc_pdh_pdl_sweep_signal", "SMC011_signal")`.
  - In `tests/test_signals_composite.py`: update mock column names from `cdl_fvg_bullish_mitigation_signal` to `smc_fvg_bullish_mitigation_signal` and `cdl_pdh_pdl_sweep_signal` to `smc_pdh_pdl_sweep_signal`.

- [ ] **Step 2: Update `src/signalx/signals/__init__.py`**
  - Export `generate_smc_signals` and `SMC_SIGNAL_COLUMNS`.
  - Add `generate_smc_signals` to `run_all_signal_generators(df, show_progress=False)`.

- [ ] **Step 3: Update `src/signalx/core.py`**
  - Add `("SMC", generate_smc_signals)` in `generate_signals` category list before composite.
  - Category list now contains 8 items: Trend, Momentum, Volatility, Volume, Candlestick, SMC, Statistical, Composite.

- [ ] **Step 4: Run composite and smc tests**

Run: `uv run pytest tests/test_signals_composite.py tests/test_signals_smc.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/signalx/signals/composite.py src/signalx/signals/__init__.py src/signalx/core.py tests/test_signals_composite.py
git commit -m "develop(v0.1.0): integrate smc signals into pipeline orchestration and composite confluence"
```

---

### Task 4: Update `metadata.py`, `cli.py`, and `test_metadata.py`

**Files:**
- Modify: `src/signalx/metadata.py`
- Modify: `src/signalx/cli.py`
- Modify: `tests/test_metadata.py`
- Modify: `tests/test_full_pipeline.py`

- [ ] **Step 1: Update `metadata.py`**
  - Add `"smc"` to `SIGNAL_CATEGORIES`.
  - Register `SMC001_signal` ... `SMC011_signal` in category `"smc"`.
  - Remove retired entries from `candlestick` (`CDL021`, `CDL022`, `CDL028`-`CDL034`, `CDL038`) and `trend` (`TRD022`).
  - Total catalog entries remain 239.

- [ ] **Step 2: Update `cli.py`**
  - Add `"smc"` to category choices in `--category` argument.

- [ ] **Step 3: Update `tests/test_metadata.py` and `tests/test_full_pipeline.py`**
  - Add test for `"smc"` category lookup.
  - Verify full pipeline produces exactly 239 columns (default code and semantic).

- [ ] **Step 4: Run metadata and pipeline tests**

Run: `uv run pytest tests/test_metadata.py tests/test_full_pipeline.py tests/test_cli.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/signalx/metadata.py src/signalx/cli.py tests/test_metadata.py tests/test_full_pipeline.py
git commit -m "develop(v0.1.0): register smc category metadata and update cli filtering"
```

---

### Task 5: Update Documentation & Re-compile AI Markdown

**Files:**
- Modify: `docs/.ai/RULE.md`
- Modify: `docs/.ai/SIGNALS_CATALOG.md`
- Modify: `docs/.ai/STRUCTURE.md`
- Modify: `docs/.ai/CONFIGURATION.md`
- Modify: `docs/.ai/USAGE.md`
- Auto-generate: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` via `bash scripts/generate_agents_markdown.sh`

- [ ] **Step 1: Update `docs/.ai/RULE.md`**
  - Update Rule 7 from 7 categories to 8 categories (adding `smc`).

- [ ] **Step 2: Update `docs/.ai/SIGNALS_CATALOG.md`**
  - Add `## SMC Signals (11 Signals)` table with `SMC001_signal` ... `SMC011_signal`.
  - Update `## Candlestick Signals (28 Signals)`.
  - Update `## Trend Signals (52 Signals)`.

- [ ] **Step 3: Update `docs/.ai/STRUCTURE.md`**
  - Add `src/signalx/signals/smc.py` description.
  - Update category counts.

- [ ] **Step 4: Re-compile markdown**

Run: `bash scripts/generate_agents_markdown.sh`
Verify: `git diff AGENTS.md` shows the updated 8 categories and catalogs.

- [ ] **Step 5: Commit**

```bash
git add docs/.ai/ AGENTS.md CLAUDE.md GEMINI.md
git commit -m "develop(v0.1.0): update documentation and ai rules for 8 categories and smc suite"
```

---

### Task 6: Full Verification & Linting

**Files:** All codebase

- [ ] **Step 1: Run full test suite with coverage**

Run: `uv run pytest --cov=src/signalx -v`
Expected: 250+ tests pass with 0 failures.

- [ ] **Step 2: Run linter and formatting checks**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: All checks pass with 0 errors.

- [ ] **Step 3: Verify CLI generation on sample dataset**

Run: `uv run signalx generate datasets/sample_ohlcv.parquet -o datasets/sample_signals.parquet --no-progress`
Expected: Successfully generated signals (1000 rows, 245 columns).

Run: `uv run signalx list --category smc`
Expected: Lists all 11 SMC signals.
