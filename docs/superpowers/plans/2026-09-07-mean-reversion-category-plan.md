# Mean Reversion Category Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `mean_reversion` as the official 9th signal category in `signalx`, implementing 12 institutional mean reversion signals prefixed with `MR001_signal` through `MR012_signal`, expanding the suite from 239 to 251 total signals.

**Architecture:** 
- Implement `src/signalx/signals/mean_reversion.py` with 12 specialized mean reversion calculators.
- Define `MEAN_REVERSION_SIGNAL_COLUMNS` with semantic identifiers `mr_*_signal`.
- Wire `generate_mean_reversion_signals` into `signals/__init__.py` and `core.py` (9 categories total).
- Register `MR001_signal` ... `MR012_signal` in `metadata.py` and `--category mean_reversion` in `cli.py`.
- Add comprehensive `tests/test_signals_mean_reversion.py` and update pipeline tests for 251 signals.
- Update documentation in `docs/.ai/` and re-compile `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, pytest, ruff, uv.

## Global Constraints

- 100% of signal values must strictly adhere to `SignalState` strings: `"buy"`, `"sell"`, `"hold"`, `"none"`.
- 100% of generated signal column names must end with `_signal`.
- Zero future leakage (strictly causal, calculations at bar $t$ use data $\le t$).
- Unit tests must execute in sub-second times with synthetic fixtures (< 500 rows).
- Full compliance with `ruff check .` and `ruff format --check .`.
- Git commit message format: `<branch>(<next_tag|next_version>): <short summary in lowercase>`.

---

### Task 1: Create `src/signalx/signals/mean_reversion.py` and `tests/test_signals_mean_reversion.py`

**Files:**
- Create: `src/signalx/signals/mean_reversion.py`
- Create: `tests/test_signals_mean_reversion.py`

**Interfaces:**
- Produces: `generate_mean_reversion_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame`
- Produces: `MEAN_REVERSION_SIGNAL_COLUMNS` (list of 12 semantic signal column names)

- [ ] **Step 1: Write the failing tests in `tests/test_signals_mean_reversion.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.mean_reversion import (
    MEAN_REVERSION_SIGNAL_COLUMNS,
    generate_mean_reversion_signals,
    _calc_connors_rsi2_regime,
    _calc_ou_process_spread_reversion,
    _calc_vwap_distance_zscore,
    _calc_bb_pct_b_hook_reversion,
    _calc_keltner_atr_stretch_reentry,
    _calc_kurtosis_fat_tail_exhaustion,
    _calc_dual_ma_disparity_index,
    _calc_linreg_residual_zscore,
    _calc_wr_cci_double_oversold,
    _calc_session_range_fade,
    _calc_volume_climax_absorption_reversion,
    _calc_multi_period_stretch_consensus,
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


def test_mean_reversion_signals_all_12_columns_present():
    df = make_synthetic_ohlcv(100)
    res = generate_mean_reversion_signals(df)
    assert len(res.columns) == 12
    assert list(res.columns) == MEAN_REVERSION_SIGNAL_COLUMNS


def test_mean_reversion_signals_all_states_valid():
    df = make_synthetic_ohlcv(100)
    res = generate_mean_reversion_signals(df)
    for col in res.columns:
        assert set(res[col].unique()).issubset(ALL_SIGNAL_STATES)


def test_mean_reversion_signals_empty_and_short_dataframe():
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_mean_reversion_signals(empty_df)
    assert len(res) == 0
    assert len(res.columns) == 12

    short_df = make_synthetic_ohlcv(3)
    res_short = generate_mean_reversion_signals(short_df)
    assert len(res_short) == 3
    assert len(res_short.columns) == 12


def test_mean_reversion_signals_flat_data():
    df_flat = pd.DataFrame({
        "open": [100.0] * 50,
        "high": [100.0] * 50,
        "low": [100.0] * 50,
        "close": [100.0] * 50,
        "volume": [1000.0] * 50,
    })
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = generate_mean_reversion_signals(df_flat)
    assert len(res) == 50
    for col in res.columns:
        assert set(res[col].unique()).issubset(ALL_SIGNAL_STATES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_signals_mean_reversion.py -v`
Expected: ModuleNotFoundError for `signalx.signals.mean_reversion`.

- [ ] **Step 3: Implement `src/signalx/signals/mean_reversion.py`**

Implement the 12 calculation functions with full vectorization and safe division:
1. `_calc_connors_rsi2_regime(close)`
2. `_calc_ou_process_spread_reversion(close, window=30)`
3. `_calc_vwap_distance_zscore(high, low, close, volume, window=20)`
4. `_calc_bb_pct_b_hook_reversion(close, length=20, std=2.0)`
5. `_calc_keltner_atr_stretch_reentry(high, low, close, length=20, mult=3.0)`
6. `_calc_kurtosis_fat_tail_exhaustion(close, window=20)`
7. `_calc_dual_ma_disparity_index(close, length=20, threshold=3.5)`
8. `_calc_linreg_residual_zscore(close, length=20)`
9. `_calc_wr_cci_double_oversold(high, low, close, wr_len=14, cci_len=20)`
10. `_calc_session_range_fade(df, open_p, high, low, close)`
11. `_calc_volume_climax_absorption_reversion(open_p, high, low, close, volume, window=20)`
12. `_calc_multi_period_stretch_consensus(close)`

Implement `generate_mean_reversion_signals(df, show_progress=False) -> pd.DataFrame` with `tqdm` progress tracking.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_signals_mean_reversion.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/signalx/signals/mean_reversion.py tests/test_signals_mean_reversion.py
git commit -m "develop(v0.1.0): implement mean reversion signal generator module and test suite"
```

---

### Task 2: Integrate `mean_reversion.py` into `signals/__init__.py` and `core.py`

**Files:**
- Modify: `src/signalx/signals/__init__.py`
- Modify: `src/signalx/core.py`

- [ ] **Step 1: Update `src/signalx/signals/__init__.py`**
  - Export `generate_mean_reversion_signals`, `MEAN_REVERSION_SIGNAL_COLUMNS`.
  - Add `generate_mean_reversion_signals` to `run_all_signal_generators(df, show_progress=False)`.

- [ ] **Step 2: Update `src/signalx/core.py`**
  - Import `generate_mean_reversion_signals`.
  - Add `("Mean Reversion", generate_mean_reversion_signals)` into the pipeline category sequence:
    1. Trend
    2. Momentum
    3. Volatility
    4. Volume
    5. Candlestick
    6. SMC
    7. Mean Reversion
    8. Statistical
    9. Composite
  - Pass intermediate signals cleanly into Composite.

- [ ] **Step 3: Run pipeline tests**

Run: `uv run pytest tests/test_signals_mean_reversion.py tests/test_signals_composite.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/signalx/signals/__init__.py src/signalx/core.py
git commit -m "develop(v0.1.0): wire mean reversion generator into 9-category pipeline"
```

---

### Task 3: Register Metadata, Update CLI, and Pipeline Tests

**Files:**
- Modify: `src/signalx/metadata.py`
- Modify: `src/signalx/cli.py`
- Modify: `tests/test_metadata.py`
- Modify: `tests/test_full_pipeline.py`

- [ ] **Step 1: Update `metadata.py`**
  - Add `"mean_reversion"` to `SIGNAL_CATEGORIES` (total 9 categories).
  - Register `MR001_signal` ... `MR012_signal` with complete metadata.
  - Verify total registered signals is 251.

- [ ] **Step 2: Update `cli.py`**
  - Add `"mean_reversion"` to `--category` choices in `list` command.

- [ ] **Step 3: Update `tests/test_metadata.py` and `tests/test_full_pipeline.py`**
  - Add test for `get_signals_by_category("mean_reversion")` returning 12 signals.
  - Update `test_full_pipeline.py` assertions to verify 251 total signals.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_metadata.py tests/test_full_pipeline.py tests/test_cli.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/signalx/metadata.py src/signalx/cli.py tests/test_metadata.py tests/test_full_pipeline.py
git commit -m "develop(v0.1.0): register mean reversion metadata and update cli filtering for 251 signals"
```

---

### Task 4: Update Documentation & Re-compile AI Markdown

**Files:**
- Modify: `docs/.ai/RULE.md`
- Modify: `docs/.ai/SIGNALS_CATALOG.md`
- Modify: `docs/.ai/STRUCTURE.md`
- Modify: `docs/.ai/CONFIGURATION.md`
- Modify: `docs/.ai/USAGE.md`
- Auto-generate: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` via `bash scripts/generate_agents_markdown.sh`

- [ ] **Step 1: Update `docs/.ai/RULE.md`**
  - Update Rule 7 to 9 valid categories and 251 signals.

- [ ] **Step 2: Update `docs/.ai/SIGNALS_CATALOG.md`**
  - Add `## Mean Reversion Signals (12 Signals)` table with `MR001_signal` ... `MR012_signal`.
  - Update total signals count to 251.

- [ ] **Step 3: Update `docs/.ai/STRUCTURE.md`**
  - Add `src/signalx/signals/mean_reversion.py` and `tests/test_signals_mean_reversion.py`.

- [ ] **Step 4: Re-compile markdown**

Run: `bash scripts/generate_agents_markdown.sh`
Verify clean sync with zero markdown syntax issues.

- [ ] **Step 5: Commit**

```bash
git add docs/.ai/ AGENTS.md CLAUDE.md GEMINI.md README.md
git commit -m "develop(v0.1.0): update documentation for 9 categories and 251 signals suite"
```

---

### Task 5: Full Verification & Linting

**Files:** All codebase

- [ ] **Step 1: Run full test suite with coverage**

Run: `uv run pytest --cov=src/signalx -v`
Expected: 260+ tests pass with 0 failures, >90% coverage.

- [ ] **Step 2: Run linter and formatting checks**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: All checks pass with 0 errors.

- [ ] **Step 3: Verify CLI generation on sample dataset**

Run: `uv run signalx generate datasets/sample_ohlcv.parquet -o datasets/sample_signals.parquet --no-progress`
Expected: Successfully generated signals (1000 rows, 257 columns: 6 ohlcv + 251 signals).

Run: `uv run signalx list --category mean_reversion`
Expected: Lists all 12 Mean Reversion signals.
