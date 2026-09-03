# SignalX 230-Signal Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the `signalx` quantitative finance library from 172 to 230 standardized trading signals (+58 new signals) across all 7 categories (Candlestick, Trend, Volume, Volatility, Momentum, Statistical, Composite).

**Architecture:** Implement vectorized indicator calculation functions in each category module under `src/signalx/signals/`, register metadata in `src/signalx/metadata.py`, write unit test fixtures in `tests/test_signals_*.py`, verify zero lookahead bias and strict 4-state contract (`"buy"`, `"sell"`, `"hold"`, `"none"`), and update auto-generated AI agent docs.

**Tech Stack:** Python 3.12+, pandas, numpy, scipy, ta, pandas-ta, pytest, ruff.

## Global Constraints

- **Python Version**: Python >= 3.12 (`from __future__ import annotations`).
- **4-State State Contract**: Output values strictly from `{"buy", "sell", "hold", "none"}` (no `NaN`, numbers, booleans).
- **Column Postfix**: 100% of generated signal columns must end with `_signal`.
- **Zero Future Leakage**: At index $t$, calculations strictly depend on data $\le t$.
- **Test Performance**: All unit tests must run in $< 1.0$s total using synthetic data (< 500 rows).
- **Linting**: 100% clean with `uv run ruff check .` and `uv run ruff format --check .`.

---

### Task 1: Candlestick Signals Expansion (`CDL028` - `CDL037`)

**Files:**
- Modify: `src/signalx/signals/candlestick.py`
- Modify: `src/signalx/metadata.py`
- Modify: `tests/test_signals_candlestick.py`

**Interfaces:**
- Consumes: `open_p`, `high`, `low`, `close`, `volume` normalized Series.
- Produces: 10 new signal columns (`cdl_fvg_bullish_mitigation_signal`, `cdl_fvg_bearish_mitigation_signal`, `cdl_order_block_retest_signal`, `cdl_break_of_structure_signal`, `cdl_change_of_character_signal`, `cdl_judas_swing_signal`, `cdl_inducement_sweep_signal`, `cdl_thrust_bar_signal`, `cdl_narrow_range_7_breakout_signal`, `cdl_wide_range_reversal_signal`).

- [ ] **Step 1: Write unit tests for new Candlestick signals**

Add test cases in `tests/test_signals_candlestick.py` verifying each of the 10 new signals produces valid 4-state strings and expected output shapes.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals_candlestick.py -k "fvg or order_block or structure or judas" -v`
Expected: FAIL (signals not yet in DataFrame).

- [ ] **Step 3: Implement calculation functions and update CANDLESTICK_SIGNAL_COLUMNS**

In `src/signalx/signals/candlestick.py`:
- Add calculation helpers for `_calc_fvg_bullish_mitigation`, `_calc_fvg_bearish_mitigation`, `_calc_order_block_retest`, `_calc_break_of_structure`, `_calc_change_of_character`, `_calc_judas_swing`, `_calc_inducement_sweep`, `_calc_thrust_bar`, `_calc_narrow_range_7_breakout`, `_calc_wide_range_reversal`.
- Append the 10 column names to `CANDLESTICK_SIGNAL_COLUMNS` (increasing count from 27 to 37).
- Update `generate_candlestick_signals` with progress updates.

In `src/signalx/metadata.py`:
- Register `CDL028_signal` through `CDL037_signal`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals_candlestick.py -v`
Expected: PASS (all 37 candlestick signals pass).

- [ ] **Step 5: Commit changes**

```bash
git add src/signalx/signals/candlestick.py src/signalx/metadata.py tests/test_signals_candlestick.py
git commit -m "feat(signals): add 10 SMC and candlestick signals (CDL028-CDL037)"
```

---

### Task 2: Trend Signals Expansion (`TRD042` - `TRD053`)

**Files:**
- Modify: `src/signalx/signals/trend.py`
- Modify: `src/signalx/metadata.py`
- Modify: `tests/test_signals_trend.py`

**Interfaces:**
- Consumes: `open_p`, `high`, `low`, `close`, `volume` normalized Series.
- Produces: 12 new trend signals (`trend_ehlers_super_smoother_cross_signal`, `trend_mcginley_dynamic_cross_signal`, `trend_gmma_ribbon_expansion_signal`, `trend_gmma_compression_breakout_signal`, `trend_rainbow_ema_alignment_signal`, `trend_ehlers_instantaneous_trend_signal`, `trend_coral_trend_filter_signal`, `trend_supertrend_atr_20_5_signal`, `trend_donchian_middle_cross_20_signal`, `trend_alligator_lips_jaw_cross_signal`, `trend_alma_cross_9_signal`, `trend_zero_lag_ema_cross_21_signal`).

- [ ] **Step 1: Write unit tests for new Trend signals**

Add test assertions in `tests/test_signals_trend.py` checking all 12 new signals for validity, non-empty states, and correct index alignment.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals_trend.py -k "ehlers or mcginley or gmma or rainbow" -v`
Expected: FAIL.

- [ ] **Step 3: Implement calculation functions and update TREND_SIGNAL_COLUMNS**

In `src/signalx/signals/trend.py`:
- Implement calculation helpers for Super Smoother, McGinley Dynamic, GMMA, Rainbow EMA, Ehlers Instantaneous Trend, Coral Filter, SuperTrend (20, 5), Donchian Median Cross, Alligator, ALMA, and ZLEMA.
- Append 12 column names to `TREND_SIGNAL_COLUMNS` (increasing count from 41 to 53).
- Update `generate_trend_signals`.

In `src/signalx/metadata.py`:
- Register `TRD042_signal` through `TRD053_signal`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals_trend.py -v`
Expected: PASS (all 53 trend signals pass).

- [ ] **Step 5: Commit changes**

```bash
git add src/signalx/signals/trend.py src/signalx/metadata.py tests/test_signals_trend.py
git commit -m "feat(signals): add 12 DSP and trend signals (TRD042-TRD053)"
```

---

### Task 3: Volume Signals Expansion (`VLM019` - `VLM028`)

**Files:**
- Modify: `src/signalx/signals/volume.py`
- Modify: `src/signalx/metadata.py`
- Modify: `tests/test_signals_volume.py`

**Interfaces:**
- Consumes: `open_p`, `high`, `low`, `close`, `volume` normalized Series.
- Produces: 10 new volume signals (`volume_klinger_osc_cross_signal`, `volume_elder_ray_bull_bear_signal`, `volume_climax_absorption_signal`, `volume_twiggs_money_flow_cross_signal`, `volume_nvi_pvi_cross_signal`, `volume_vwap_anchored_dev1_signal`, `volume_vwap_anchored_dev3_signal`, `volume_delta_proxy_surge_signal`, `volume_vwma_sma_divergence_signal`, `volume_volume_weighted_rsi_14_signal`).

- [ ] **Step 1: Write unit tests for new Volume signals**

Add test assertions in `tests/test_signals_volume.py` checking the 10 new volume signals.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals_volume.py -k "klinger or elder_ray or climax or twiggs" -v`
Expected: FAIL.

- [ ] **Step 3: Implement calculation functions and update VOLUME_SIGNAL_COLUMNS**

In `src/signalx/signals/volume.py`:
- Implement Klinger Oscillator, Elder Ray Index, Volume Climax Absorption, Twiggs Money Flow, NVI/PVI, VWAP 1-std & 3-std bands, Intrabar Delta Proxy, VWMA-SMA divergence, and Volume-Weighted RSI.
- Append 10 column names to `VOLUME_SIGNAL_COLUMNS` (increasing count from 18 to 28).
- Update `generate_volume_signals`.

In `src/signalx/metadata.py`:
- Register `VLM019_signal` through `VLM028_signal`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals_volume.py -v`
Expected: PASS (all 28 volume signals pass).

- [ ] **Step 5: Commit changes**

```bash
git add src/signalx/signals/volume.py src/signalx/metadata.py tests/test_signals_volume.py
git commit -m "feat(signals): add 10 volume dynamics and order flow signals (VLM019-VLM028)"
```

---

### Task 4: Volatility Signals Expansion (`VOL033` - `VOL042`)

**Files:**
- Modify: `src/signalx/signals/volatility.py`
- Modify: `src/signalx/metadata.py`
- Modify: `tests/test_signals_volatility.py`

**Interfaces:**
- Consumes: `open_p`, `high`, `low`, `close`, `volume` normalized Series.
- Produces: 10 new volatility signals (`vol_rvi_ob_os_14_signal`, `vol_garman_klass_expansion_signal`, `vol_parkinson_volatility_surge_signal`, `vol_squeeze_momentum_pro_signal`, `vol_keltner_width_squeeze_signal`, `vol_atr_ratio_fast_slow_signal`, `vol_chandelier_exit_reversal_signal`, `vol_mass_index_reversal_bulge_signal`, `vol_normalized_atr_stretch_signal`, `vol_dual_thrust_range_breakout_signal`).

- [ ] **Step 1: Write unit tests for new Volatility signals**

Add test assertions in `tests/test_signals_volatility.py` checking the 10 new volatility signals.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals_volatility.py -k "rvi or garman_klass or squeeze_momentum_pro or dual_thrust" -v`
Expected: FAIL.

- [ ] **Step 3: Implement calculation functions and update VOLATILITY_SIGNAL_COLUMNS**

In `src/signalx/signals/volatility.py`:
- Implement RVI, Garman-Klass Volatility, Parkinson Volatility, Squeeze Momentum Pro, Keltner Width Squeeze, ATR Fast/Slow Ratio, Chandelier Exit, Mass Index Bulge, NATR Stretch, and Dual Thrust Breakout.
- Append 10 column names to `VOLATILITY_SIGNAL_COLUMNS` (increasing count from 32 to 42).
- Update `generate_volatility_signals`.

In `src/signalx/metadata.py`:
- Register `VOL033_signal` through `VOL042_signal`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals_volatility.py -v`
Expected: PASS (all 42 volatility signals pass).

- [ ] **Step 5: Commit changes**

```bash
git add src/signalx/signals/volatility.py src/signalx/metadata.py tests/test_signals_volatility.py
git commit -m "feat(signals): add 10 volatility and breakout signals (VOL033-VOL042)"
```

---

### Task 5: Momentum Signals Expansion (`MOM031` - `MOM038`)

**Files:**
- Modify: `src/signalx/signals/momentum.py`
- Modify: `src/signalx/metadata.py`
- Modify: `tests/test_signals_momentum.py`

**Interfaces:**
- Consumes: `open_p`, `high`, `low`, `close`, `volume` normalized Series.
- Produces: 8 new momentum signals (`mom_rmi_ob_os_14_signal`, `mom_dmi_variable_lookback_signal`, `mom_coppock_curve_zero_cross_signal`, `mom_stoch_momentum_index_cross_signal`, `mom_schaff_trend_cycle_cross_signal`, `mom_cmo_divergence_signal`, `mom_kst_oscillator_cross_signal`, `mom_demarker_indicator_cross_signal`).

- [ ] **Step 1: Write unit tests for new Momentum signals**

Add test assertions in `tests/test_signals_momentum.py` checking the 8 new momentum signals.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals_momentum.py -k "rmi or coppock or schaff or kst or demarker" -v`
Expected: FAIL.

- [ ] **Step 3: Implement calculation functions and update MOMENTUM_SIGNAL_COLUMNS**

In `src/signalx/signals/momentum.py`:
- Implement RMI, Dynamic Momentum (DMI), Coppock Curve, SMI, Schaff Trend Cycle (STC), CMO divergence, KST Oscillator, and DeMarker Indicator.
- Append 8 column names to `MOMENTUM_SIGNAL_COLUMNS` (increasing count from 30 to 38).
- Update `generate_momentum_signals`.

In `src/signalx/metadata.py`:
- Register `MOM031_signal` through `MOM038_signal`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals_momentum.py -v`
Expected: PASS (all 38 momentum signals pass).

- [ ] **Step 5: Commit changes**

```bash
git add src/signalx/signals/momentum.py src/signalx/metadata.py tests/test_signals_momentum.py
git commit -m "feat(signals): add 8 advanced momentum signals (MOM031-MOM038)"
```

---

### Task 6: Statistical Signals Expansion (`STA017` - `STA020`)

**Files:**
- Modify: `src/signalx/signals/statistical.py`
- Modify: `src/signalx/metadata.py`
- Modify: `tests/test_signals_statistical.py`

**Interfaces:**
- Consumes: `open_p`, `high`, `low`, `close`, `volume` normalized Series.
- Produces: 4 new statistical signals (`stat_fractal_dimension_index_signal`, `stat_rolling_half_life_reversion_signal`, `stat_variance_ratio_test_signal`, `stat_rolling_skewness_reversal_signal`).

- [ ] **Step 1: Write unit tests for new Statistical signals**

Add test assertions in `tests/test_signals_statistical.py` checking the 4 new statistical signals.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals_statistical.py -k "fractal or half_life or variance_ratio or skewness" -v`
Expected: FAIL.

- [ ] **Step 3: Implement calculation functions and update STATISTICAL_SIGNAL_COLUMNS**

In `src/signalx/signals/statistical.py`:
- Implement Fractal Dimension Index (FDI), Ornstein-Uhlenbeck Half-Life, Lo-MacKinlay Variance Ratio, and Rolling Return Skewness.
- Append 4 column names to `STATISTICAL_SIGNAL_COLUMNS` (increasing count from 16 to 20).
- Update `generate_statistical_signals`.

In `src/signalx/metadata.py`:
- Register `STA017_signal` through `STA020_signal`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals_statistical.py -v`
Expected: PASS (all 20 statistical signals pass).

- [ ] **Step 5: Commit changes**

```bash
git add src/signalx/signals/statistical.py src/signalx/metadata.py tests/test_signals_statistical.py
git commit -m "feat(signals): add 4 statistical regime signals (STA017-STA020)"
```

---

### Task 7: Composite Signals Expansion (`CMP009` - `CMP012`)

**Files:**
- Modify: `src/signalx/signals/composite.py`
- Modify: `src/signalx/metadata.py`
- Modify: `tests/test_signals_composite.py`

**Interfaces:**
- Consumes: `df` normalized DataFrame with underlying signals.
- Produces: 4 new composite signals (`comp_smc_trend_volume_confluence_signal`, `comp_triple_screen_trading_system_signal`, `comp_squeeze_momentum_volume_surge_signal`, `comp_master_ensemble_v2_signal`).

- [ ] **Step 1: Write unit tests for new Composite signals**

Add test assertions in `tests/test_signals_composite.py` checking the 4 new composite signals.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals_composite.py -k "smc_trend or triple_screen or squeeze_momentum_volume or master_ensemble_v2" -v`
Expected: FAIL.

- [ ] **Step 3: Implement calculation functions and update COMPOSITE_SIGNAL_COLUMNS**

In `src/signalx/signals/composite.py`:
- Implement SMC Confluence, Triple Screen Trading System, Squeeze Momentum + Volume Surge, and Master Ensemble v2.
- Append 4 column names to `COMPOSITE_SIGNAL_COLUMNS` (increasing count from 8 to 12).
- Update `generate_composite_signals`.

In `src/signalx/metadata.py`:
- Register `CMP009_signal` through `CMP012_signal`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals_composite.py -v`
Expected: PASS (all 12 composite signals pass).

- [ ] **Step 5: Commit changes**

```bash
git add src/signalx/signals/composite.py src/signalx/metadata.py tests/test_signals_composite.py
git commit -m "feat(signals): add 4 multi-factor composite signals (CMP009-CMP012)"
```

---

### Task 8: End-to-End Integration, Pipeline Verification & Documentation Sync

**Files:**
- Modify: `tests/test_metadata.py`
- Modify: `tests/test_full_pipeline.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/.ai/SIGNALS_CATALOG.md`
- Modify: `docs/.ai/CONFIGURATION.md`
- Modify: `docs/.ai/STRUCTURE.md`
- Modify: `docs/.ai/USAGE.md`
- Modify: `docs/.ai/AI_AGENT_GUIDELINE.md`
- Run: `bash scripts/generate_agents_markdown.sh`

- [ ] **Step 1: Update metadata and integration tests for 230 signals**

In `tests/test_metadata.py`, `tests/test_full_pipeline.py`, `tests/test_cli.py`:
- Update expected count assertions from 172 to 230.
- Update category counts (Candlestick: 37, Trend: 53, Volume: 28, Volatility: 42, Momentum: 38, Statistical: 20, Composite: 12).

- [ ] **Step 2: Run full test suite and verify sub-second execution**

Run: `uv run pytest -v`
Expected: PASS with 100% tests passing in $< 1.0$s.

- [ ] **Step 3: Update documentation in docs/.ai/ and recompile agent guidelines**

Update `docs/.ai/SIGNALS_CATALOG.md` with all 58 new signals.
Run: `bash scripts/generate_agents_markdown.sh`
Expected: `AGENTS.md`, `GEMINI.md`, `CLAUDE.md` re-compiled cleanly.

- [ ] **Step 4: Run code linter and formatting checks**

Run: `uv run ruff check .` and `uv run ruff format --check .`
Expected: 0 errors.

- [ ] **Step 5: Commit final integration and documentation updates**

```bash
git add tests/ docs/ AGENTS.md GEMINI.md CLAUDE.md
git commit -m "docs(catalog): update signal catalog and agent instructions to 230 signals"
```
