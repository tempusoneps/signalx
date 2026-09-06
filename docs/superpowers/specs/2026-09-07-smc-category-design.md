# SignalX SMC (Smart Money Concepts) Category Design Specification

- **Date**: 2026-09-07
- **Topic**: Introduce Dedicated SMC Category (8th Category) & Module Migration
- **Status**: Approved

---

## 1. Executive Summary

This specification defines the architectural migration of Smart Money Concepts (SMC) signals from `candlestick` and `trend` categories into a dedicated first-class category `smc` (8th category in `signalx`).

Currently, 11 SMC signals (such as FVG, Order Blocks, Market Structure Breaks, Liquidity Sweeps, and Judas Swings) are scattered inside `candlestick.py` (10 signals) and `trend.py` (1 signal). Moving them into `src/signalx/signals/smc.py` provides clean domain separation, dedicated CLI inspection (`signalx list --category smc`), and an independent test suite.

Per user agreement:
- Migration is a **clean cut**: old codes (`CDL028`... `TRD022`) are replaced by `SMC001_signal` ... `SMC011_signal`.
- The remaining signals in `candlestick` (28 signals) and `trend` (52 signals) maintain their existing numeric indices without renumbering.
- Total signals across `signalx` remain **239 signals** (52 trend + 39 momentum + 44 volatility + 32 volume + 28 candlestick + 20 statistical + 13 composite + 11 smc).

---

## 2. Signal Catalog Mapping

| New Code | Old Code | New Semantic Name | Old Semantic Name | Trigger Summary |
| :--- | :--- | :--- | :--- | :--- |
| `SMC001_signal` | `CDL028_signal` | `smc_fvg_bullish_mitigation_signal` | `cdl_fvg_bullish_mitigation_signal` | Bullish FVG creation ($Low_t > High_{t-2}$) and mitigation retest |
| `SMC002_signal` | `CDL029_signal` | `smc_fvg_bearish_mitigation_signal` | `cdl_fvg_bearish_mitigation_signal` | Bearish FVG creation ($High_t < Low_{t-2}$) and mitigation retest |
| `SMC003_signal` | `CDL030_signal` | `smc_order_block_retest_signal` | `cdl_order_block_retest_signal` | Bullish/Bearish Order Block formation and retest |
| `SMC004_signal` | `CDL031_signal` | `smc_break_of_structure_signal` | `cdl_break_of_structure_signal` | Break of Structure (BOS) continuation with trend filter |
| `SMC005_signal` | `CDL032_signal` | `smc_change_of_character_signal` | `cdl_change_of_character_signal` | Change of Character (CHoCH) structural reversal |
| `SMC006_signal` | `TRD022_signal` | `smc_market_structure_break_signal` | `trend_market_structure_break_signal` | Market Structure Break (MSB) HH/LL reversal |
| `SMC007_signal` | `CDL021_signal` | `smc_liquidity_sweep_signal` | `cdl_liquidity_sweep_signal` | 5-bar high/low liquidity sweep with close back inside |
| `SMC008_signal` | `CDL022_signal` | `smc_equal_high_low_sweep_signal` | `cdl_equal_high_low_sweep_signal` | Equal Highs / Equal Lows (EQH/EQL) liquidity sweep |
| `SMC009_signal` | `CDL033_signal` | `smc_judas_swing_signal` | `cdl_judas_swing_signal` | ICT Judas Swing false opening breakout & reversal |
| `SMC010_signal` | `CDL034_signal` | `smc_inducement_sweep_signal` | `cdl_inducement_sweep_signal` | Inducement (IDM) minor liquidity sweep & wick rejection |
| `SMC011_signal` | `CDL038_signal` | `smc_pdh_pdl_sweep_signal` | `cdl_pdh_pdl_sweep_signal` | Prior Day High/Low (PDH/PDL) sweep & mean-reversion |

---

## 3. Architecture & File Layout

### 3.1 New Module: `src/signalx/signals/smc.py`
- Exports `SMC_SIGNAL_COLUMNS` containing the 11 semantic column names.
- Contains the 11 calculation functions:
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
- Exposes entrypoint:
  `generate_smc_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame`.

### 3.2 Modifications in Existing Modules
1. **`src/signalx/signals/candlestick.py`**:
   - Remove the 10 migrated calculation functions and column names.
   - `CANDLESTICK_SIGNAL_COLUMNS` updated to 28 entries.
2. **`src/signalx/signals/trend.py`**:
   - Remove `_calc_market_structure_break` and `trend_market_structure_break_signal`.
   - `TREND_SIGNAL_COLUMNS` updated to 52 entries.
3. **`src/signalx/signals/composite.py`**:
   - Update `_calc_smc_trend_volume_confluence` to look for `smc_` keywords.
   - Update `_calc_intraday_confluence` to read `smc_pdh_pdl_sweep_signal` / `SMC011_signal`.
4. **`src/signalx/signals/__init__.py`**:
   - Import and export `generate_smc_signals`, `SMC_SIGNAL_COLUMNS`.
   - Include `generate_smc_signals` in `run_all_signal_generators`.
5. **`src/signalx/core.py`**:
   - Add `"smc"` category step in `generate_signals` orchestration.
   - Total categories: 8 (`trend`, `momentum`, `volatility`, `volume`, `candlestick`, `smc`, `statistical`, `composite`).
6. **`src/signalx/metadata.py`**:
   - Add `"smc"` to `SIGNAL_CATEGORIES`.
   - Register `SMC001_signal` ... `SMC011_signal` with full trigger metadata.
   - Remove retired `CDL` / `TRD` entries.
7. **`src/signalx/cli.py`**:
   - Add `"smc"` to category choices for `signalx list --category`.

---

## 4. Test Suite Strategy

1. **`tests/test_signals_smc.py`**:
   - `test_smc_signals_all_11_columns_present`
   - `test_smc_signals_all_states_valid`
   - `test_smc_signals_empty_dataframe`
   - `test_smc_signals_short_dataframe`
   - `test_smc_signals_datetime_index_preserved`
   - Direct calculation unit tests for FVG, OB, BOS, CHoCH, MSB, Sweeps, Judas swing, PDH/PDL.
2. **Update Existing Tests**:
   - `test_signals_candlestick.py`: update expected columns from 38 to 28.
   - `test_signals_trend.py`: update expected columns from 53 to 52.
   - `test_signals_composite.py`: update SMC column fixtures.
   - `test_full_pipeline.py`: verify 8 categories and 239 signals total.
   - `test_metadata.py`: verify `"smc"` category lookup.

---

## 5. Documentation & AI Rule Synchronization

1. Update `docs/.ai/RULE.md` (Rule 7: 8 valid categories).
2. Update `docs/.ai/SIGNALS_CATALOG.md` (new SMC table, updated Candlestick and Trend tables).
3. Update `docs/.ai/STRUCTURE.md`, `CONFIGURATION.md`, `USAGE.md`.
4. Run `bash scripts/generate_agents_markdown.sh` to compile `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md`.
