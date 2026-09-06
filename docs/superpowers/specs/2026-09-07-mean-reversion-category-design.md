# SignalX Mean Reversion Category Design Specification

- **Date**: 2026-09-07
- **Topic**: Introduce Dedicated Mean Reversion Category (9th Category) with `MRxxx_signal`
- **Status**: Approved

---

## 1. Executive Summary

This specification defines the creation and integration of a brand-new, dedicated **Mean Reversion** category (`mean_reversion`) as the 9th official analytical category in `signalx`.

The new category introduces **12 classic and institutional mean reversion trading signals** prefixed with **`MR001_signal` through `MR012_signal`** (semantic prefix `mr_*_signal`). These signals are built from scratch, keeping all existing 239 signals and categories intact, expanding the total library size from **239 to 251 signals**.

---

## 2. Signal Catalog & Mathematical Specifications

| Code | Semantic Name | Strategy & Mathematical Formulation | Buy Trigger (Long Reversion) | Sell Trigger (Short Reversion) |
| :--- | :--- | :--- | :--- | :--- |
| `MR001_signal` | `mr_connors_rsi2_regime_signal` | **Larry Connors RSI(2) Trend-Filtered Reversion**: Captures extreme short-term oversold/overbought pullbacks within the dominant macro trend regime ($SMA_{200}$). | $Close > SMA_{200}$ and $RSI(2) < 10.0$ | $Close < SMA_{200}$ and $RSI(2) > 90.0$ |
| `MR002_signal` | `mr_ou_process_spread_reversion_signal` | **Ornstein-Uhlenbeck (OU) Equilibrium Spread Reversion**: Fits continuous mean-reverting process $dx_t = \theta (\mu - x_t)dt + \sigma dW_t$ over a 30-bar rolling window. Computes standardized deviation $(Close_t - \mu) / \sigma_{OU}$. | Deviation $< -2.0$ with positive reversion speed $\theta > 0$ | Deviation $> +2.0$ with positive reversion speed $\theta > 0$ |
| `MR003_signal` | `mr_vwap_distance_zscore_signal` | **Rolling VWAP Distance Z-Score Stretch**: Measures rolling 20-bar Z-score of price distance relative to 20-bar Rolling VWAP: $Z = (Distance - \mu_{dist}) / \sigma_{dist}$. | Distance Z-Score $< -2.0$ (severely stretched below VWAP) | Distance Z-Score $> +2.0$ (severely stretched above VWAP) |
| `MR004_signal` | `mr_bb_pct_b_hook_reversion_signal` | **Bollinger Bands %B Extreme Hook**: Price plunges below lower band ($\%B < 0$) or pierces above upper band ($\%B > 1$), then hooks back inside with confirming candle. | $\%B_{t-1} < 0.0$ and $\%B_t \ge 0.0$ with bullish close ($Close_t > Open_t$) | $\%B_{t-1} > 1.0$ and $\%B_t \le 1.0$ with bearish close ($Close_t < Open_t$) |
| `MR005_signal` | `mr_keltner_atr_stretch_reentry_signal` | **Keltner Channel 3-ATR Re-entry Reversal**: Extreme extension beyond $EMA_{20} \pm 3 \times ATR_{14}$ followed by re-entry into the outer channel boundary. | $Low_{t-1} < EMA_{20} - 3 \times ATR$ and $Close_t \ge EMA_{20} - 3 \times ATR$ | $High_{t-1} > EMA_{20} + 3 \times ATR$ and $Close_t \le EMA_{20} + 3 \times ATR$ |
| `MR006_signal` | `mr_kurtosis_fat_tail_exhaustion_signal` | **Rolling Kurtosis Fat-Tail Shock Reversal**: Identifies leptokurtic distribution spikes (excess kurtosis $> 3.0$ on 20-bar returns) combined with extreme return Z-scores ($|Z| > 2.5$) indicating panic/euphoria exhaustion. | Excess Kurtosis $> 3.0$ and Return Z-Score $< -2.5$ with upward hook | Excess Kurtosis $> 3.0$ and Return Z-Score $> +2.5$ with downward hook |
| `MR007_signal` | `mr_dual_ma_disparity_index_signal` | **Disparity Index Stretch**: Evaluates percentage disparity between price and $SMA_{20}$: $DI = ((Close - SMA_{20}) / SMA_{20}) \times 100$. | $DI < -3.5\%$ with bullish candle ($Close_t > Open_t$) | $DI > +3.5\%$ with bearish candle ($Close_t < Open_t$) |
| `MR008_signal` | `mr_linreg_residual_zscore_signal` | **Linear Regression Residuals Z-Score**: Computes 20-bar linear regression trendline and evaluates Z-score of regression residuals $e_t = Close_t - \hat{y}_t$. | Residual Z-Score $< -2.0$ (price stretched deeply below dynamic regression line) | Residual Z-Score $> +2.0$ (price stretched deeply above dynamic regression line) |
| `MR009_signal` | `mr_wr_cci_double_oversold_signal` | **Williams %R & CCI Confluence Mean Reversion**: Simultaneous extreme momentum exhaustion across Williams %R ($<-85$ / $>-15$) and CCI ($<-150$ / $>+150$) hooking inward. | Williams %R $< -85$ and CCI $< -150$ with both hooking upward | Williams %R $> -15$ and CCI $> +150$ with both hooking downward |
| `MR010_signal` | `mr_session_range_fade_signal` | **Intraday Session Initial Balance Fade**: Detects false extensions beyond the 30-minute Initial Balance (IB) range where price fails to sustain and reverts back into IB. | Price wicks above IB High by $>0.2\%$ but closes back inside IB ($Close < IB_{high}$) | Price wicks below IB Low by $>0.2\%$ but closes back inside IB ($Close > IB_{low}$) |
| `MR011_signal` | `mr_volume_climax_absorption_reversion_signal` | **Volume Climax Absorption at Extremes**: Giant volume spike ($Volume \ge 3 \times SMA_{20}(Volume)$) occurring at 20-bar High/Low with narrow spread / long rejection wick. | Volume spike at 20-bar Low with lower rejection wick $\ge 40\%$ and bullish close | Volume spike at 20-bar High with upper rejection wick $\ge 40\%$ and bearish close |
| `MR012_signal` | `mr_multi_period_stretch_consensus_signal` | **Multi-Period Deviation Consensus**: Simultaneous statistical stretch across 3 rolling timeframes (10, 20, 50 bars) exceeding $-1.8\sigma$ or $+1.8\sigma$. | Price simultaneously $< \mu_k - 1.8\sigma_k$ for $k \in \{10, 20, 50\}$ | Price simultaneously $> \mu_k + 1.8\sigma_k$ for $k \in \{10, 20, 50\}$ |

---

## 3. Architecture & File Layout

### 3.1 New Module: `src/signalx/signals/mean_reversion.py`
- Exports `MEAN_REVERSION_SIGNAL_COLUMNS` containing all 12 semantic column names.
- Contains the 12 calculation functions:
  - `_calc_connors_rsi2_regime`
  - `_calc_ou_process_spread_reversion`
  - `_calc_vwap_distance_zscore`
  - `_calc_bb_pct_b_hook_reversion`
  - `_calc_keltner_atr_stretch_reentry`
  - `_calc_kurtosis_fat_tail_exhaustion`
  - `_calc_dual_ma_disparity_index`
  - `_calc_linreg_residual_zscore`
  - `_calc_wr_cci_double_oversold`
  - `_calc_session_range_fade`
  - `_calc_volume_climax_absorption_reversion`
  - `_calc_multi_period_stretch_consensus`
- Exposes generator entrypoint:
  `generate_mean_reversion_signals(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame`.

### 3.2 Modifications in Core Library
1. **`src/signalx/signals/__init__.py`**:
   - Export `generate_mean_reversion_signals`, `MEAN_REVERSION_SIGNAL_COLUMNS`.
   - Add `generate_mean_reversion_signals` to `run_all_signal_generators`.
2. **`src/signalx/core.py`**:
   - Add `("Mean Reversion", generate_mean_reversion_signals)` to `generate_signals` category list.
   - Total categories: 9 (`trend`, `momentum`, `volatility`, `volume`, `candlestick`, `smc`, `mean_reversion`, `statistical`, `composite`).
3. **`src/signalx/metadata.py`**:
   - Add `"mean_reversion"` to `SIGNAL_CATEGORIES`.
   - Register `MR001_signal` ... `MR012_signal` in master catalogs.
   - Total catalog count: **251 signals**.
4. **`src/signalx/cli.py`**:
   - Add `"mean_reversion"` to category choices for `signalx list --category`.

---

## 4. Test Suite Strategy

1. **`tests/test_signals_mean_reversion.py`**:
   - Verify all 12 columns are generated and match `MEAN_REVERSION_SIGNAL_COLUMNS`.
   - Verify all signal states strictly belong to `SignalState` canonical values.
   - Verify edge cases: empty DataFrame, short DataFrame ($N < 5$), zero volume, constant prices, DatetimeIndex preservation.
   - Direct unit tests for each indicator function.
2. **`tests/test_full_pipeline.py` & `tests/test_metadata.py`**:
   - Update assertions to expect 251 signals across 9 categories.
   - Verify roundtrip renaming (`to_code_names` / `to_semantic_names`) for all 251 signals.

---

## 5. Documentation & AI Guidelines

1. Update `docs/.ai/RULE.md` (Rule 7: 9 valid categories, total 251 signals).
2. Update `docs/.ai/SIGNALS_CATALOG.md` (add `## Mean Reversion Signals (12 Signals)` table).
3. Update `docs/.ai/STRUCTURE.md`, `CONFIGURATION.md`, `USAGE.md`.
4. Re-compile AI guides via `bash scripts/generate_agents_markdown.sh`.
