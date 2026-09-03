# SignalX 230-Signal Expansion Design Specification

- **Date**: 2026-09-04
- **Author**: Antigravity Quantitative AI Team
- **Scope**: Expansion of SignalX from 172 to 230 standardized trading signals (+58 new signals)
- **Status**: Approved for Implementation Planning

---

## 1. Executive Summary & Goals

`signalx` currently provides 172 standardized, leak-free trading signals across 7 categories. This specification defines the architecture, calculation logic, metadata catalogs, and verification procedures for adding **58 new high-conviction quantitative signals**, bringing the total library count to **230 signals**.

### Core Invariants Maintained:
1. **4-State Strict Contract**: Every signal outputs strictly `"buy"`, `"sell"`, `"hold"`, `"none"`.
2. **Naming Convention**: 
   - Coded format: `<CAT><INDEX>_signal` (e.g. `CDL028_signal`, `TRD042_signal`, `CMP009_signal`).
   - Semantic format: `<category>_<name>_<params>_signal` (e.g. `cdl_fvg_bullish_mitigation_signal`).
   - All columns end in `_signal`.
3. **Zero Future Leakage**: All calculations at index $t$ use strictly data $\le t$.
4. **Sub-second Test Suite**: All unit and integration tests execute in $<1$ second total using lightweight synthetic datasets.

---

## 2. Category Distribution Overview

| Category | Existing Count | Added Count | New Total | Added Code Range | Key Themes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Candlestick** | 27 | **+10** | **37** | `CDL028_signal` - `CDL037_signal` | SMC (FVG, Order Block, BOS, CHoCH, Inducement, Judas Swing, Range extremes) |
| **Trend** | 41 | **+12** | **53** | `TRD042_signal` - `TRD053_signal` | DSP (Ehlers Super Smoother, Instantaneous Trend), McGinley, GMMA, Ribbons, ALMA, ZLEMA |
| **Volume** | 18 | **+10** | **28** | `VLM019_signal` - `VLM028_signal` | Klinger Osc, Elder Ray, Twiggs Money Flow, Volume Climax, Delta Proxy, VWAP $\pm 1\sigma, \pm 3\sigma$ |
| **Volatility** | 32 | **+10** | **42** | `VOL033_signal` - `VOL042_signal` | RVI, Squeeze Pro, Garman-Klass, Parkinson, Chandelier Exit, Mass Index, Dual Thrust |
| **Momentum** | 30 | **+8** | **38** | `MOM031_signal` - `MOM038_signal` | RMI, Dynamic Momentum (DMI), Coppock Curve, SMI, Schaff Trend Cycle, KST, DeMarker |
| **Statistical** | 16 | **+4** | **20** | `STA017_signal` - `STA020_signal` | Fractal Dimension Index (FDI), Half-Life Mean Reversion, Variance Ratio Test, Skewness |
| **Composite** | 8 | **+4** | **12** | `CMP009_signal` - `CMP012_signal` | SMC Confluence, Triple Screen System, Squeeze Pro + Volume Surge, Master Ensemble v2 |
| **Total** | **172** | **+58** | **230** | | Full Quantitative Feature Suite |

---

## 3. Detailed Signal Specifications (58 New Signals)

### 3.1 Candlestick Signals (+10 Signals, `CDL028` - `CDL037`)

| Code | Semantic Name | Description & Formula | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `CDL028_signal` | `cdl_fvg_bullish_mitigation_signal` | Bullish Fair Value Gap mitigation (Gap: $Low_{t} > High_{t-2}$). Signal triggers when price later retraces into $[High_{t-2}, Low_{t}]$ and bounces. | Price pulls back into active Bullish FVG zone and closes bullish ($Close > Open$). | Active Bullish FVG zone violated and closed below ($Close < High_{FVG\_origin}$). |
| `CDL029_signal` | `cdl_fvg_bearish_mitigation_signal` | Bearish Fair Value Gap mitigation (Gap: $High_{t} < Low_{t-2}$). Signal triggers when price retraces up into $[Low_{t-2}, High_{t}]$ and rejects. | Active Bearish FVG zone invalidated upward ($Close > Low_{FVG\_origin}$). | Price rallies into active Bearish FVG zone and closes bearish ($Close < Open$). |
| `CDL030_signal` | `cdl_order_block_retest_signal` | Order Block retest: last counter-candle before a 3-bar strong displacement breakout. | Price retraces into Bullish OB range and closes higher. | Price rallies into Bearish OB range and closes lower. |
| `CDL031_signal` | `cdl_break_of_structure_signal` | Break of Structure (BOS): Bar closes beyond the rolling 10-period swing high/low in the direction of the medium-term trend ($SMA20 > SMA50$). | $Close > \max(High_{10})$ and $SMA20 > SMA50$. | $Close < \min(Low_{10})$ and $SMA20 < SMA50$. |
| `CDL032_signal` | `cdl_change_of_character_signal` | Change of Character (CHoCH): First break of previous lower high during a downtrend or higher low during an uptrend. | $Close > \max(High_5)$ while previous 10-bar regime was bearish. | $Close < \min(Low_5)$ while previous 10-bar regime was bullish. |
| `CDL033_signal` | `cdl_judas_swing_signal` | Judas Swing: False breakout of prior 5-bar high/low followed immediately by a sharp reversal bar. | $Low < \min(Low_5)_{t-1}$ but bar closes strong green ($Close > Open$ and upper half). | $High > \max(High_5)_{t-1}$ but bar closes strong red ($Close < Open$ and lower half). |
| `CDL034_signal` | `cdl_inducement_sweep_signal` | Inducement Sweep: Minor internal high/low (3-bar) swept before resuming major move. | Minor low swept ($Low < Low_{t-1}$) with long lower rejection shadow $> 50\%$ bar range. | Minor high swept ($High > High_{t-1}$) with long upper rejection shadow $> 50\%$ bar range. |
| `CDL035_signal` | `cdl_thrust_bar_signal` | Directional Thrust Bar: Candle body $\ge 75\%$ total range and Body $\ge 1.8 \times SMA20(Body)$. | Bullish Thrust Bar ($Close > Open$). | Bearish Thrust Bar ($Close < Open$). |
| `CDL036_signal` | `cdl_narrow_range_7_breakout_signal` | Narrow Range 7 (NR7): Smallest candle range of past 7 bars, followed by expansion breakout. | Current bar breaks out above NR7 high. | Current bar breaks down below NR7 low. |
| `CDL037_signal` | `cdl_wide_range_reversal_signal` | Wide Range Bar (WRB) Reversal: Candle range $\ge 2.5 \times SMA20(Range)$ with counter-directional rejection. | Wide range bar dipping into lows but closing in upper $30\%$ of bar range. | Wide range bar spiking into highs but closing in lower $30\%$ of bar range. |

---

### 3.2 Trend Signals (+12 Signals, `TRD042` - `TRD053`)

| Code | Semantic Name | Description & Formula | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `TRD042_signal` | `trend_ehlers_super_smoother_cross_signal` | John Ehlers 2-Pole Super Smoother Filter price crossover. | Price crosses above Super Smoother line. | Price crosses below Super Smoother line. |
| `TRD043_signal` | `trend_mcginley_dynamic_cross_signal` | McGinley Dynamic ($MD_t = MD_{t-1} + \frac{Price - MD_{t-1}}{N \cdot (Price/MD_{t-1})^4}$) price cross. | Price crosses above McGinley Dynamic ($N=14$). | Price crosses below McGinley Dynamic ($N=14$). |
| `TRD044_signal` | `trend_gmma_ribbon_expansion_signal` | Guppy Multiple Moving Average (GMMA): Fast group (3, 5, 8, 10, 12, 15) vs Slow group (30, 35, 40, 45, 50, 60) expansion. | $\min(Fast) > \max(Slow)$ and Fast spread expanding. | $\max(Fast) < \min(Slow)$ and Fast spread expanding. |
| `TRD045_signal` | `trend_gmma_compression_breakout_signal` | GMMA Compression Breakout: All 12 EMAs compress within a tight band ($\le 1.5\%$) and price breaks out. | Breakout above compressed GMMA ribbon. | Breakdown below compressed GMMA ribbon. |
| `TRD046_signal` | `trend_rainbow_ema_alignment_signal` | Rainbow EMA Alignment: 5 EMAs ($8, 13, 21, 34, 55$) perfectly ordered. | $EMA8 > EMA13 > EMA21 > EMA34 > EMA55$. | $EMA8 < EMA13 < EMA21 < EMA34 < EMA55$. |
| `TRD047_signal` | `trend_ehlers_instantaneous_trend_signal` | Ehlers Instantaneous Trendline price crossover with zero-phase response. | Price crosses above Instantaneous Trendline. | Price crosses below Instantaneous Trendline. |
| `TRD048_signal` | `trend_coral_trend_filter_signal` | Coral Filter (Braid smoothed regime filter) slope direction. | Coral Filter slope flips positive ($Coral_t > Coral_{t-1}$). | Coral Filter slope flips negative ($Coral_t < Coral_{t-1}$). |
| `TRD049_signal` | `trend_supertrend_atr_20_5_signal` | Conservative Slow SuperTrend ($Length=20, Multiplier=5.0$). | SuperTrend direction flips bullish. | SuperTrend direction flips bearish. |
| `TRD050_signal` | `trend_donchian_middle_cross_20_signal` | 20-period Donchian Channel median line ($(High_{20} + Low_{20})/2$) price crossover. | Price crosses above Donchian Median. | Price crosses below Donchian Median. |
| `TRD051_signal` | `trend_alligator_lips_jaw_cross_signal` | Bill Williams Alligator Lips (EMA 5 shifted) vs Jaw (EMA 13 shifted) crossover. | Lips cross above Jaw and Teeth in expansion. | Lips cross below Jaw and Teeth in expansion. |
| `TRD052_signal` | `trend_alma_cross_9_signal` | Arnaud Legoux Moving Average (ALMA 9, $\sigma=6, offset=0.85$) price cross. | Price crosses above ALMA 9. | Price crosses below ALMA 9. |
| `TRD053_signal` | `trend_zero_lag_ema_cross_21_signal` | Zero-Lag EMA ($ZLEMA = 2 \cdot EMA - EMA(EMA)$) 21-period price crossover. | Price crosses above ZLEMA 21. | Price crosses below ZLEMA 21. |

---

### 3.3 Volume Signals (+10 Signals, `VLM019` - `VLM028`)

| Code | Semantic Name | Description & Formula | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `VLM019_signal` | `volume_klinger_osc_cross_signal` | Klinger Volume Oscillator (KVO 34/55/13) signal line crossover. | KVO line crosses above KVO signal line. | KVO line crosses below KVO signal line. |
| `VLM020_signal` | `volume_elder_ray_bull_bear_signal` | Elder Ray Index: Bull Power ($High - EMA13$) and Bear Power ($Low - EMA13$). | Bear Power $< 0$ but rising and Bull Power $> 0$. | Bull Power $> 0$ but falling and Bear Power $< 0$. |
| `VLM021_signal` | `volume_climax_absorption_signal` | Volume Climax Absorption: Volume $\ge 3.0 \times SMA20(Volume)$ with small body / rejection wick. | Ultra-high volume on lower wick absorption candle ($Close > Open$). | Ultra-high volume on upper wick exhaustion candle ($Close < Open$). |
| `VLM022_signal` | `volume_twiggs_money_flow_cross_signal` | Twiggs Money Flow (21 period) zero centerline crossover. | TMF crosses above 0. | TMF crosses below 0. |
| `VLM023_signal` | `volume_nvi_pvi_cross_signal` | Negative Volume Index (NVI) vs Positive Volume Index (PVI) moving average cross. | NVI crosses above its 255-period (or 20-period short) EMA. | NVI crosses below its EMA. |
| `VLM024_signal` | `volume_vwap_anchored_dev1_signal` | Rolling VWAP $\pm 1.0\sigma$ standard deviation band bounce. | Price bounces up from VWAP $-1.0\sigma$ band. | Price rejects down from VWAP $+1.0\sigma$ band. |
| `VLM025_signal` | `volume_vwap_anchored_dev3_signal` | Rolling VWAP $\pm 3.0\sigma$ extreme mean-reversion boundary. | Price touches/dips below VWAP $-3.0\sigma$ band (deep value). | Price touches/exceeds VWAP $+3.0\sigma$ band (deep stretch). |
| `VLM026_signal` | `volume_delta_proxy_surge_signal` | Intrabar Delta Volume Proxy: $V_{buy} = V \cdot \frac{Close - Low}{High - Low}$. | Estimated Buy Delta $> 70\%$ total volume and $Close > Open$. | Estimated Sell Delta $> 70\%$ total volume and $Close < Open$. |
| `VLM027_signal` | `volume_vwma_sma_divergence_signal` | VWMA(20) vs SMA(20) divergence: high volume accumulation premium. | VWMA(20) crosses above SMA(20) by $> 0.5\%$. | VWMA(20) crosses below SMA(20) by $> 0.5\%$. |
| `VLM028_signal` | `volume_volume_weighted_rsi_14_signal` | Volume-Weighted RSI (14 period) overbought/oversold boundaries. | V-RSI(14) crosses above 30 from oversold. | V-RSI(14) crosses below 70 from overbought. |

---

### 3.4 Volatility Signals (+10 Signals, `VOL033` - `VOL042`)

| Code | Semantic Name | Description & Formula | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `VOL033_signal` | `vol_rvi_ob_os_14_signal` | Relative Volatility Index (RVI 14) overbought ($>70$) / oversold ($<30$) thresholds. | RVI 14 crosses above 30. | RVI 14 crosses below 70. |
| `VOL034_signal` | `vol_garman_klass_expansion_signal` | Garman-Klass Volatility Estimator 20-period surge $> 90$th percentile. | GK Volatility surge with bullish price return ($Close > Close_{t-1}$). | GK Volatility surge with bearish price return ($Close < Close_{t-1}$). |
| `VOL035_signal` | `vol_parkinson_volatility_surge_signal` | Parkinson High-Low Volatility Estimator expansion with directional move. | Parkinson Volatility $> 1.8 \times SMA20$ and $Close > Open$. | Parkinson Volatility $> 1.8 \times SMA20$ and $Close < Open$. |
| `VOL036_signal` | `vol_squeeze_momentum_pro_signal` | Squeeze Momentum Pro: BB(20, 2.0) inside KC(20, 1.5) squeeze release + LinReg slope. | Squeeze releases and LinReg slope is positive/increasing. | Squeeze releases and LinReg slope is negative/decreasing. |
| `VOL037_signal` | `vol_keltner_width_squeeze_signal` | Keltner Channel Bandwidth compression ($Width < 0.7 \times SMA20(Width)$). | KC Width compression with price above KC midline. | KC Width compression with price below KC midline. |
| `VOL038_signal` | `vol_atr_ratio_fast_slow_signal` | ATR Fast/Slow Ratio: $ATR(5) / ATR(20) > 1.4$ volatility explosion. | ATR ratio $> 1.4$ with positive price momentum. | ATR ratio $> 1.4$ with negative price momentum. |
| `VOL039_signal` | `vol_chandelier_exit_reversal_signal` | Chandelier Exit ($Highest(High, 22) - 3.0 \times ATR22$) trailing stop flip. | Price crosses above Chandelier Short Stop. | Price crosses below Chandelier Long Stop. |
| `VOL040_signal` | `vol_mass_index_reversal_bulge_signal` | Mass Index (25 period) Reversal Bulge (Mass Index $> 27.0$ then drops below $26.5$). | Reversal bulge trigger with $EMA9 > EMA9_{t-1}$. | Reversal bulge trigger with $EMA9 < EMA9_{t-1}$. |
| `VOL041_signal` | `vol_normalized_atr_stretch_signal` | Normalized ATR ($NATR = ATR / Close \times 100$) reaching extreme expansion $> 2.0 \times SMA20(NATR)$. | NATR extreme expansion with oversold dip bounce. | NATR extreme expansion with overbought spike rejection. |
| `VOL042_signal` | `vol_dual_thrust_range_breakout_signal` | Dual Thrust Breakout System ($Range = \max(HH-LC, HC-LL)$). | $Close > Open + K_1 \times Range$. | $Close < Open - K_2 \times Range$. |

---

### 3.5 Momentum Signals (+8 Signals, `MOM031` - `MOM038`)

| Code | Semantic Name | Description & Formula | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `MOM031_signal` | `mom_rmi_ob_os_14_signal` | Relative Momentum Index (RMI 14, momentum lookback 5) overbought/oversold. | RMI crosses above 30. | RMI crosses below 70. |
| `MOM032_signal` | `mom_dmi_variable_lookback_signal` | Dynamic Momentum Index (DMI with volatility-adjusted dynamic RSI length). | Dynamic DMI crosses above 30. | Dynamic DMI crosses below 70. |
| `MOM033_signal` | `mom_coppock_curve_zero_cross_signal` | Coppock Curve ($WMA10(ROC14 + ROC11)$) zero centerline crossover. | Coppock Curve crosses above 0. | Coppock Curve crosses below 0. |
| `MOM034_signal` | `mom_stoch_momentum_index_cross_signal` | Stochastic Momentum Index (SMI 13, 25, 2) signal line crossover in extreme zones. | SMI crosses above signal line in oversold zone ($< -40$). | SMI crosses below signal line in overbought zone ($> +40$). |
| `MOM035_signal` | `mom_schaff_trend_cycle_cross_signal` | Schaff Trend Cycle (STC 23, 50, 10) $25 / 75$ threshold crossover. | STC crosses above 25. | STC crosses below 75. |
| `MOM036_signal` | `mom_cmo_divergence_signal` | Chande Momentum Oscillator (CMO 14) 5-bar regular divergence. | Low $< Low_{t-5}$ while CMO $> CMO_{t-5}$ (Bullish CMO divergence). | High $> High_{t-5}$ while CMO $< CMO_{t-5}$ (Bearish CMO divergence). |
| `MOM037_signal` | `mom_kst_oscillator_cross_signal` | Martin Pring's Know Sure Thing (KST) signal line crossover. | KST line crosses above KST signal line. | KST line crosses below KST signal line. |
| `MOM038_signal` | `mom_demarker_indicator_cross_signal` | Tom DeMarker Indicator (DeM 14) $0.3 / 0.7$ boundary reversals. | DeM 14 crosses above 0.3 from oversold. | DeM 14 crosses below 0.7 from overbought. |

---

### 3.6 Statistical Signals (+4 Signals, `STA017` - `STA020`)

| Code | Semantic Name | Description & Formula | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `STA017_signal` | `stat_fractal_dimension_index_signal` | Fractal Dimension Index (FDI 30): $FDI < 1.5$ indicates strong trend, $FDI > 1.5$ indicates mean-reverting chop. | $FDI < 1.45$ and $Close > SMA20$ (Trending bullish breakout). | $FDI < 1.45$ and $Close < SMA20$ (Trending bearish breakdown). |
| `STA018_signal` | `stat_rolling_half_life_reversion_signal` | Ornstein-Uhlenbeck Process Half-Life ($HalfLife = -\frac{\ln(2)}{\lambda}$ from regression $\Delta P_t = \lambda P_{t-1} + \epsilon$). | $HalfLife \in [3, 15]$ (Strong mean-reverting regime) and Price $Z < -1.8$. | $HalfLife \in [3, 15]$ (Strong mean-reverting regime) and Price $Z > +1.8$. |
| `STA019_signal` | `stat_variance_ratio_test_signal` | Lo-MacKinlay Variance Ratio ($VR(q) = \frac{\sigma^2(q)}{q \cdot \sigma^2(1)}$): $VR > 1$ trending, $VR < 1$ mean-reverting. | $VR(5) > 1.25$ and price momentum positive. | $VR(5) > 1.25$ and price momentum negative. |
| `STA020_signal` | `stat_rolling_skewness_reversal_signal` | Rolling 20-period Return Skewness extreme reversal ($Skew < -1.5$ or $> +1.5$). | Negative return skewness $Skew < -1.5$ with positive price reversal. | Positive return skewness $Skew > +1.5$ with negative price reversal. |

---

### 3.7 Composite Signals (+4 Signals, `CMP009` - `CMP012`)

| Code | Semantic Name | Description & Formula | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `CMP009_signal` | `comp_smc_trend_volume_confluence_signal` | SMC Confluence: FVG or OrderBlock mitigation + Trend Alignment + Volume surge confirmation. | Bullish FVG/OB active + $SMA20 > SMA50$ + Volume $> SMA20(Volume)$. | Bearish FVG/OB active + $SMA20 < SMA50$ + Volume $> SMA20(Volume)$. |
| `CMP010_signal` | `comp_triple_screen_trading_system_signal` | Alexander Elder Triple Screen: Trend Screen (EMA50 slope) + Oscillator Screen (RSI/Stoch pullback) + Entry Trigger (Price breakout). | EMA50 rising + RSI $< 40$ + Price breaks previous bar high. | EMA50 falling + RSI $> 60$ + Price breaks previous bar low. |
| `CMP011_signal` | `comp_squeeze_momentum_volume_surge_signal` | Squeeze Momentum Pro firing + Volume expansion surge $> 1.5 \times SMA20$. | Squeeze Momentum fires Bullish + Volume $> 1.5 \times SMA20(Volume)$. | Squeeze Momentum fires Bearish + Volume $> 1.5 \times SMA20(Volume)$. |
| `CMP012_signal` | `comp_master_ensemble_v2_signal` | Master Ensemble v2: Multi-factor weighted consensus across all 230 library signals ($\ge 30\%$ threshold across active signals). | $\ge 30\%$ of all library signals vote BUY. | $\ge 30\%$ of all library signals vote SELL. |

---

## 4. Implementation Strategy & File Structure

### Files to Modify:
1. `src/signalx/signals/candlestick.py`: Implement `CDL028` through `CDL037`.
2. `src/signalx/signals/trend.py`: Implement `TRD042` through `TRD053`.
3. `src/signalx/signals/volume.py`: Implement `VLM019` through `VLM028`.
4. `src/signalx/signals/volatility.py`: Implement `VOL033` through `VOL042`.
5. `src/signalx/signals/momentum.py`: Implement `MOM031` through `MOM038`.
6. `src/signalx/signals/statistical.py`: Implement `STA017` through `STA020`.
7. `src/signalx/signals/composite.py`: Implement `CMP009` through `CMP012`.
8. `src/signalx/metadata.py`: Register all 58 new signals in `SIGNAL_CATALOG` and `SIGNAL_CODE_CATALOG`.
9. `tests/`: Update unit test fixtures and assertions across all `test_signals_*.py`, `test_metadata.py`, and `test_full_pipeline.py`.
10. `docs/.ai/SIGNALS_CATALOG.md` & `docs/.ai/`: Update documentation counts and catalogs, then run `bash scripts/generate_agents_markdown.sh`.

---

## 5. Verification & Test Plan

1. **Sub-second Execution Test**:
   - `uv run pytest -v` must execute all unit tests in $<1.0$s.
2. **Contract State Compliance**:
   - Every column in output dataframe must contain only `{"buy", "sell", "hold", "none"}`.
   - Zero `NaN` or unhandled exceptions during warmup periods.
3. **Column Count & Postfix Verification**:
   - `len(signal_columns) == 230`.
   - $100\%$ of signal columns end with `_signal`.
4. **Zero Future Leakage Check**:
   - Verify all rolling, diff, and shift calculations use only historical bars ($\le t$).
5. **Linting & Type Safety**:
   - `uv run ruff check .` passes with 0 errors.
   - `uv run ruff format --check .` passes with 0 format errors.
