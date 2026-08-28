# SignalX Signals Catalog (114 Signals)

`signalx` provides 114 standardized trading signals partitioned across 7 distinct analytical families. Every signal strictly outputs values from `{"buy", "sell", "hold", "none"}`.

## Summary by Category

| Category | Count | Primary Focus |
| :--- | :--- | :--- |
| **Candlestick** | 14 | Price action geometry, rejection wicks, and single/multi-bar reversal formations |
| **Composite** | 7 | Consensus voting, trend/momentum confluence, and multi-indicator ensembles |
| **Momentum** | 23 | Oscillators, overbought/oversold boundaries, and speed of price change |
| **Statistical** | 11 | Rolling Z-scores, linear regression slope/crossings, and market efficiency filters |
| **Trend** | 30 | Directional trend following, moving average crossovers, MACD, and regime tracking |
| **Volatility** | 17 | Band breakouts, volatility squeezes, channel bounds, and ATR trailing stops |
| **Volume** | 12 | Volume dynamics, flow accumulation/distribution, VWAP, and volume spikes |
| **Total** | **114** | **Full Quantitative Feature Suite** |

## Candlestick Signals (14 Signals)

| Signal Name | Description | Library | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `cdl_engulfing_signal` | Bullish and Bearish Engulfing 2-bar pattern | `signalx_native` | Bullish Engulfing: green body completely engulfs prior red body | Bearish Engulfing: red body completely engulfs prior green body |
| `cdl_hammer_star_signal` | Hammer, Inverted Hammer, Shooting Star, and Hanging Man patterns | `signalx_native` | Hammer / Inverted Hammer with long lower shadow rejection | Shooting Star / Hanging Man with long upper shadow rejection |
| `cdl_pinbar_signal` | Pinbar / Price rejection bar (wick >= 2x body) | `signalx_native` | Bullish pinbar: lower wick >= 60% of candle range | Bearish pinbar: upper wick >= 60% of candle range |
| `cdl_marubozu_signal` | Marubozu strong momentum directional candle (body >= 85% range) | `signalx_native` | Bullish Marubozu (large green body, tiny wicks) | Bearish Marubozu (large red body, tiny wicks) |
| `cdl_harami_signal` | Harami inside body pattern (Bullish / Bearish Harami) | `signalx_native` | Bullish Harami: small green body inside prior large red body | Bearish Harami: small red body inside prior large green body |
| `cdl_inside_bar_breakout_signal` | Inside Bar breakout (current bar contained within prior bar range) | `signalx_native` | Inside bar closes bullish (Close > Open) | Inside bar closes bearish (Close < Open) |
| `cdl_outside_bar_signal` | Outside Bar breakout (High > prev High and Low < prev Low) | `signalx_native` | Outside bar closes bullish (Close > Open) | Outside bar closes bearish (Close < Open) |
| `cdl_doji_reversal_signal` | Doji reversal pattern (Dragonfly Doji vs Gravestone Doji) | `signalx_native` | Dragonfly Doji: tiny body with long lower shadow | Gravestone Doji: tiny body with long upper shadow |
| `cdl_three_soldiers_crows_signal` | Three White Soldiers (bullish) and Three Black Crows (bearish) pattern | `signalx_native` | Three White Soldiers: 3 consecutive expanding green candles | Three Black Crows: 3 consecutive expanding red candles |
| `cdl_consecutive_3_signal` | Three consecutive directional bars in same direction | `signalx_native` | 3 consecutive bullish closes (Close > Open) | 3 consecutive bearish closes (Close < Open) |
| `cdl_consecutive_5_signal` | Five consecutive directional bars in same direction | `signalx_native` | 5 consecutive bullish closes (Close > Open) | 5 consecutive bearish closes (Close < Open) |
| `cdl_morning_evening_star_signal` | Morning Star (bullish) and Evening Star (bearish) 3-bar reversal pattern | `signalx_native` | Morning Star: large red candle, small gapping body, strong green candle | Evening Star: large green candle, small gapping body, strong red candle |
| `cdl_piercing_darkcloud_signal` | Piercing Line and Dark Cloud Cover reversal pattern | `signalx_native` | Piercing Line: green candle closes > 50% into prior red body | Dark Cloud Cover: red candle closes > 50% into prior green body |
| `cdl_tweezer_tops_bottoms_signal` | Tweezer Top (bearish) and Tweezer Bottom (bullish) dual matching shadow reversal | `signalx_native` | Tweezer Bottom: matching lows with bullish second candle | Tweezer Top: matching highs with bearish second candle |

## Composite Signals (7 Signals)

| Signal Name | Description | Library | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `comp_trend_consensus_signal` | Trend Family Consensus: majority vote across all trend signals | `signalx_native` | >50% of trend signals are BUY | >50% of trend signals are SELL |
| `comp_momentum_consensus_signal` | Momentum Family Consensus: majority vote across all momentum signals | `signalx_native` | >50% of momentum signals are BUY | >50% of momentum signals are SELL |
| `comp_master_ensemble_signal` | Master Ensemble: weighted consensus across all library signals | `signalx_native` | >=35% of all active signals vote BUY | >=35% of all active signals vote SELL |
| `comp_ma_consensus_signal` | Moving Average Consensus across multiple SMA/EMA periods | `signalx_native` | Majority of MA signals indicate bullish trend | Majority of MA signals indicate bearish trend |
| `comp_trend_momentum_align_signal` | Trend & Momentum Alignment confluence | `signalx_native` | Both Trend consensus and Momentum consensus are BUY | Both Trend consensus and Momentum consensus are SELL |
| `comp_breakout_volume_confirmed_signal` | Price breakout confirmed with volume expansion | `signalx_native` | Volatility breakout (Donchian/BB) confirmed with Volume Spike | Volatility breakdown (Donchian/BB) confirmed with Volume Spike |
| `comp_mean_reversion_confluence_signal` | Multi-oscillator mean reversion confluence (RSI + BB lower + Z-Score) | `signalx_native` | Confluence of Oversold RSI (<30), lower BB touch, and Price Z-score < -2.0 | Confluence of Overbought RSI (>70), upper BB touch, and Price Z-score > +2.0 |

## Momentum Signals (23 Signals)

| Signal Name | Description | Library | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `mom_rsi_ob_os_14_signal` | RSI (14 period) overbought (>70) and oversold (<30) thresholds | `ta` | RSI 14 enters oversold zone (<30) or bounces back above 30 | RSI 14 enters overbought zone (>70) or drops back below 70 |
| `mom_rsi_ob_os_7_signal` | Fast RSI (7 period) overbought (>80) and oversold (<20) thresholds | `ta` | RSI 7 enters oversold zone (<20) or bounces back above 20 | RSI 7 enters overbought zone (>80) or drops back below 80 |
| `mom_rsi_ob_os_21_signal` | Medium RSI (21 period) overbought (>70) and oversold (<30) thresholds | `ta` | RSI 21 enters oversold zone (<30) or bounces back above 30 | RSI 21 enters overbought zone (>70) or drops back below 70 |
| `mom_rsi_ob_os_28_signal` | Slow RSI (28 period) overbought (>70) and oversold (<30) thresholds | `ta` | RSI 28 enters oversold zone (<30) or bounces back above 30 | RSI 28 enters overbought zone (>70) or drops back below 70 |
| `mom_rsi_50_cross_14_signal` | RSI 14 centerline 50 momentum shift | `ta` | RSI 14 crosses above 50 | RSI 14 crosses below 50 |
| `mom_rsi_50_cross_21_signal` | RSI 21 centerline 50 momentum shift | `ta` | RSI 21 crosses above 50 | RSI 21 crosses below 50 |
| `mom_stoch_kd_cross_14_3_3_signal` | Stochastic (14, 3, 3) %K/%D crossover in extreme zones | `ta` | %K crosses above %D in oversold zone (<25) | %K crosses below %D in overbought zone (>75) |
| `mom_stoch_kd_cross_5_3_3_signal` | Fast Stochastic (5, 3, 3) %K/%D crossover in extreme zones | `ta` | Fast %K crosses above %D in oversold zone (<25) | Fast %K crosses below %D in overbought zone (>75) |
| `mom_stoch_rsi_cross_14_signal` | StochRSI (14 period) %K/%D crossover in extreme zones | `ta` | StochRSI %K crosses above %D in oversold zone | StochRSI %K crosses below %D in overbought zone |
| `mom_williams_r_14_signal` | Williams %R (14 period) overbought (>-20) and oversold (<-80) levels | `ta` | Williams %R 14 drops below -80 | Williams %R 14 rises above -20 |
| `mom_williams_r_28_signal` | Williams %R (28 period) overbought (>-20) and oversold (<-80) levels | `ta` | Williams %R 28 drops below -80 | Williams %R 28 rises above -20 |
| `mom_cci_100_14_signal` | Commodity Channel Index (14 period) +/-100 boundary breakouts | `ta` | CCI 14 crosses above +100 | CCI 14 crosses below -100 |
| `mom_cci_200_20_signal` | Commodity Channel Index (20 period) +/-200 extreme breakouts | `ta` | CCI 20 crosses above +200 | CCI 20 crosses below -200 |
| `mom_roc_zero_cross_5_signal` | Rate of Change (5 period) zero centerline crossover | `ta` | ROC 5 crosses above 0 | ROC 5 crosses below 0 |
| `mom_roc_zero_cross_10_signal` | Rate of Change (10 period) zero centerline crossover | `ta` | ROC 10 crosses above 0 | ROC 10 crosses below 0 |
| `mom_roc_zero_cross_20_signal` | Rate of Change (20 period) zero centerline crossover | `ta` | ROC 20 crosses above 0 | ROC 20 crosses below 0 |
| `mom_mfi_ob_os_14_signal` | Money Flow Index (14 period) volume-weighted overbought/oversold levels | `ta` | MFI 14 drops below 20 (oversold) | MFI 14 rises above 80 (overbought) |
| `mom_tsi_cross_13_25_signal` | True Strength Index (13, 25) zero line crossover | `ta` | TSI crosses above 0 | TSI crosses below 0 |
| `mom_fisher_cross_9_signal` | Fisher Transform (9 period) trigger line crossover | `pandas_ta` | Fisher Transform crosses above signal line | Fisher Transform crosses below signal line |
| `mom_ao_zero_cross_signal` | Awesome Oscillator centerline zero crossover | `ta` | Awesome Oscillator crosses above 0 | Awesome Oscillator crosses below 0 |
| `mom_ao_saucer_signal` | Awesome Oscillator saucer setup pattern | `pandas_ta` | AO bullish saucer pattern above zero | AO bearish saucer pattern below zero |
| `mom_ultimate_osc_signal` | Ultimate Oscillator (7, 14, 28) boundary extremes | `ta` | Ultimate Oscillator < 30 (oversold) | Ultimate Oscillator > 70 (overbought) |
| `mom_cmo_14_signal` | Chande Momentum Oscillator (14 period) +/-50 threshold levels | `pandas_ta` | CMO 14 crosses above +50 | CMO 14 crosses below -50 |

## Statistical Signals (11 Signals)

| Signal Name | Description | Library | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `stat_price_zscore_10_signal` | 10-period rolling price Z-score (+/-2.0 std bounds) | `scipy` | Price Z-Score 10 < -2.0 (statistically oversold) | Price Z-Score 10 > +2.0 (statistically overbought) |
| `stat_price_zscore_20_signal` | 20-period rolling price Z-score (+/-2.0 std bounds) | `scipy` | Price Z-Score 20 < -2.0 (statistically oversold) | Price Z-Score 20 > +2.0 (statistically overbought) |
| `stat_price_zscore_50_signal` | 50-period rolling price Z-score (+/-2.0 std bounds) | `scipy` | Price Z-Score 50 < -2.0 (statistically oversold) | Price Z-Score 50 > +2.0 (statistically overbought) |
| `stat_price_zscore_100_signal` | 100-period rolling price Z-score (+/-2.0 std bounds) | `scipy` | Price Z-Score 100 < -2.0 (statistically oversold) | Price Z-Score 100 > +2.0 (statistically overbought) |
| `stat_return_zscore_20_signal` | 20-period rolling return Z-score (+/-2.0 std bounds) | `scipy` | Return Z-Score 20 < -2.0 (extreme return drawdown) | Return Z-Score 20 > +2.0 (extreme return surge) |
| `stat_ker_trend_filter_10_signal` | Kaufman Efficiency Ratio (10 period) trend efficiency filter | `signalx_native` | KER 10 > 0.6 and net direction is positive | KER 10 > 0.6 and net direction is negative |
| `stat_ker_trend_filter_20_signal` | Kaufman Efficiency Ratio (20 period) trend efficiency filter | `signalx_native` | KER 20 > 0.6 and net direction is positive | KER 20 > 0.6 and net direction is negative |
| `stat_chop_regime_14_signal` | Choppiness Index (14 period) trending (<38.2) vs consolidating (>61.8) regime | `ta` | Choppiness Index < 38.2 (strong directional trend mode) | Choppiness Index > 61.8 (consolidating / chop mode) |
| `stat_rolling_quantile_extremes_20_signal` | 20-period rolling quantile extremes (5th and 95th percentiles) | `pandas` | Price <= 5th percentile of 20-period window | Price >= 95th percentile of 20-period window |
| `stat_linreg_slope_14_signal` | 14-period linear regression slope direction | `scipy` | Linear regression slope > 0 with statistically significant t-stat | Linear regression slope < 0 with statistically significant t-stat |
| `stat_linreg_price_cross_30_signal` | 30-period linear regression trendline price crossover | `scipy` | Price crosses above 30-period linear regression line | Price crosses below 30-period linear regression line |

## Trend Signals (30 Signals)

| Signal Name | Description | Library | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `trend_sma_cross_5_20_signal` | Fast SMA(5) crosses Slow SMA(20) | `ta` | SMA 5 crosses above SMA 20 | SMA 5 crosses below SMA 20 |
| `trend_sma_cross_10_50_signal` | Medium SMA(10) crosses Slow SMA(50) | `ta` | SMA 10 crosses above SMA 50 | SMA 10 crosses below SMA 50 |
| `trend_sma_cross_20_50_signal` | SMA(20) crosses Slow SMA(50) | `ta` | SMA 20 crosses above SMA 50 | SMA 20 crosses below SMA 50 |
| `trend_golden_cross_50_200_signal` | Classic Golden / Death Cross of SMA(50) and SMA(200) | `ta` | SMA 50 crosses above SMA 200 (Golden Cross) | SMA 50 crosses below SMA 200 (Death Cross) |
| `trend_ema_cross_9_21_signal` | Fast EMA(9) crosses Slow EMA(21) | `ta` | EMA 9 crosses above EMA 21 | EMA 9 crosses below EMA 21 |
| `trend_ema_cross_12_26_signal` | Standard MACD-base EMA(12) crosses EMA(26) | `ta` | EMA 12 crosses above EMA 26 | EMA 12 crosses below EMA 26 |
| `trend_ema_cross_50_200_signal` | Exponential Golden / Death Cross of EMA(50) and EMA(200) | `ta` | EMA 50 crosses above EMA 200 | EMA 50 crosses below EMA 200 |
| `trend_dema_cross_10_30_signal` | Double Exponential Moving Average (DEMA 10/30) cross | `pandas_ta` | DEMA 10 crosses above DEMA 30 | DEMA 10 crosses below DEMA 30 |
| `trend_tema_cross_10_30_signal` | Triple Exponential Moving Average (TEMA 10/30) cross | `pandas_ta` | TEMA 10 crosses above TEMA 30 | TEMA 10 crosses below TEMA 30 |
| `trend_hma_cross_9_21_signal` | Hull Moving Average (HMA 9/21) crossover | `pandas_ta` | HMA 9 crosses above HMA 21 | HMA 9 crosses below HMA 21 |
| `trend_vwma_cross_10_30_signal` | Volume-Weighted Moving Average (VWMA 10/30) cross | `pandas_ta` | VWMA 10 crosses above VWMA 30 | VWMA 10 crosses below VWMA 30 |
| `trend_price_above_sma20_signal` | Price position relative to SMA 20 baseline | `ta` | Close is above SMA 20 | Close is below SMA 20 |
| `trend_price_above_ema50_signal` | Price position relative to intermediate EMA 50 | `ta` | Close is above EMA 50 | Close is below EMA 50 |
| `trend_price_above_ema200_signal` | Price position relative to institutional EMA 200 | `ta` | Close is above EMA 200 | Close is below EMA 200 |
| `trend_macd_cross_signal` | Standard MACD (12, 26, 9) signal line crossover | `ta` | MACD line crosses above MACD signal line | MACD line crosses below MACD signal line |
| `trend_macd_zero_cross_signal` | MACD (12, 26) line crossover of zero centerline | `ta` | MACD line crosses above zero | MACD line crosses below zero |
| `trend_macd_hist_reversal_signal` | MACD histogram sign flip / momentum reversal | `ta` | MACD histogram turns positive from negative | MACD histogram turns negative from positive |
| `trend_macd_fast_cross_signal` | Fast MACD (6, 13, 5) signal line crossover | `ta` | Fast MACD line crosses above fast MACD signal line | Fast MACD line crosses below fast MACD signal line |
| `trend_macd_slow_cross_signal` | Slow MACD (19, 39, 9) signal line crossover | `ta` | Slow MACD line crosses above slow MACD signal line | Slow MACD line crosses below slow MACD signal line |
| `trend_supertrend_10_3_signal` | SuperTrend (length 10, multiplier 3.0) direction regime | `pandas_ta` | SuperTrend direction is positive (bullish) | SuperTrend direction is negative (bearish) |
| `trend_supertrend_7_2_signal` | Fast SuperTrend (length 7, multiplier 2.0) direction regime | `pandas_ta` | Fast SuperTrend direction is positive | Fast SuperTrend direction is negative |
| `trend_supertrend_14_4_signal` | Slow SuperTrend (length 14, multiplier 4.0) direction regime | `pandas_ta` | Slow SuperTrend direction is positive | Slow SuperTrend direction is negative |
| `trend_psar_reversal_signal` | Parabolic SAR trailing stop reversal | `ta` | Parabolic SAR dots flip below price | Parabolic SAR dots flip above price |
| `trend_aroon_cross_14_signal` | Aroon Oscillator (14 period) Up/Down crossover | `ta` | Aroon Up 14 crosses above Aroon Down 14 | Aroon Up 14 crosses below Aroon Down 14 |
| `trend_aroon_cross_25_signal` | Aroon Oscillator (25 period) Up/Down crossover | `ta` | Aroon Up 25 crosses above Aroon Down 25 | Aroon Up 25 crosses below Aroon Down 25 |
| `trend_adx_dmi_14_signal` | ADX/DMI (14 period) directional trend strength filter | `ta` | +DI > -DI and ADX > 25 | -DI > +DI and ADX > 25 |
| `trend_adx_dmi_28_signal` | ADX/DMI (28 period) slow trend strength filter | `ta` | +DI > -DI and ADX > 25 | -DI > +DI and ADX > 25 |
| `trend_ichimoku_tk_cross_signal` | Ichimoku Tenkan-sen and Kijun-sen crossover | `ta` | Tenkan-sen crosses above Kijun-sen | Tenkan-sen crosses below Kijun-sen |
| `trend_ichimoku_cloud_breakout_signal` | Price breakout relative to Ichimoku Kumo Cloud boundaries | `ta` | Close breaks above upper Kumo Cloud span | Close breaks below lower Kumo Cloud span |
| `trend_vortex_cross_14_signal` | Vortex Indicator (14 period) +VI/-VI crossover | `ta` | +VI 14 crosses above -VI 14 | +VI 14 crosses below -VI 14 |

## Volatility Signals (17 Signals)

| Signal Name | Description | Library | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `vol_bb_breakout_20_20_signal` | Bollinger Bands (20, 2.0 std) outer band breakout | `ta` | Close breaks above upper Bollinger Band | Close breaks below lower Bollinger Band |
| `vol_bb_bounce_20_20_signal` | Bollinger Bands (20, 2.0 std) mean-reverting bounce | `ta` | Price dips below lower band then bounces back inside | Price rises above upper band then rejects back inside |
| `vol_bb_breakout_50_25_signal` | Bollinger Bands (50, 2.5 std) wide channel breakout | `ta` | Close breaks above upper Bollinger Band 50/2.5 | Close breaks below lower Bollinger Band 50/2.5 |
| `vol_bb_bounce_50_25_signal` | Bollinger Bands (50, 2.5 std) wide channel mean-reversion bounce | `ta` | Price bounces back above lower band 50/2.5 | Price bounces back below upper band 50/2.5 |
| `vol_bb_pct_b_reversal_20_signal` | Bollinger %B (20 period) extreme boundary reversal | `ta` | %B crosses back above 0.0 | %B crosses back below 1.0 |
| `vol_bb_pct_b_reversal_50_signal` | Bollinger %B (50 period) extreme boundary reversal | `ta` | %B crosses back above 0.0 | %B crosses back below 1.0 |
| `vol_donchian_breakout_10_signal` | Fast Donchian Channel (10 period) high/low channel breakout | `ta` | Close >= 10-period highest high | Close <= 10-period lowest low |
| `vol_donchian_breakout_20_signal` | Standard Donchian Channel (20 period) Turtle breakout System 1 | `ta` | Close >= 20-period highest high | Close <= 20-period lowest low |
| `vol_donchian_breakout_55_signal` | Slow Donchian Channel (55 period) Turtle breakout System 2 | `ta` | Close >= 55-period highest high | Close <= 55-period lowest low |
| `vol_keltner_breakout_20_15_signal` | Keltner Channel (20, 1.5 ATR) channel breakout | `ta` | Close breaks above upper Keltner Channel | Close breaks below lower Keltner Channel |
| `vol_keltner_breakout_20_20_signal` | Keltner Channel (20, 2.0 ATR) channel breakout | `ta` | Close breaks above upper Keltner Channel | Close breaks below lower Keltner Channel |
| `vol_ttm_squeeze_signal` | TTM Squeeze breakout: Bollinger Bands contract inside Keltner Channels then expand | `ta` | Squeeze fires and price is above 20-SMA baseline | Squeeze fires and price is below 20-SMA baseline |
| `vol_bb_bandwidth_expansion_signal` | Bollinger Bandwidth surge indicating volatility expansion | `ta` | Bandwidth expands above rolling quantile with upward price | Bandwidth expands above rolling quantile with downward price |
| `vol_atr_trailing_stop_2x_signal` | ATR Trailing Stop (14 period, 2.0x multiplier) price direction | `ta` | Close > ATR(14) 2.0x trailing stop | Close < ATR(14) 2.0x trailing stop |
| `vol_atr_trailing_stop_3x_signal` | ATR Trailing Stop (14 period, 3.0x multiplier) price direction | `ta` | Close > ATR(14) 3.0x trailing stop | Close < ATR(14) 3.0x trailing stop |
| `vol_chaikin_volatility_surge_signal` | Chaikin Volatility indicator surge | `pandas_ta` | Chaikin Volatility surges positive | Chaikin Volatility contracts or turns negative |
| `vol_hv_ratio_breakout_10_30_signal` | Historical Volatility ratio (10/30) breakout | `signalx_native` | HV(10)/HV(30) > 1.5 with positive price return | HV(10)/HV(30) > 1.5 with negative price return |

## Volume Signals (12 Signals)

| Signal Name | Description | Library | Buy Trigger | Sell Trigger |
| :--- | :--- | :--- | :--- | :--- |
| `volume_obv_ema_cross_20_signal` | On-Balance Volume (OBV) crosses its 20-period EMA | `ta` | OBV crosses above EMA(OBV, 20) | OBV crosses below EMA(OBV, 20) |
| `volume_cmf_zero_cross_20_signal` | Chaikin Money Flow (20 period) zero centerline crossover | `ta` | CMF 20 crosses above 0 | CMF 20 crosses below 0 |
| `volume_cmf_threshold_cross_20_signal` | Chaikin Money Flow (20 period) institutional flow +/-0.05 thresholds | `ta` | CMF 20 > +0.05 | CMF 20 < -0.05 |
| `volume_vwap_cross_20_signal` | Rolling 20-period VWAP price crossover | `ta` | Close crosses above 20-period rolling VWAP | Close crosses below 20-period rolling VWAP |
| `volume_vwap_cross_50_signal` | Rolling 50-period VWAP price crossover | `ta` | Close crosses above 50-period rolling VWAP | Close crosses below 50-period rolling VWAP |
| `volume_vwap_cross_100_signal` | Rolling 100-period VWAP price crossover | `ta` | Close crosses above 100-period rolling VWAP | Close crosses below 100-period rolling VWAP |
| `volume_vwap_band_reversal_20_signal` | Rolling 20-period VWAP standard deviation band reversal | `ta` | Price bounces above lower VWAP -2.0 std band | Price rejects below upper VWAP +2.0 std band |
| `volume_spike_direction_20_signal` | Volume Spike (>2.0x 20-SMA) combined with directional candle body | `ta` | Volume > 2.0x 20-SMA with bullish candle (Close > Open) | Volume > 2.0x 20-SMA with bearish candle (Close < Open) |
| `volume_pvt_ma_cross_14_signal` | Price Volume Trend (PVT) crosses 14-period SMA | `ta` | PVT crosses above SMA(PVT, 14) | PVT crosses below SMA(PVT, 14) |
| `volume_adl_ma_cross_signal` | Accumulation / Distribution Line (ADL) crosses 20-period SMA | `ta` | ADL crosses above SMA(ADL, 20) | ADL crosses below SMA(ADL, 20) |
| `volume_force_index_13_signal` | Elder's Force Index (13 period) zero line crossover | `ta` | Force Index 13 crosses above 0 | Force Index 13 crosses below 0 |
| `volume_eom_zero_14_signal` | Ease of Movement (14 period) zero line crossover | `ta` | Ease of Movement 14 crosses above 0 | Ease of Movement 14 crosses below 0 |
