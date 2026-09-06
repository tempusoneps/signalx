from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "trend",
        "momentum",
        "volatility",
        "volume",
        "candlestick",
        "smc",
        "statistical",
        "composite",
    }
)
SIGNAL_CATEGORIES: frozenset[str] = VALID_CATEGORIES


@dataclass(frozen=True)
class SignalMetadata:
    """Metadata describing a standardized trading signal."""

    code: str
    name: str
    category: str
    description: str
    library: str
    buy_trigger: str
    sell_trigger: str


SIGNAL_CATALOG: dict[str, SignalMetadata] = {}
SIGNAL_CODE_CATALOG: dict[str, SignalMetadata] = {}


def register_signal(
    code: str,
    name: str,
    category: str,
    description: str,
    library: str,
    buy_trigger: str,
    sell_trigger: str,
) -> None:
    """Register a signal in global catalogs with full metadata validation."""
    if not isinstance(code, str) or not code.endswith("_signal") or not code.strip():
        raise ValueError(f"Signal code '{code}' must end with '_signal'")
    if not isinstance(name, str) or not name.endswith("_signal") or not name.strip():
        raise ValueError(f"Signal name '{name}' must end with '_signal'")
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of {sorted(list(VALID_CATEGORIES))}"
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description cannot be empty")
    if not isinstance(library, str) or not library.strip():
        raise ValueError("library cannot be empty")
    if not isinstance(buy_trigger, str) or not buy_trigger.strip():
        raise ValueError("buy_trigger cannot be empty")
    if not isinstance(sell_trigger, str) or not sell_trigger.strip():
        raise ValueError("sell_trigger cannot be empty")

    meta = SignalMetadata(
        code=code.strip(),
        name=name.strip(),
        category=category,
        description=description.strip(),
        library=library.strip(),
        buy_trigger=buy_trigger.strip(),
        sell_trigger=sell_trigger.strip(),
    )
    SIGNAL_CATALOG[name] = meta
    SIGNAL_CODE_CATALOG[code] = meta


def get_signals_by_category(category: str) -> list[SignalMetadata]:
    """Retrieve all registered signal metadata objects belonging to a category."""
    return [meta for meta in SIGNAL_CATALOG.values() if meta.category == category]


def list_categories() -> list[str]:
    """Return a sorted list of all valid signal categories."""
    return sorted(list(VALID_CATEGORIES))


def get_signal_by_code(code: str) -> SignalMetadata | None:
    """Retrieve metadata for a specific signal code, or None if not found."""
    return SIGNAL_CODE_CATALOG.get(code)


def get_signal_by_name(name: str) -> SignalMetadata | None:
    """Retrieve metadata for a specific signal name, or None if not found."""
    return SIGNAL_CATALOG.get(name)


def get_signal_metadata(key: str) -> SignalMetadata | None:
    """Retrieve metadata by signal code first, then by signal name, or None if not found."""
    return SIGNAL_CODE_CATALOG.get(key) or SIGNAL_CATALOG.get(key)


def get_code_to_name_map() -> dict[str, str]:
    """Return mapping from coded signal names to semantic signal names."""
    return {meta.code: meta.name for meta in SIGNAL_CATALOG.values()}


def get_name_to_code_map() -> dict[str, str]:
    """Return mapping from semantic signal names to coded signal names."""
    return {meta.name: meta.code for meta in SIGNAL_CATALOG.values()}


def to_code_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any semantic signal columns in DataFrame to coded names, preserving non-signal columns."""
    return df.rename(columns=get_name_to_code_map())


def to_semantic_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any coded signal columns in DataFrame to semantic names, preserving non-signal columns."""
    return df.rename(columns=get_code_to_name_map())


def _initialize_default_catalog() -> None:
    """Populate the default catalog with 100+ standard trading signals across all 8 categories."""
    # -------------------------------------------------------------------------
    # 1. TREND SIGNALS (30 signals)
    # -------------------------------------------------------------------------
    trend_definitions = [
        (
            "trend_sma_cross_5_20_signal",
            "Fast SMA(5) crosses Slow SMA(20)",
            "ta",
            "SMA 5 crosses above SMA 20",
            "SMA 5 crosses below SMA 20",
        ),
        (
            "trend_sma_cross_10_50_signal",
            "Medium SMA(10) crosses Slow SMA(50)",
            "ta",
            "SMA 10 crosses above SMA 50",
            "SMA 10 crosses below SMA 50",
        ),
        (
            "trend_sma_cross_20_50_signal",
            "SMA(20) crosses Slow SMA(50)",
            "ta",
            "SMA 20 crosses above SMA 50",
            "SMA 20 crosses below SMA 50",
        ),
        (
            "trend_golden_cross_50_200_signal",
            "Classic Golden / Death Cross of SMA(50) and SMA(200)",
            "ta",
            "SMA 50 crosses above SMA 200 (Golden Cross)",
            "SMA 50 crosses below SMA 200 (Death Cross)",
        ),
        (
            "trend_ema_cross_9_21_signal",
            "Fast EMA(9) crosses Slow EMA(21)",
            "ta",
            "EMA 9 crosses above EMA 21",
            "EMA 9 crosses below EMA 21",
        ),
        (
            "trend_ema_cross_12_26_signal",
            "Standard MACD-base EMA(12) crosses EMA(26)",
            "ta",
            "EMA 12 crosses above EMA 26",
            "EMA 12 crosses below EMA 26",
        ),
        (
            "trend_ema_cross_50_200_signal",
            "Exponential Golden / Death Cross of EMA(50) and EMA(200)",
            "ta",
            "EMA 50 crosses above EMA 200",
            "EMA 50 crosses below EMA 200",
        ),
        (
            "trend_dema_cross_10_30_signal",
            "Double Exponential Moving Average (DEMA 10/30) cross",
            "pandas_ta",
            "DEMA 10 crosses above DEMA 30",
            "DEMA 10 crosses below DEMA 30",
        ),
        (
            "trend_tema_cross_10_30_signal",
            "Triple Exponential Moving Average (TEMA 10/30) cross",
            "pandas_ta",
            "TEMA 10 crosses above TEMA 30",
            "TEMA 10 crosses below TEMA 30",
        ),
        (
            "trend_hma_cross_9_21_signal",
            "Hull Moving Average (HMA 9/21) crossover",
            "pandas_ta",
            "HMA 9 crosses above HMA 21",
            "HMA 9 crosses below HMA 21",
        ),
        (
            "trend_vwma_cross_10_30_signal",
            "Volume-Weighted Moving Average (VWMA 10/30) cross",
            "pandas_ta",
            "VWMA 10 crosses above VWMA 30",
            "VWMA 10 crosses below VWMA 30",
        ),
        (
            "trend_price_above_sma20_signal",
            "Price position relative to SMA 20 baseline",
            "ta",
            "Close is above SMA 20",
            "Close is below SMA 20",
        ),
        (
            "trend_price_above_ema50_signal",
            "Price position relative to intermediate EMA 50",
            "ta",
            "Close is above EMA 50",
            "Close is below EMA 50",
        ),
        (
            "trend_price_above_ema200_signal",
            "Price position relative to institutional EMA 200",
            "ta",
            "Close is above EMA 200",
            "Close is below EMA 200",
        ),
        (
            "trend_macd_cross_signal",
            "Standard MACD (12, 26, 9) signal line crossover",
            "ta",
            "MACD line crosses above MACD signal line",
            "MACD line crosses below MACD signal line",
        ),
        (
            "trend_macd_zero_cross_signal",
            "MACD (12, 26) line crossover of zero centerline",
            "ta",
            "MACD line crosses above zero",
            "MACD line crosses below zero",
        ),
        (
            "trend_macd_hist_reversal_signal",
            "MACD histogram sign flip / momentum reversal",
            "ta",
            "MACD histogram turns positive from negative",
            "MACD histogram turns negative from positive",
        ),
        (
            "trend_macd_fast_cross_signal",
            "Fast MACD (6, 13, 5) signal line crossover",
            "ta",
            "Fast MACD line crosses above fast MACD signal line",
            "Fast MACD line crosses below fast MACD signal line",
        ),
        (
            "trend_macd_slow_cross_signal",
            "Slow MACD (19, 39, 9) signal line crossover",
            "ta",
            "Slow MACD line crosses above slow MACD signal line",
            "Slow MACD line crosses below slow MACD signal line",
        ),
        (
            "trend_supertrend_10_3_signal",
            "SuperTrend (length 10, multiplier 3.0) direction regime",
            "pandas_ta",
            "SuperTrend direction is positive (bullish)",
            "SuperTrend direction is negative (bearish)",
        ),
        (
            "trend_supertrend_7_2_signal",
            "Fast SuperTrend (length 7, multiplier 2.0) direction regime",
            "pandas_ta",
            "Fast SuperTrend direction is positive",
            "Fast SuperTrend direction is negative",
        ),
        (
            "trend_supertrend_14_4_signal",
            "Slow SuperTrend (length 14, multiplier 4.0) direction regime",
            "pandas_ta",
            "Slow SuperTrend direction is positive",
            "Slow SuperTrend direction is negative",
        ),
        (
            "trend_psar_reversal_signal",
            "Parabolic SAR trailing stop reversal",
            "ta",
            "Parabolic SAR dots flip below price",
            "Parabolic SAR dots flip above price",
        ),
        (
            "trend_aroon_cross_14_signal",
            "Aroon Oscillator (14 period) Up/Down crossover",
            "ta",
            "Aroon Up 14 crosses above Aroon Down 14",
            "Aroon Up 14 crosses below Aroon Down 14",
        ),
        (
            "trend_aroon_cross_25_signal",
            "Aroon Oscillator (25 period) Up/Down crossover",
            "ta",
            "Aroon Up 25 crosses above Aroon Down 25",
            "Aroon Up 25 crosses below Aroon Down 25",
        ),
        (
            "trend_adx_dmi_14_signal",
            "ADX/DMI (14 period) directional trend strength filter",
            "ta",
            "+DI > -DI and ADX > 25",
            "-DI > +DI and ADX > 25",
        ),
        (
            "trend_adx_dmi_28_signal",
            "ADX/DMI (28 period) slow trend strength filter",
            "ta",
            "+DI > -DI and ADX > 25",
            "-DI > +DI and ADX > 25",
        ),
        (
            "trend_ichimoku_tk_cross_signal",
            "Ichimoku Tenkan-sen and Kijun-sen crossover",
            "ta",
            "Tenkan-sen crosses above Kijun-sen",
            "Tenkan-sen crosses below Kijun-sen",
        ),
        (
            "trend_ichimoku_cloud_breakout_signal",
            "Price breakout relative to Ichimoku Kumo Cloud boundaries",
            "ta",
            "Close breaks above upper Kumo Cloud span",
            "Close breaks below lower Kumo Cloud span",
        ),
        (
            "trend_vortex_cross_14_signal",
            "Vortex Indicator (14 period) +VI/-VI crossover",
            "ta",
            "+VI 14 crosses above -VI 14",
            "+VI 14 crosses below -VI 14",
        ),
        (
            "trend_trix_cross_15_signal",
            "TRIX (15 period) oscillator crosses its 9-period signal line",
            "signalx_native",
            "TRIX 15 crosses above TRIX signal line",
            "TRIX 15 crosses below TRIX signal line",
        ),
        (
            "trend_kama_reversal_10_signal",
            "Kaufman Adaptive Moving Average (KAMA 10) slope inflection reversal",
            "pandas_ta",
            "KAMA 10 hooks upwards after a downward slope",
            "KAMA 10 hooks downwards after an upward slope",
        ),
        (
            "trend_tma_cross_10_signal",
            "Triangular Moving Average (TMA 10) price crossover",
            "signalx_native",
            "Close crosses above TMA 10",
            "Close crosses below TMA 10",
        ),
        (
            "trend_ma_alignment_20_50_signal",
            "Moving Average Alignment (Close > SMA50 and SMA20 > SMA50)",
            "signalx_native",
            "Close > SMA50 and SMA20 > SMA50 (Bullish trend alignment)",
            "Close < SMA50 and SMA20 < SMA50 (Bearish trend alignment)",
        ),
        (
            "trend_pullback_sma20_50_signal",
            "Pullback to SMA20 within established higher-timeframe trend",
            "signalx_native",
            "Close dips below SMA20 while SMA20 > SMA50 (Bullish pullback)",
            "Close rallies above SMA20 while SMA20 < SMA50 (Bearish pullback)",
        ),
        (
            "trend_micro_trend_3_signal",
            "3-bar micro trend persistence",
            "signalx_native",
            "3 consecutive higher closes (Close > Close[1] > Close[2])",
            "3 consecutive lower closes (Close < Close[1] < Close[2])",
        ),
        (
            "trend_micro_reversal_signal",
            "1-bar micro reversal after 2-bar persistent run",
            "signalx_native",
            "Close > Close[1] after 2 consecutive lower closes",
            "Close < Close[1] after 2 consecutive higher closes",
        ),
        (
            "trend_price_cross_sma20_signal",
            "Price crossover with 20-period Simple Moving Average",
            "signalx_native",
            "Close crosses above SMA 20",
            "Close crosses below SMA 20",
        ),
        (
            "trend_prev_bar_break_signal",
            "Price breakout above previous bar high or below previous bar low",
            "signalx_native",
            "Close > High[1] (Prior high breakout)",
            "Close < Low[1] (Prior low breakdown)",
        ),
        (
            "trend_tii_14_signal",
            "Trend Intensity Index (TII 14) trend regime boundaries",
            "pandas_ta",
            "TII 14 > 80 (Strong upward trend intensity)",
            "TII 14 < 20 (Strong downward trend intensity)",
        ),
        (
            "trend_ehlers_super_smoother_cross_signal",
            "Ehlers 2-Pole Super Smoother Filter (Length 10) price crossover",
            "signalx_native",
            "Close crosses above Ehlers Super Smoother 10",
            "Close crosses below Ehlers Super Smoother 10",
        ),
        (
            "trend_mcginley_dynamic_cross_signal",
            "McGinley Dynamic (14) adaptive moving average price crossover",
            "signalx_native",
            "Close crosses above McGinley Dynamic 14",
            "Close crosses below McGinley Dynamic 14",
        ),
        (
            "trend_gmma_ribbon_expansion_signal",
            "Guppy Multiple Moving Average (GMMA) fast vs slow ribbon expansion",
            "signalx_native",
            "Fast GMMA ribbon expands above Slow GMMA ribbon",
            "Fast GMMA ribbon expands below Slow GMMA ribbon",
        ),
        (
            "trend_gmma_compression_breakout_signal",
            "Guppy GMMA compression ribbon breakout",
            "signalx_native",
            "Close breaks above GMMA ribbon following compression (spread <= 1.5%)",
            "Close breaks below GMMA ribbon following compression (spread <= 1.5%)",
        ),
        (
            "trend_rainbow_ema_alignment_signal",
            "Rainbow 5-EMA (8, 13, 21, 34, 55) full sequential trend alignment",
            "signalx_native",
            "EMA 8 > EMA 13 > EMA 21 > EMA 34 > EMA 55 (Bullish Rainbow)",
            "EMA 8 < EMA 13 < EMA 21 < EMA 34 < EMA 55 (Bearish Rainbow)",
        ),
        (
            "trend_ehlers_instantaneous_trend_signal",
            "Ehlers Instantaneous Trendline price crossover",
            "signalx_native",
            "Close crosses above Instantaneous Trendline",
            "Close crosses below Instantaneous Trendline",
        ),
        (
            "trend_coral_trend_filter_signal",
            "Coral smoothed multi-order EMA filter slope direction",
            "signalx_native",
            "Coral Trend Filter is rising (Slope > 0)",
            "Coral Trend Filter is falling (Slope < 0)",
        ),
        (
            "trend_supertrend_atr_20_5_signal",
            "Conservative slow SuperTrend (20, 5.0) trend direction",
            "pandas_ta",
            "SuperTrend (20, 5.0) bullish direction",
            "SuperTrend (20, 5.0) bearish direction",
        ),
        (
            "trend_donchian_middle_cross_20_signal",
            "Donchian Channel 20 median line price crossover",
            "signalx_native",
            "Close crosses above Donchian 20 median line",
            "Close crosses below Donchian 20 median line",
        ),
        (
            "trend_alligator_lips_jaw_cross_signal",
            "Bill Williams Alligator Lips (5, shift 3) and Jaw (13, shift 8) crossover",
            "signalx_native",
            "Alligator Lips crosses above Alligator Jaw",
            "Alligator Lips crosses below Alligator Jaw",
        ),
        (
            "trend_alma_cross_9_signal",
            "Arnaud Legoux Moving Average (ALMA 9, offset 0.85, sigma 6) price crossover",
            "pandas_ta",
            "Close crosses above ALMA 9",
            "Close crosses below ALMA 9",
        ),
        (
            "trend_zero_lag_ema_cross_21_signal",
            "Zero-Lag EMA (ZLEMA 21) price crossover",
            "signalx_native",
            "Close crosses above ZLEMA 21",
            "Close crosses below ZLEMA 21",
        ),
    ]
    trend_codes = [f"TRD{i:03d}_signal" for i in range(1, 34)] + [
        f"TRD{i:03d}_signal" for i in range(35, 54)
    ]
    for code, (name, desc, lib, buy_t, sell_t) in zip(trend_codes, trend_definitions, strict=True):
        register_signal(
            code=code,
            name=name,
            category="trend",
            description=desc,
            library=lib,
            buy_trigger=buy_t,
            sell_trigger=sell_t,
        )

    # -------------------------------------------------------------------------
    # 2. MOMENTUM SIGNALS (39 signals)
    # -------------------------------------------------------------------------
    mom_definitions = [
        (
            "mom_rsi_ob_os_14_signal",
            "RSI (14 period) overbought (>70) and oversold (<30) thresholds",
            "ta",
            "RSI 14 enters oversold zone (<30) or bounces back above 30",
            "RSI 14 enters overbought zone (>70) or drops back below 70",
        ),
        (
            "mom_rsi_ob_os_7_signal",
            "Fast RSI (7 period) overbought (>80) and oversold (<20) thresholds",
            "ta",
            "RSI 7 enters oversold zone (<20) or bounces back above 20",
            "RSI 7 enters overbought zone (>80) or drops back below 80",
        ),
        (
            "mom_rsi_ob_os_21_signal",
            "Medium RSI (21 period) overbought (>70) and oversold (<30) thresholds",
            "ta",
            "RSI 21 enters oversold zone (<30) or bounces back above 30",
            "RSI 21 enters overbought zone (>70) or drops back below 70",
        ),
        (
            "mom_rsi_ob_os_28_signal",
            "Slow RSI (28 period) overbought (>70) and oversold (<30) thresholds",
            "ta",
            "RSI 28 enters oversold zone (<30) or bounces back above 30",
            "RSI 28 enters overbought zone (>70) or drops back below 70",
        ),
        (
            "mom_rsi_50_cross_14_signal",
            "RSI 14 centerline 50 momentum shift",
            "ta",
            "RSI 14 crosses above 50",
            "RSI 14 crosses below 50",
        ),
        (
            "mom_rsi_50_cross_21_signal",
            "RSI 21 centerline 50 momentum shift",
            "ta",
            "RSI 21 crosses above 50",
            "RSI 21 crosses below 50",
        ),
        (
            "mom_stoch_kd_cross_14_3_3_signal",
            "Stochastic (14, 3, 3) %K/%D crossover in extreme zones",
            "ta",
            "%K crosses above %D in oversold zone (<25)",
            "%K crosses below %D in overbought zone (>75)",
        ),
        (
            "mom_stoch_kd_cross_5_3_3_signal",
            "Fast Stochastic (5, 3, 3) %K/%D crossover in extreme zones",
            "ta",
            "Fast %K crosses above %D in oversold zone (<25)",
            "Fast %K crosses below %D in overbought zone (>75)",
        ),
        (
            "mom_stoch_rsi_cross_14_signal",
            "StochRSI (14 period) %K/%D crossover in extreme zones",
            "ta",
            "StochRSI %K crosses above %D in oversold zone",
            "StochRSI %K crosses below %D in overbought zone",
        ),
        (
            "mom_williams_r_14_signal",
            "Williams %R (14 period) overbought (>-20) and oversold (<-80) levels",
            "ta",
            "Williams %R 14 drops below -80",
            "Williams %R 14 rises above -20",
        ),
        (
            "mom_williams_r_28_signal",
            "Williams %R (28 period) overbought (>-20) and oversold (<-80) levels",
            "ta",
            "Williams %R 28 drops below -80",
            "Williams %R 28 rises above -20",
        ),
        (
            "mom_cci_100_14_signal",
            "Commodity Channel Index (14 period) +/-100 boundary breakouts",
            "ta",
            "CCI 14 crosses above +100",
            "CCI 14 crosses below -100",
        ),
        (
            "mom_cci_200_20_signal",
            "Commodity Channel Index (20 period) +/-200 extreme breakouts",
            "ta",
            "CCI 20 crosses above +200",
            "CCI 20 crosses below -200",
        ),
        (
            "mom_roc_zero_cross_5_signal",
            "Rate of Change (5 period) zero centerline crossover",
            "ta",
            "ROC 5 crosses above 0",
            "ROC 5 crosses below 0",
        ),
        (
            "mom_roc_zero_cross_10_signal",
            "Rate of Change (10 period) zero centerline crossover",
            "ta",
            "ROC 10 crosses above 0",
            "ROC 10 crosses below 0",
        ),
        (
            "mom_roc_zero_cross_20_signal",
            "Rate of Change (20 period) zero centerline crossover",
            "ta",
            "ROC 20 crosses above 0",
            "ROC 20 crosses below 0",
        ),
        (
            "mom_mfi_ob_os_14_signal",
            "Money Flow Index (14 period) volume-weighted overbought/oversold levels",
            "ta",
            "MFI 14 drops below 20 (oversold)",
            "MFI 14 rises above 80 (overbought)",
        ),
        (
            "mom_tsi_cross_13_25_signal",
            "True Strength Index (13, 25) zero line crossover",
            "ta",
            "TSI crosses above 0",
            "TSI crosses below 0",
        ),
        (
            "mom_fisher_cross_9_signal",
            "Fisher Transform (9 period) trigger line crossover",
            "pandas_ta",
            "Fisher Transform crosses above signal line",
            "Fisher Transform crosses below signal line",
        ),
        (
            "mom_ao_zero_cross_signal",
            "Awesome Oscillator centerline zero crossover",
            "ta",
            "Awesome Oscillator crosses above 0",
            "Awesome Oscillator crosses below 0",
        ),
        (
            "mom_ao_saucer_signal",
            "Awesome Oscillator saucer setup pattern",
            "pandas_ta",
            "AO bullish saucer pattern above zero",
            "AO bearish saucer pattern below zero",
        ),
        (
            "mom_ultimate_osc_signal",
            "Ultimate Oscillator (7, 14, 28) boundary extremes",
            "ta",
            "Ultimate Oscillator < 30 (oversold)",
            "Ultimate Oscillator > 70 (overbought)",
        ),
        (
            "mom_cmo_14_signal",
            "Chande Momentum Oscillator (14 period) +/-50 threshold levels",
            "pandas_ta",
            "CMO 14 crosses above +50",
            "CMO 14 crosses below -50",
        ),
        (
            "mom_min_max_10_rsi_signal",
            "10-period price extreme combined with RSI(14) overbought/oversold",
            "signalx_native",
            "Close reaches 10-period low with RSI 14 < 30 (Oversold extreme)",
            "Close reaches 10-period high with RSI 14 > 70 (Overbought extreme)",
        ),
        (
            "mom_rsi_divergence_5_signal",
            "5-bar regular bullish/bearish RSI divergence",
            "signalx_native",
            "Low < Low[5] while RSI > RSI[5] (Bullish regular divergence)",
            "High > High[5] while RSI < RSI[5] (Bearish regular divergence)",
        ),
        (
            "mom_connors_rsi_3_2_100_signal",
            "Connors RSI (3, 2, 100) extreme overbought (>85) / oversold (<15) zones",
            "signalx_native",
            "Connors RSI < 15.0 (Extreme oversold)",
            "Connors RSI > 85.0 (Extreme overbought)",
        ),
        (
            "mom_mfi_reversal_20_80_signal",
            "Money Flow Index (MFI 14) exiting overbought/oversold boundary reversal",
            "signalx_native",
            "MFI 14 < 20 and hooks upward (MFI > MFI[1])",
            "MFI 14 > 80 and hooks downward (MFI < MFI[1])",
        ),
        (
            "mom_shift_3_bar_signal",
            "3-bar short-term momentum turnaround shift",
            "signalx_native",
            "Close > Close[1] after Close[1] < Close[2] (Bullish momentum shift)",
            "Close < Close[1] after Close[1] > Close[2] (Bearish momentum shift)",
        ),
        (
            "mom_return_momentum_5_signal",
            "5-bar cumulative return momentum (> +2% / < -2%)",
            "signalx_native",
            "5-bar return > +2.0%",
            "5-bar return < -2.0%",
        ),
        (
            "mom_extreme_move_10_signal",
            "10-bar extreme move price extension (> +5% / < -5%)",
            "signalx_native",
            "10-bar return > +5.0%",
            "10-bar return < -5.0%",
        ),
        (
            "mom_rmi_ob_os_14_signal",
            "Relative Momentum Index (14, 5) overbought (>70) and oversold (<30) thresholds",
            "signalx_native",
            "RMI 14 crosses above 30 from oversold zone",
            "RMI 14 crosses below 70 from overbought zone",
        ),
        (
            "mom_dmi_variable_lookback_signal",
            "Dynamic Momentum Index (variable lookback 5-30) overbought (>70) and oversold (<30) thresholds",
            "signalx_native",
            "DMI crosses above 30 from oversold zone",
            "DMI crosses below 70 from overbought zone",
        ),
        (
            "mom_coppock_curve_zero_cross_signal",
            "Coppock Curve zero centerline crossover",
            "pandas_ta",
            "Coppock Curve crosses above 0",
            "Coppock Curve crosses below 0",
        ),
        (
            "mom_stoch_momentum_index_cross_signal",
            "Stochastic Momentum Index (SMI 13, 25, 2) signal line crossover in extreme zones",
            "pandas_ta",
            "SMI crosses above signal line when SMI < -40 (oversold)",
            "SMI crosses below signal line when SMI > +40 (overbought)",
        ),
        (
            "mom_schaff_trend_cycle_cross_signal",
            "Schaff Trend Cycle (STC 23, 50, 10) 25/75 cycle threshold crossover",
            "pandas_ta",
            "STC crosses above 25 (bullish cycle initiation)",
            "STC crosses below 75 (bearish cycle initiation)",
        ),
        (
            "mom_cmo_divergence_signal",
            "Chande Momentum Oscillator (CMO 14) 5-bar regular divergence",
            "signalx_native",
            "Low < Low[5] while CMO > CMO[5] (Bullish CMO divergence)",
            "High > High[5] while CMO < CMO[5] (Bearish CMO divergence)",
        ),
        (
            "mom_kst_oscillator_cross_signal",
            "Know Sure Thing (KST) oscillator and 9-SMA signal line crossover",
            "ta",
            "KST line crosses above 9-SMA signal line",
            "KST line crosses below 9-SMA signal line",
        ),
        (
            "mom_demarker_indicator_cross_signal",
            "Tom DeMarker Indicator (DeM 14) 0.30/0.70 threshold crossover",
            "signalx_native",
            "DeMarker 14 crosses above 0.30 from oversold zone",
            "DeMarker 14 crosses below 0.70 from overbought zone",
        ),
        (
            "mom_afternoon_open_breakout_signal",
            "VN30F1M 5m afternoon session open (13:00-13:30) momentum breakout above/below morning range",
            "signalx_native",
            "Close in afternoon open breaks above morning session high (bullish afternoon open breakout)",
            "Close in afternoon open breaks below morning session low (bearish afternoon open breakdown)",
        ),
    ]
    for idx, (name, desc, lib, buy_t, sell_t) in enumerate(mom_definitions, start=1):
        register_signal(
            code=f"MOM{idx:03d}_signal",
            name=name,
            category="momentum",
            description=desc,
            library=lib,
            buy_trigger=buy_t,
            sell_trigger=sell_t,
        )

    # -------------------------------------------------------------------------
    # 3. VOLATILITY SIGNALS (44 signals)
    # -------------------------------------------------------------------------
    vol_definitions = [
        (
            "vol_bb_breakout_20_20_signal",
            "Bollinger Bands (20, 2.0 std) outer band breakout",
            "ta",
            "Close breaks above upper Bollinger Band",
            "Close breaks below lower Bollinger Band",
        ),
        (
            "vol_bb_bounce_20_20_signal",
            "Bollinger Bands (20, 2.0 std) mean-reverting bounce",
            "ta",
            "Price dips below lower band then bounces back inside",
            "Price rises above upper band then rejects back inside",
        ),
        (
            "vol_bb_breakout_50_25_signal",
            "Bollinger Bands (50, 2.5 std) wide channel breakout",
            "ta",
            "Close breaks above upper Bollinger Band 50/2.5",
            "Close breaks below lower Bollinger Band 50/2.5",
        ),
        (
            "vol_bb_bounce_50_25_signal",
            "Bollinger Bands (50, 2.5 std) wide channel mean-reversion bounce",
            "ta",
            "Price bounces back above lower band 50/2.5",
            "Price bounces back below upper band 50/2.5",
        ),
        (
            "vol_bb_pct_b_reversal_20_signal",
            "Bollinger %B (20 period) extreme boundary reversal",
            "ta",
            "%B crosses back above 0.0",
            "%B crosses back below 1.0",
        ),
        (
            "vol_bb_pct_b_reversal_50_signal",
            "Bollinger %B (50 period) extreme boundary reversal",
            "ta",
            "%B crosses back above 0.0",
            "%B crosses back below 1.0",
        ),
        (
            "vol_donchian_breakout_10_signal",
            "Fast Donchian Channel (10 period) high/low channel breakout",
            "ta",
            "Close >= 10-period highest high",
            "Close <= 10-period lowest low",
        ),
        (
            "vol_donchian_breakout_20_signal",
            "Standard Donchian Channel (20 period) Turtle breakout System 1",
            "ta",
            "Close >= 20-period highest high",
            "Close <= 20-period lowest low",
        ),
        (
            "vol_donchian_breakout_55_signal",
            "Slow Donchian Channel (55 period) Turtle breakout System 2",
            "ta",
            "Close >= 55-period highest high",
            "Close <= 55-period lowest low",
        ),
        (
            "vol_keltner_breakout_20_15_signal",
            "Keltner Channel (20, 1.5 ATR) channel breakout",
            "ta",
            "Close breaks above upper Keltner Channel",
            "Close breaks below lower Keltner Channel",
        ),
        (
            "vol_keltner_breakout_20_20_signal",
            "Keltner Channel (20, 2.0 ATR) channel breakout",
            "ta",
            "Close breaks above upper Keltner Channel",
            "Close breaks below lower Keltner Channel",
        ),
        (
            "vol_ttm_squeeze_signal",
            "TTM Squeeze breakout: Bollinger Bands contract inside Keltner Channels then expand",
            "ta",
            "Squeeze fires and price is above 20-SMA baseline",
            "Squeeze fires and price is below 20-SMA baseline",
        ),
        (
            "vol_bb_bandwidth_expansion_signal",
            "Bollinger Bandwidth surge indicating volatility expansion",
            "ta",
            "Bandwidth expands above rolling quantile with upward price",
            "Bandwidth expands above rolling quantile with downward price",
        ),
        (
            "vol_atr_trailing_stop_2x_signal",
            "ATR Trailing Stop (14 period, 2.0x multiplier) price direction",
            "ta",
            "Close > ATR(14) 2.0x trailing stop",
            "Close < ATR(14) 2.0x trailing stop",
        ),
        (
            "vol_atr_trailing_stop_3x_signal",
            "ATR Trailing Stop (14 period, 3.0x multiplier) price direction",
            "ta",
            "Close > ATR(14) 3.0x trailing stop",
            "Close < ATR(14) 3.0x trailing stop",
        ),
        (
            "vol_chaikin_volatility_surge_signal",
            "Chaikin Volatility indicator surge",
            "pandas_ta",
            "Chaikin Volatility surges positive",
            "Chaikin Volatility contracts or turns negative",
        ),
        (
            "vol_hv_ratio_breakout_10_30_signal",
            "Historical Volatility ratio (10/30) breakout",
            "signalx_native",
            "HV(10)/HV(30) > 1.5 with positive price return",
            "HV(10)/HV(30) > 1.5 with negative price return",
        ),
        (
            "vol_bb_rejection_20_signal",
            "Bollinger Bands (20, 2.0 std) outer band rejection with wick and RSI confluence",
            "signalx_native",
            "Low < LB, Close > LB, RSI < 35, and Lower Wick > Body (Bullish rejection)",
            "High > UB, Close < UB, RSI > 65, and Upper Wick > Body (Bearish rejection)",
        ),
        (
            "vol_bb_bandwidth_regime_signal",
            "Bollinger Bandwidth Squeeze vs Expansion relative to 20-period SMA",
            "signalx_native",
            "BB Width < 0.7 * SMA20(BB Width) (Volatility squeeze / contraction)",
            "BB Width > 1.3 * SMA20(BB Width) (Volatility expansion surge)",
        ),
        (
            "vol_compression_breakout_5_20_signal",
            "Volatility compression STD(5) < 0.5*STD(20) combined with 10-bar price breakout",
            "signalx_native",
            "STD5 < 0.5*STD20 and Close > 10-bar High (Squeeze upside breakout)",
            "STD5 < 0.5*STD20 and Close < 10-bar Low (Squeeze downside breakdown)",
        ),
        (
            "vol_atr_expansion_14_20_signal",
            "ATR(14) expansion > 1.5x SMA20(ATR) with directional price move",
            "signalx_native",
            "ATR14 > 1.5*SMA20(ATR) and Close > Close[1]",
            "ATR14 > 1.5*SMA20(ATR) and Close < Close[1]",
        ),
        (
            "vol_range_breakout_10_signal",
            "10-period High/Low Range Breakout",
            "signalx_native",
            "Close > Highest(High, 10)[1]",
            "Close < Lowest(Low, 10)[1]",
        ),
        (
            "vol_volatility_break_10_50_signal",
            "Short-term vs long-term volatility regime shift (STD10 vs STD50)",
            "signalx_native",
            "STD10 > STD50 (Short-term volatility expansion)",
            "STD10 < STD50 (Short-term volatility compression)",
        ),
        (
            "vol_range_compression_20_signal",
            "Bar range compression (Height < 0.5x SMA20 Range) with SMA20 position",
            "signalx_native",
            "Bar Range < 0.5*SMA20(Range) and Close >= SMA20",
            "Bar Range < 0.5*SMA20(Range) and Close < SMA20",
        ),
        (
            "vol_volatility_drop_5_20_signal",
            "Volatility Drop (STD5 < 0.5x STD20) relative to SMA20 baseline",
            "signalx_native",
            "STD5 < 0.5*STD20 and Close >= SMA20",
            "STD5 < 0.5*STD20 and Close < SMA20",
        ),
        (
            "vol_range_expansion_20_signal",
            "Bar range expansion (Height > 1.5x SMA20 Range) with directional candle",
            "signalx_native",
            "Bar Range > 1.5*SMA20(Range) and Close > Open",
            "Bar Range > 1.5*SMA20(Range) and Close < Open",
        ),
        (
            "vol_range_shift_5_10_signal",
            "5-bar vs 10-bar range extreme shift",
            "signalx_native",
            "Lowest(Low, 5) > Lowest(Low, 10) (Rising base / higher low)",
            "Highest(High, 5) < Highest(High, 10) (Falling ceiling / lower high)",
        ),
        (
            "vol_range_position_10_signal",
            "Range Position within 10-period high-low channel",
            "signalx_native",
            "Close in lower 20% of 10-period range (Deep value)",
            "Close in upper 20% of 10-period range (Overextended)",
        ),
        (
            "vol_range_flip_5_signal",
            "5-period range flip breakout",
            "signalx_native",
            "Close > Highest(High, 5)[1]",
            "Close < Lowest(Low, 5)[1]",
        ),
        (
            "vol_linreg_channel_reversal_20_signal",
            "20-period Linear Regression Channel lower/upper band reversal",
            "signalx_native",
            "Close dips below lower 2-std linreg band and bounces back",
            "Close exceeds upper 2-std linreg band and reverses back",
        ),
        (
            "vol_keltner_reversal_20_signal",
            "Keltner Channel (20, 2.0 ATR) outer band reversal",
            "signalx_native",
            "Close dips below lower Keltner Channel and bounces back",
            "Close exceeds upper Keltner Channel and reverses back",
        ),
        (
            "vol_envelope_breakout_20_signal",
            "Moving Average Envelope (20, +/-2.5%) breakout",
            "signalx_native",
            "Close > SMA20 * 1.025 (Upper envelope breakout)",
            "Close < SMA20 * 0.975 (Lower envelope breakdown)",
        ),
        (
            "vol_rvi_ob_os_14_signal",
            "Relative Volatility Index (RVI 14 period) overbought/oversold crossover",
            "signalx_native",
            "RVI 14 crosses above 30 from oversold",
            "RVI 14 crosses below 70 from overbought",
        ),
        (
            "vol_garman_klass_expansion_signal",
            "Garman-Klass Volatility Estimator rolling expansion surge",
            "signalx_native",
            "GK Volatility > 90th percentile and Close > Close[1]",
            "GK Volatility > 90th percentile and Close < Close[1]",
        ),
        (
            "vol_parkinson_volatility_surge_signal",
            "Parkinson High-Low Volatility surge >= 1.8x SMA20",
            "signalx_native",
            "Parkinson Volatility >= 1.8 * SMA20(Park) and Close > Open",
            "Parkinson Volatility >= 1.8 * SMA20(Park) and Close < Open",
        ),
        (
            "vol_squeeze_momentum_pro_signal",
            "LazyBear Squeeze Pro BB inside KC breakout with LinReg momentum",
            "signalx_native",
            "Squeeze releases (BB width > KC width) with positive momentum slope",
            "Squeeze releases (BB width > KC width) with negative momentum slope",
        ),
        (
            "vol_keltner_width_squeeze_signal",
            "Keltner Channel Bandwidth compression (< 0.70x SMA20 Width)",
            "signalx_native",
            "KC Width < 0.70 * SMA20(KC Width) and Close >= SMA20",
            "KC Width < 0.70 * SMA20(KC Width) and Close < SMA20",
        ),
        (
            "vol_atr_ratio_fast_slow_signal",
            "ATR Fast/Slow Ratio (ATR(5) / ATR(20) > 1.40) volatility explosion",
            "signalx_native",
            "ATR(5) / ATR(20) > 1.40 and Close > Close[1]",
            "ATR(5) / ATR(20) > 1.40 and Close < Close[1]",
        ),
        (
            "vol_chandelier_exit_reversal_signal",
            "Chandelier Exit (22 period, 3.0 ATR) trailing stop direction reversal",
            "signalx_native",
            "Close flips above Short Stop (Lowest Low 22 + 3*ATR22)",
            "Close drops below Long Stop (Highest High 22 - 3*ATR22)",
        ),
        (
            "vol_mass_index_reversal_bulge_signal",
            "Mass Index (25 period) reversal bulge (>27.0 then <26.5)",
            "signalx_native",
            "Mass Index drops below 26.5 after bulge > 27.0 with EMA9(Close) rising",
            "Mass Index drops below 26.5 after bulge > 27.0 with EMA9(Close) falling",
        ),
        (
            "vol_normalized_atr_stretch_signal",
            "Normalized ATR (NATR = ATR(14)/Close * 100) extreme expansion >= 2.0x SMA20",
            "signalx_native",
            "NATR >= 2.0 * SMA20(NATR) and Close > Open",
            "NATR >= 2.0 * SMA20(NATR) and Close < Open",
        ),
        (
            "vol_dual_thrust_range_breakout_signal",
            "Dual Thrust 5-period range breakout",
            "signalx_native",
            "Close > Open + 0.5 * Range(5)",
            "Close < Open - 0.5 * Range(5)",
        ),
        (
            "vol_ib_breakout_30m_signal",
            "VN30F1M 5m Initial Balance (first 6 bars, 08:45-09:15) breakout or trap reversal",
            "signalx_native",
            "Close breaks above IB High and sustains (momentum breakout)",
            "Close breaks below IB Low and sustains, or false breakout reversal (trap)",
        ),
        (
            "vol_pre_atc_squeeze_signal",
            "VN30F1M 5m Pre-ATC session (14:00-14:25) volatility squeeze and breakout",
            "signalx_native",
            "Bollinger Band width below 20th percentile then expands with bullish candle (pre-ATC squeeze breakout long)",
            "Bollinger Band width below 20th percentile then expands with bearish candle (pre-ATC squeeze breakout short)",
        ),
    ]
    for idx, (name, desc, lib, buy_t, sell_t) in enumerate(vol_definitions, start=1):
        register_signal(
            code=f"VOL{idx:03d}_signal",
            name=name,
            category="volatility",
            description=desc,
            library=lib,
            buy_trigger=buy_t,
            sell_trigger=sell_t,
        )

    # -------------------------------------------------------------------------
    # 4. VOLUME SIGNALS (32 signals)
    # -------------------------------------------------------------------------
    vol_flow_definitions = [
        (
            "volume_obv_ema_cross_20_signal",
            "On-Balance Volume (OBV) crosses its 20-period EMA",
            "ta",
            "OBV crosses above EMA(OBV, 20)",
            "OBV crosses below EMA(OBV, 20)",
        ),
        (
            "volume_cmf_zero_cross_20_signal",
            "Chaikin Money Flow (20 period) zero centerline crossover",
            "ta",
            "CMF 20 crosses above 0",
            "CMF 20 crosses below 0",
        ),
        (
            "volume_cmf_threshold_cross_20_signal",
            "Chaikin Money Flow (20 period) institutional flow +/-0.05 thresholds",
            "ta",
            "CMF 20 > +0.05",
            "CMF 20 < -0.05",
        ),
        (
            "volume_vwap_cross_20_signal",
            "Rolling 20-period VWAP price crossover",
            "ta",
            "Close crosses above 20-period rolling VWAP",
            "Close crosses below 20-period rolling VWAP",
        ),
        (
            "volume_vwap_cross_50_signal",
            "Rolling 50-period VWAP price crossover",
            "ta",
            "Close crosses above 50-period rolling VWAP",
            "Close crosses below 50-period rolling VWAP",
        ),
        (
            "volume_vwap_cross_100_signal",
            "Rolling 100-period VWAP price crossover",
            "ta",
            "Close crosses above 100-period rolling VWAP",
            "Close crosses below 100-period rolling VWAP",
        ),
        (
            "volume_vwap_band_reversal_20_signal",
            "Rolling 20-period VWAP standard deviation band reversal",
            "ta",
            "Price bounces above lower VWAP -2.0 std band",
            "Price rejects below upper VWAP +2.0 std band",
        ),
        (
            "volume_spike_direction_20_signal",
            "Volume Spike (>2.0x 20-SMA) combined with directional candle body",
            "ta",
            "Volume > 2.0x 20-SMA with bullish candle (Close > Open)",
            "Volume > 2.0x 20-SMA with bearish candle (Close < Open)",
        ),
        (
            "volume_pvt_ma_cross_14_signal",
            "Price Volume Trend (PVT) crosses 14-period SMA",
            "ta",
            "PVT crosses above SMA(PVT, 14)",
            "PVT crosses below SMA(PVT, 14)",
        ),
        (
            "volume_adl_ma_cross_signal",
            "Accumulation / Distribution Line (ADL) crosses 20-period SMA",
            "ta",
            "ADL crosses above SMA(ADL, 20)",
            "ADL crosses below SMA(ADL, 20)",
        ),
        (
            "volume_force_index_13_signal",
            "Elder's Force Index (13 period) zero line crossover",
            "ta",
            "Force Index 13 crosses above 0",
            "Force Index 13 crosses below 0",
        ),
        (
            "volume_eom_zero_14_signal",
            "Ease of Movement (14 period) zero line crossover",
            "ta",
            "Ease of Movement 14 crosses above 0",
            "Ease of Movement 14 crosses below 0",
        ),
        (
            "volume_vsa_confirmation_20_signal",
            "Volume Spread Analysis (VSA) volume surge confirmation (> 1.2x SMA20 Volume)",
            "signalx_native",
            "Close > Open and Volume > 1.2 * SMA20(Volume) (Bullish volume confirmation)",
            "Close < Open and Volume > 1.2 * SMA20(Volume) (Bearish volume confirmation)",
        ),
        (
            "volume_price_confirmation_20_signal",
            "Volume expansion confirming price direction (Volume > SMA20)",
            "signalx_native",
            "Close > Close[1] and Volume > SMA20(Volume)",
            "Close < Close[1] and Volume > SMA20(Volume)",
        ),
        (
            "volume_vpt_divergence_5_signal",
            "5-bar regular Volume Price Trend (VPT) divergence",
            "signalx_native",
            "Low < Low[5] while VPT > VPT[5] (Bullish VPT divergence)",
            "High > High[5] while VPT < VPT[5] (Bearish VPT divergence)",
        ),
        (
            "volume_trend_3_bar_signal",
            "3-bar volume trend (expansion vs contraction)",
            "signalx_native",
            "3 consecutive bars of rising volume (Volume > Volume[1] > Volume[2])",
            "3 consecutive bars of falling volume (Volume < Volume[1] < Volume[2])",
        ),
        (
            "volume_price_divergence_signal",
            "Volume-Price divergence (absorption vs churn)",
            "signalx_native",
            "Close < Close[1] while Volume > Volume[1] (Bullish absorption)",
            "Close > Close[1] while Volume > Volume[1] (Bearish churn/effort vs result)",
        ),
        (
            "volume_amv_cross_20_signal",
            "Adaptive Moving Volume (AMV) 20-period price crossover",
            "signalx_native",
            "Close crosses above 20-period AMV baseline",
            "Close crosses below 20-period AMV baseline",
        ),
        (
            "volume_klinger_osc_cross_signal",
            "Klinger Volume Oscillator (34, 55, 13) signal cross",
            "signalx_native",
            "KVO crosses above KVO signal (13)",
            "KVO crosses below KVO signal (13)",
        ),
        (
            "volume_elder_ray_bull_bear_signal",
            "Elder Ray Index Bull & Bear Power indicator",
            "signalx_native",
            "Bear Power < 0 and rising with Bull Power > 0",
            "Bull Power > 0 and falling with Bear Power < 0",
        ),
        (
            "volume_climax_absorption_signal",
            "Volume Climax (>3.0x SMA20) with long wick absorption",
            "signalx_native",
            "Volume >= 3.0x SMA20 with lower wick >= 40% range and Close > Open",
            "Volume >= 3.0x SMA20 with upper wick >= 40% range and Close < Open",
        ),
        (
            "volume_twiggs_money_flow_cross_signal",
            "Twiggs Money Flow (21 period) zero centerline crossover",
            "signalx_native",
            "Twiggs Money Flow 21 crosses above 0",
            "Twiggs Money Flow 21 crosses below 0",
        ),
        (
            "volume_nvi_pvi_cross_signal",
            "Negative Volume Index (NVI) 20-period EMA crossover",
            "signalx_native",
            "NVI crosses above 20-period EMA",
            "NVI crosses below 20-period EMA",
        ),
        (
            "volume_vwap_anchored_dev1_signal",
            "Rolling 20-period VWAP +-1.0 std standard deviation band reversal",
            "signalx_native",
            "Low <= VWAP - 1.0 std and Close > VWAP - 1.0 std (Lower band bounce)",
            "High >= VWAP + 1.0 std and Close < VWAP + 1.0 std (Upper band rejection)",
        ),
        (
            "volume_vwap_anchored_dev3_signal",
            "Rolling 20-period VWAP +-3.0 std extreme mean-reversion bands",
            "signalx_native",
            "Low <= VWAP - 3.0 std or Close <= VWAP - 3.0 std (Extreme oversold touch)",
            "High >= VWAP + 3.0 std or Close >= VWAP + 3.0 std (Extreme overbought touch)",
        ),
        (
            "volume_delta_proxy_surge_signal",
            "Intrabar Delta Volume Proxy aggressive directional surge",
            "signalx_native",
            "Delta Buy Volume > 70% total volume and Close > Open",
            "Delta Sell Volume > 70% total volume and Close < Open",
        ),
        (
            "volume_vwma_sma_divergence_signal",
            "VWMA(20) vs SMA(20) Divergence (volume-weighted accumulation/distribution)",
            "signalx_native",
            "VWMA(20) > SMA(20) * 1.005 (Accumulation premium)",
            "VWMA(20) < SMA(20) * 0.995 (Distribution discount)",
        ),
        (
            "volume_volume_weighted_rsi_14_signal",
            "Volume-Weighted RSI (14 period) overbought/oversold threshold crossover",
            "signalx_native",
            "Volume-Weighted RSI 14 crosses above 30 from oversold",
            "Volume-Weighted RSI 14 crosses below 70 from overbought",
        ),
        (
            "volume_session_vwap_cross_signal",
            "VN30F1M 5m cumulative session VWAP crossover signal",
            "signalx_native",
            "Close crosses above intraday session VWAP (bullish bias shift)",
            "Close crosses below intraday session VWAP (bearish bias shift)",
        ),
        (
            "volume_rvol_time_bucket_signal",
            "VN30F1M 5m Relative Volume normalized by time-of-day bucket (RVOL >= 2.0 threshold)",
            "signalx_native",
            "RVOL >= 2.0 and green candle (bullish high-volume surge in time bucket)",
            "RVOL >= 2.0 and red candle (bearish high-volume surge in time bucket)",
        ),
        (
            "volume_cvd_divergence_signal",
            "VN30F1M 5m Cumulative Volume Delta (CVD) intraday proxy divergence vs price extremes (10 bars)",
            "signalx_native",
            "CVD makes higher low while price makes lower low (bullish CVD divergence)",
            "CVD makes lower high while price makes higher high (bearish CVD divergence)",
        ),
        (
            "volume_stopping_climax_signal",
            "VN30F1M 5m stopping volume climax (Volume >= 2.5x SMA20, wick >= 40% range, absorption close)",
            "signalx_native",
            "Volume climax with long lower wick and close in upper half (bullish absorption)",
            "Volume climax with long upper wick and close in lower half (bearish absorption)",
        ),
    ]
    for idx, (name, desc, lib, buy_t, sell_t) in enumerate(vol_flow_definitions, start=1):
        register_signal(
            code=f"VLM{idx:03d}_signal",
            name=name,
            category="volume",
            description=desc,
            library=lib,
            buy_trigger=buy_t,
            sell_trigger=sell_t,
        )

    # -------------------------------------------------------------------------
    # 5. CANDLESTICK SIGNALS (28 signals)
    # -------------------------------------------------------------------------
    cdl_definitions = [
        (
            "cdl_engulfing_signal",
            "Bullish and Bearish Engulfing 2-bar pattern",
            "signalx_native",
            "Bullish Engulfing: green body completely engulfs prior red body",
            "Bearish Engulfing: red body completely engulfs prior green body",
        ),
        (
            "cdl_hammer_star_signal",
            "Hammer, Inverted Hammer, Shooting Star, and Hanging Man patterns",
            "signalx_native",
            "Hammer / Inverted Hammer with long lower shadow rejection",
            "Shooting Star / Hanging Man with long upper shadow rejection",
        ),
        (
            "cdl_pinbar_signal",
            "Pinbar / Price rejection bar (wick >= 2x body)",
            "signalx_native",
            "Bullish pinbar: lower wick >= 60% of candle range",
            "Bearish pinbar: upper wick >= 60% of candle range",
        ),
        (
            "cdl_marubozu_signal",
            "Marubozu strong momentum directional candle (body >= 85% range)",
            "signalx_native",
            "Bullish Marubozu (large green body, tiny wicks)",
            "Bearish Marubozu (large red body, tiny wicks)",
        ),
        (
            "cdl_harami_signal",
            "Harami inside body pattern (Bullish / Bearish Harami)",
            "signalx_native",
            "Bullish Harami: small green body inside prior large red body",
            "Bearish Harami: small red body inside prior large green body",
        ),
        (
            "cdl_inside_bar_breakout_signal",
            "Inside Bar breakout (current bar contained within prior bar range)",
            "signalx_native",
            "Inside bar closes bullish (Close > Open)",
            "Inside bar closes bearish (Close < Open)",
        ),
        (
            "cdl_outside_bar_signal",
            "Outside Bar breakout (High > prev High and Low < prev Low)",
            "signalx_native",
            "Outside bar closes bullish (Close > Open)",
            "Outside bar closes bearish (Close < Open)",
        ),
        (
            "cdl_doji_reversal_signal",
            "Doji reversal pattern (Dragonfly Doji vs Gravestone Doji)",
            "signalx_native",
            "Dragonfly Doji: tiny body with long lower shadow",
            "Gravestone Doji: tiny body with long upper shadow",
        ),
        (
            "cdl_three_soldiers_crows_signal",
            "Three White Soldiers (bullish) and Three Black Crows (bearish) pattern",
            "signalx_native",
            "Three White Soldiers: 3 consecutive expanding green candles",
            "Three Black Crows: 3 consecutive expanding red candles",
        ),
        (
            "cdl_consecutive_3_signal",
            "Three consecutive directional bars in same direction",
            "signalx_native",
            "3 consecutive bullish closes (Close > Open)",
            "3 consecutive bearish closes (Close < Open)",
        ),
        (
            "cdl_consecutive_5_signal",
            "Five consecutive directional bars in same direction",
            "signalx_native",
            "5 consecutive bullish closes (Close > Open)",
            "5 consecutive bearish closes (Close < Open)",
        ),
        (
            "cdl_morning_evening_star_signal",
            "Morning Star (bullish) and Evening Star (bearish) 3-bar reversal pattern",
            "signalx_native",
            "Morning Star: large red candle, small gapping body, strong green candle",
            "Evening Star: large green candle, small gapping body, strong red candle",
        ),
        (
            "cdl_piercing_darkcloud_signal",
            "Piercing Line and Dark Cloud Cover reversal pattern",
            "signalx_native",
            "Piercing Line: green candle closes > 50% into prior red body",
            "Dark Cloud Cover: red candle closes > 50% into prior green body",
        ),
        (
            "cdl_tweezer_tops_bottoms_signal",
            "Tweezer Top (bearish) and Tweezer Bottom (bullish) dual matching shadow reversal",
            "signalx_native",
            "Tweezer Bottom: matching lows with bullish second candle",
            "Tweezer Top: matching highs with bearish second candle",
        ),
        (
            "cdl_couple_cs_signal",
            "Couple Candlestick pattern (Green-Green breakout or Red-Red breakdown)",
            "signalx_native",
            "Green candle closes at high and breaks previous green candle high",
            "Red candle closes at low and breaks previous red candle low",
        ),
        (
            "cdl_fakey_pattern_signal",
            "Fakey Pattern (False 5-bar breakout with strong reversal close)",
            "signalx_native",
            "Low sweeps 5-bar low and reverses to close bullish (Close > Open)",
            "High sweeps 5-bar high and reverses to close bearish (Close < Open)",
        ),
        (
            "cdl_gap_up_down_signal",
            "Opening Price Gap Up / Down relative to prior bar range",
            "signalx_native",
            "Open > High[1] (Opening gap up)",
            "Open < Low[1] (Opening gap down)",
        ),
        (
            "cdl_body_size_expansion_signal",
            "Candle body expansion (> 2.0x 20-period SMA body)",
            "signalx_native",
            "Body > 2.0 * SMA20(Body) and Close > Open (Bullish body expansion)",
            "Body > 2.0 * SMA20(Body) and Close < Open (Bearish body expansion)",
        ),
        (
            "cdl_wick_rejection_signal",
            "Wick Rejection (> 2.0x body size)",
            "signalx_native",
            "Lower Wick > 2.0 * Body and Close > Open (Bullish wick rejection)",
            "Upper Wick > 2.0 * Body and Close < Open (Bearish wick rejection)",
        ),
        (
            "cdl_body_direction_signal",
            "2-bar consecutive directional body agreement",
            "signalx_native",
            "2 consecutive bullish candle bodies (Close > Open and Close[1] > Open[1])",
            "2 consecutive bearish candle bodies (Close < Open and Close[1] < Open[1])",
        ),
        (
            "cdl_close_strength_signal",
            "Close Strength relative to candle midpoint and open",
            "signalx_native",
            "Close > Midpoint and Close > Open (Strong bullish close)",
            "Close < Midpoint and Close < Open (Strong bearish close)",
        ),
        (
            "cdl_price_rejection_signal",
            "Price Rejection (Sweeps prior extreme but closes directional)",
            "signalx_native",
            "Low < Low[1] and Close > Open (Bullish price rejection)",
            "High > High[1] and Close < Open (Bearish price rejection)",
        ),
        (
            "cdl_break_retest_signal",
            "Break and Retest of 10-period High/Low extremes",
            "signalx_native",
            "Close[1] breaks 10-bar High and current bar pulls back and holds above the level",
            "Close[1] breaks 10-bar Low and current bar bounces and stays below the level",
        ),
        (
            "cdl_trend_exhaustion_signal",
            "Trend Exhaustion (Counter-trend reaction after 3-bar directional move)",
            "signalx_native",
            "Close > Close[1] after 2 prior lower closes (Bullish exhaustion bounce)",
            "Close < Close[1] after 2 prior higher closes (Bearish exhaustion pullback)",
        ),
        (
            "cdl_final_push_signal",
            "Final Push (Higher/lower close on diminishing volume)",
            "signalx_native",
            "Close > Close[1] > Close[2] on declining Volume (Bullish exhaustion push)",
            "Close < Close[1] < Close[2] on declining Volume (Bearish exhaustion push)",
        ),
        (
            "cdl_thrust_bar_signal",
            "Thrust Bar (Body >= 75% range and >= 1.8x SMA20 body)",
            "signalx_native",
            "Body >= 75% range, Body >= 1.8 * SMA20(Body), and Close > Open (Bullish thrust)",
            "Body >= 75% range, Body >= 1.8 * SMA20(Body), and Close < Open (Bearish thrust)",
        ),
        (
            "cdl_narrow_range_7_breakout_signal",
            "Narrow Range 7 (NR7) volatility compression breakout",
            "signalx_native",
            "Close breaks above High of NR7 bar (Bullish NR7 breakout)",
            "Close breaks below Low of NR7 bar (Bearish NR7 breakdown)",
        ),
        (
            "cdl_wide_range_reversal_signal",
            "Wide Range Reversal (Range >= 2.5x SMA20 range with extreme close)",
            "signalx_native",
            "Range >= 2.5 * SMA20(Range) and Close finishes in top 30% of bar",
            "Range >= 2.5 * SMA20(Range) and Close finishes in bottom 30% of bar",
        ),
    ]
    cdl_codes = (
        [f"CDL{i:03d}_signal" for i in range(1, 17)]
        + [f"CDL{i:03d}_signal" for i in range(19, 28)]
        + [f"CDL{i:03d}_signal" for i in range(35, 38)]
    )
    for code, (name, desc, lib, buy_t, sell_t) in zip(cdl_codes, cdl_definitions, strict=True):
        register_signal(
            code=code,
            name=name,
            category="candlestick",
            description=desc,
            library=lib,
            buy_trigger=buy_t,
            sell_trigger=sell_t,
        )

    # -------------------------------------------------------------------------
    # 6. SMART MONEY CONCEPTS (SMC) SIGNALS (11 signals)
    # -------------------------------------------------------------------------
    smc_definitions = [
        (
            "smc_fvg_bullish_mitigation_signal",
            "Bullish Fair Value Gap (FVG) mitigation & retest",
            "signalx_native",
            "Price retraces into bullish FVG zone (l <= fvg_top and c >= fvg_bottom with bullish close)",
            "Price breaks below bullish FVG bottom (invalidation)",
        ),
        (
            "smc_fvg_bearish_mitigation_signal",
            "Bearish Fair Value Gap (FVG) mitigation & retest",
            "signalx_native",
            "Price retraces into bearish FVG zone (h >= fvg_bottom and c <= fvg_top with bearish close)",
            "Price breaks above bearish FVG top (invalidation)",
        ),
        (
            "smc_order_block_retest_signal",
            "Bullish / Bearish Order Block (OB) formation and retest",
            "signalx_native",
            "Price retests bullish OB body and closes bullish",
            "Price retests bearish OB body and closes bearish",
        ),
        (
            "smc_break_of_structure_signal",
            "Break of Structure (BOS) trend continuation",
            "signalx_native",
            "Close breaks 10-bar high with SMA20 > SMA50 (bullish continuation)",
            "Close breaks 10-bar low with SMA20 < SMA50 (bearish continuation)",
        ),
        (
            "smc_change_of_character_signal",
            "Change of Character (CHoCH) structural trend reversal",
            "signalx_native",
            "Close breaks 5-bar high after sustained bearish regime (bullish reversal)",
            "Close breaks 5-bar low after sustained bullish regime (bearish reversal)",
        ),
        (
            "smc_market_structure_break_signal",
            "Market Structure Break (MSB) higher high / lower low breakout",
            "signalx_native",
            "Close breaks recent high after lower low (bullish market structure break)",
            "Close breaks recent low after higher high (bearish market structure break)",
        ),
        (
            "smc_liquidity_sweep_signal",
            "5-bar high/low liquidity sweep with close back inside",
            "signalx_native",
            "Low < 5-bar min but Close > 5-bar min (Bullish liquidity sweep)",
            "High > 5-bar max but Close < 5-bar max (Bearish liquidity sweep)",
        ),
        (
            "smc_equal_high_low_sweep_signal",
            "Equal Highs / Equal Lows (EQH/EQL) liquidity sweep",
            "signalx_native",
            "Equal low swept and Close > previous Low (Bullish equal low sweep)",
            "Equal high swept and Close < previous High (Bearish equal high sweep)",
        ),
        (
            "smc_judas_swing_signal",
            "ICT Judas Swing false opening breakout & reversal",
            "signalx_native",
            "Low sweeps 5-bar low then closes above midpoint and open (Bullish Judas Swing)",
            "High sweeps 5-bar high then closes below midpoint and open (Bearish Judas Swing)",
        ),
        (
            "smc_inducement_sweep_signal",
            "Inducement (IDM) minor liquidity sweep & wick rejection",
            "signalx_native",
            "Low sweeps previous low with lower wick >= 50% and close > open (Bullish Inducement)",
            "High sweeps previous high with upper wick >= 50% and close < open (Bearish Inducement)",
        ),
        (
            "smc_pdh_pdl_sweep_signal",
            "VN30F1M 5m Previous Day High/Low liquidity sweep and reversal (wick through PDH/PDL, close back inside)",
            "signalx_native",
            "Price wicks below PDL then closes above it (bullish PDL sweep reversal)",
            "Price wicks above PDH then closes below it (bearish PDH sweep reversal)",
        ),
    ]
    for idx, (name, desc, lib, buy_t, sell_t) in enumerate(smc_definitions, start=1):
        register_signal(
            code=f"SMC{idx:03d}_signal",
            name=name,
            category="smc",
            description=desc,
            library=lib,
            buy_trigger=buy_t,
            sell_trigger=sell_t,
        )

    # -------------------------------------------------------------------------
    # 6. STATISTICAL SIGNALS (11 signals)
    # -------------------------------------------------------------------------
    stat_definitions = [
        (
            "stat_price_zscore_10_signal",
            "10-period rolling price Z-score (+/-2.0 std bounds)",
            "scipy",
            "Price Z-Score 10 < -2.0 (statistically oversold)",
            "Price Z-Score 10 > +2.0 (statistically overbought)",
        ),
        (
            "stat_price_zscore_20_signal",
            "20-period rolling price Z-score (+/-2.0 std bounds)",
            "scipy",
            "Price Z-Score 20 < -2.0 (statistically oversold)",
            "Price Z-Score 20 > +2.0 (statistically overbought)",
        ),
        (
            "stat_price_zscore_50_signal",
            "50-period rolling price Z-score (+/-2.0 std bounds)",
            "scipy",
            "Price Z-Score 50 < -2.0 (statistically oversold)",
            "Price Z-Score 50 > +2.0 (statistically overbought)",
        ),
        (
            "stat_price_zscore_100_signal",
            "100-period rolling price Z-score (+/-2.0 std bounds)",
            "scipy",
            "Price Z-Score 100 < -2.0 (statistically oversold)",
            "Price Z-Score 100 > +2.0 (statistically overbought)",
        ),
        (
            "stat_return_zscore_20_signal",
            "20-period rolling return Z-score (+/-2.0 std bounds)",
            "scipy",
            "Return Z-Score 20 < -2.0 (extreme return drawdown)",
            "Return Z-Score 20 > +2.0 (extreme return surge)",
        ),
        (
            "stat_ker_trend_filter_10_signal",
            "Kaufman Efficiency Ratio (10 period) trend efficiency filter",
            "signalx_native",
            "KER 10 > 0.6 and net direction is positive",
            "KER 10 > 0.6 and net direction is negative",
        ),
        (
            "stat_ker_trend_filter_20_signal",
            "Kaufman Efficiency Ratio (20 period) trend efficiency filter",
            "signalx_native",
            "KER 20 > 0.6 and net direction is positive",
            "KER 20 > 0.6 and net direction is negative",
        ),
        (
            "stat_chop_regime_14_signal",
            "Choppiness Index (14 period) trending (<38.2) vs consolidating (>61.8) regime",
            "ta",
            "Choppiness Index < 38.2 (strong directional trend mode)",
            "Choppiness Index > 61.8 (consolidating / chop mode)",
        ),
        (
            "stat_rolling_quantile_extremes_20_signal",
            "20-period rolling quantile extremes (5th and 95th percentiles)",
            "pandas",
            "Price <= 5th percentile of 20-period window",
            "Price >= 95th percentile of 20-period window",
        ),
        (
            "stat_linreg_slope_14_signal",
            "14-period linear regression slope direction",
            "scipy",
            "Linear regression slope > 0 with statistically significant t-stat",
            "Linear regression slope < 0 with statistically significant t-stat",
        ),
        (
            "stat_linreg_price_cross_30_signal",
            "30-period linear regression trendline price crossover",
            "scipy",
            "Price crosses above 30-period linear regression line",
            "Price crosses below 30-period linear regression line",
        ),
        (
            "stat_ma_stretch_zscore_20_signal",
            "Moving Average Stretch (Z-score > 2.0 or < -2.0 relative to 20-period SMA)",
            "signalx_native",
            "Price Z-Score 20 < -2.0 (Price stretched deeply below MA - mean reversion buy)",
            "Price Z-Score 20 > +2.0 (Price stretched deeply above MA - mean reversion sell)",
        ),
        (
            "stat_hurst_proxy_signal",
            "Rolling Hurst Exponent proxy (<0.5 mean-reversion, >0.5 trending regime)",
            "signalx_native",
            "Hurst proxy < 0.5 (Mean-reverting market regime)",
            "Hurst proxy > 0.5 (Persistent trending market regime)",
        ),
        (
            "stat_range_mid_reversion_10_signal",
            "10-period High/Low Range Midpoint Reversion",
            "signalx_native",
            "Close < (Highest(High, 10) + Lowest(Low, 10)) / 2 (Below range midpoint)",
            "Close > (Highest(High, 10) + Lowest(Low, 10)) / 2 (Above range midpoint)",
        ),
        (
            "stat_price_acceleration_signal",
            "1-bar Price Change Acceleration (Delta Momentum)",
            "signalx_native",
            "(Close - Close[1]) > (Close[1] - Close[2]) (Price accelerating upward)",
            "(Close - Close[1]) < (Close[1] - Close[2]) (Price accelerating downward)",
        ),
        (
            "stat_mean_distance_5pct_signal",
            "Mean Distance Deviation (> 5% away from 20-period SMA)",
            "signalx_native",
            "Close < SMA20 * 0.95 (Deep discount > 5% below SMA20)",
            "Close > SMA20 * 1.05 (Stretched premium > 5% above SMA20)",
        ),
        (
            "stat_fractal_dimension_index_signal",
            "30-period Fractal Dimension Index (FDI < 1.45 trending regime with SMA20 filter)",
            "signalx_native",
            "FDI 30 < 1.45 and Close > SMA20 (Persistent trending bull regime)",
            "FDI 30 < 1.45 and Close < SMA20 (Persistent trending bear regime)",
        ),
        (
            "stat_rolling_half_life_reversion_signal",
            "Ornstein-Uhlenbeck 30-period Half-Life Mean Reversion (HL in [3, 15] with Z-score +/-1.8)",
            "signalx_native",
            "Half-Life in [3, 15] and Price Z-Score 20 < -1.8 (Mean-reverting oversold buy)",
            "Half-Life in [3, 15] and Price Z-Score 20 > +1.8 (Mean-reverting overbought sell)",
        ),
        (
            "stat_variance_ratio_test_signal",
            "Lo-MacKinlay Variance Ratio Test (q=5, 30-period, VR > 1.25 trending structure)",
            "signalx_native",
            "Variance Ratio > 1.25 and ROC5 > 0 (Trending bullish market structure)",
            "Variance Ratio > 1.25 and ROC5 < 0 (Trending bearish market structure)",
        ),
        (
            "stat_rolling_skewness_reversal_signal",
            "20-period Rolling Return Skewness Reversal (Skew < -1.5 panic absorption / > +1.5 euphoria exhaustion)",
            "signalx_native",
            "Return Skewness 20 < -1.50 and Close > Close[1] (Panic selling absorption reversal buy)",
            "Return Skewness 20 > +1.50 and Close < Close[1] (Euphoria exhaustion reversal sell)",
        ),
    ]
    for idx, (name, desc, lib, buy_t, sell_t) in enumerate(stat_definitions, start=1):
        register_signal(
            code=f"STA{idx:03d}_signal",
            name=name,
            category="statistical",
            description=desc,
            library=lib,
            buy_trigger=buy_t,
            sell_trigger=sell_t,
        )

    # -------------------------------------------------------------------------
    # 7. COMPOSITE SIGNALS (13 signals)
    # -------------------------------------------------------------------------
    comp_definitions = [
        (
            "comp_trend_consensus_signal",
            "Trend Family Consensus: majority vote across all trend signals",
            "signalx_native",
            ">50% of trend signals are BUY",
            ">50% of trend signals are SELL",
        ),
        (
            "comp_momentum_consensus_signal",
            "Momentum Family Consensus: majority vote across all momentum signals",
            "signalx_native",
            ">50% of momentum signals are BUY",
            ">50% of momentum signals are SELL",
        ),
        (
            "comp_master_ensemble_signal",
            "Master Ensemble: weighted consensus across all library signals",
            "signalx_native",
            ">=35% of all active signals vote BUY",
            ">=35% of all active signals vote SELL",
        ),
        (
            "comp_ma_consensus_signal",
            "Moving Average Consensus across multiple SMA/EMA periods",
            "signalx_native",
            "Majority of MA signals indicate bullish trend",
            "Majority of MA signals indicate bearish trend",
        ),
        (
            "comp_trend_momentum_align_signal",
            "Trend & Momentum Alignment confluence",
            "signalx_native",
            "Both Trend consensus and Momentum consensus are BUY",
            "Both Trend consensus and Momentum consensus are SELL",
        ),
        (
            "comp_breakout_volume_confirmed_signal",
            "Price breakout confirmed with volume expansion",
            "signalx_native",
            "Volatility breakout (Donchian/BB) confirmed with Volume Spike",
            "Volatility breakdown (Donchian/BB) confirmed with Volume Spike",
        ),
        (
            "comp_mean_reversion_confluence_signal",
            "Multi-oscillator mean reversion confluence (RSI + BB lower + Z-Score)",
            "signalx_native",
            "Confluence of Oversold RSI (<30), lower BB touch, and Price Z-score < -2.0",
            "Confluence of Overbought RSI (>70), upper BB touch, and Price Z-score > +2.0",
        ),
        (
            "comp_macd_hist_candle_reversal_signal",
            "MACD Histogram turning point combined with candlestick reversal confirmation",
            "signalx_native",
            "MACD Histogram trough reversal in negative zone with bullish engulfing close",
            "MACD Histogram peak reversal in positive zone with bearish engulfing close",
        ),
        (
            "comp_smc_trend_volume_confluence_signal",
            "Smart Money Alignment Confluence (FVG/OB + SMA20>SMA50 + Volume > SMA20(Volume))",
            "signalx_native",
            "Bullish SMC structure (FVG mitigation / OB retest) with SMA20 > SMA50 and Volume > SMA20(Volume)",
            "Bearish SMC structure (FVG mitigation / OB retest) with SMA20 < SMA50 and Volume > SMA20(Volume)",
        ),
        (
            "comp_triple_screen_trading_system_signal",
            "Alexander Elder Triple Screen Trading System",
            "signalx_native",
            "EMA50 rising (Screen 1) + RSI/Stoch oversold pullback (Screen 2) + Close > prev High (Screen 3)",
            "EMA50 falling (Screen 1) + RSI/Stoch overbought pullback (Screen 2) + Close < prev Low (Screen 3)",
        ),
        (
            "comp_squeeze_momentum_volume_surge_signal",
            "Squeeze Momentum Breakout with Volume Surge",
            "signalx_native",
            "Bullish Squeeze Pro breakout release with Volume >= 1.5x SMA20(Volume)",
            "Bearish Squeeze Pro breakdown release with Volume >= 1.5x SMA20(Volume)",
        ),
        (
            "comp_master_ensemble_v2_signal",
            "Master Ensemble v2: Advanced weighted consensus across all library signals (>=30% threshold)",
            "signalx_native",
            ">=30% of all active signals vote BUY and buy votes exceed sell votes",
            ">=30% of all active signals vote SELL and sell votes exceed buy votes",
        ),
        (
            "comp_vn30_intraday_confluence_signal",
            "VN30F1M 5m intraday confluence: session VWAP bearing + RVOL surge + session open momentum alignment",
            "signalx_native",
            "Price above session VWAP, RVOL surge bullish, and session open breakout aligned bullish",
            "Price below session VWAP, RVOL surge bearish, and session open breakout aligned bearish",
        ),
    ]
    for idx, (name, desc, lib, buy_t, sell_t) in enumerate(comp_definitions, start=1):
        register_signal(
            code=f"CMP{idx:03d}_signal",
            name=name,
            category="composite",
            description=desc,
            library=lib,
            buy_trigger=buy_t,
            sell_trigger=sell_t,
        )


# Initialize default catalog on import
_initialize_default_catalog()

__all__ = [
    "SIGNAL_CATALOG",
    "SIGNAL_CATEGORIES",
    "SIGNAL_CODE_CATALOG",
    "VALID_CATEGORIES",
    "SignalMetadata",
    "get_code_to_name_map",
    "get_name_to_code_map",
    "get_signal_by_code",
    "get_signal_by_name",
    "get_signal_metadata",
    "get_signals_by_category",
    "list_categories",
    "register_signal",
    "to_code_names",
    "to_semantic_names",
]
