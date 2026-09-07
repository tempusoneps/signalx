from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from signalx.constants import ALL_SIGNAL_STATES
from signalx.signals.mean_reversion import (
    MEAN_REVERSION_SIGNAL_COLUMNS,
    _calc_amihud_liquidity_exhaustion,
    _calc_bb_pct_b_hook_reversion,
    _calc_bb_w_bottom_m_top,
    _calc_connors_rsi2_regime,
    _calc_dual_ma_disparity_index,
    _calc_ehlers_roofing_filter_reversion,
    _calc_keltner_atr_stretch_reentry,
    _calc_kurtosis_fat_tail_exhaustion,
    _calc_lehmann_short_term_reversal,
    _calc_linreg_residual_zscore,
    _calc_lo_mackinlay_variance_ratio,
    _calc_multi_period_stretch_consensus,
    _calc_ou_process_spread_reversion,
    _calc_session_range_fade,
    _calc_vn30_afternoon_reversal_trap,
    _calc_vn30_intraday_exhaustion_fade,
    _calc_vn30_midday_lunch_range_fade,
    _calc_vn30_morning_gap_fade,
    _calc_vn30_opening_drive_reversal,
    _calc_vn30_pdh_pdl_false_break_fade,
    _calc_vn30_pre_atc_vwap_snapback,
    _calc_vn30_session_vwap_band_fade,
    _calc_volume_climax_absorption_reversion,
    _calc_vwap_distance_zscore,
    _calc_wr_cci_double_oversold,
    generate_mean_reversion_signals,
)
from signalx.signals.session_helper import extract_session_context


def make_synthetic_ohlcv(n: int = 150, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.015
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.random.uniform(0.002, 0.015, n))
    low = close * (1.0 - np.random.uniform(0.002, 0.015, n))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_mean_reversion_signals_all_25_columns_present():
    df = make_synthetic_ohlcv(100)
    res = generate_mean_reversion_signals(df)
    assert len(res.columns) == 25
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
    assert len(res.columns) == 25

    short_df = make_synthetic_ohlcv(3)
    res_short = generate_mean_reversion_signals(short_df)
    assert len(res_short) == 3
    assert len(res_short.columns) == 25
    for col in res_short.columns:
        assert set(res_short[col].unique()).issubset(ALL_SIGNAL_STATES)


def test_mean_reversion_signals_flat_data():
    df_flat = pd.DataFrame(
        {
            "open": [100.0] * 50,
            "high": [100.0] * 50,
            "low": [100.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000.0] * 50,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = generate_mean_reversion_signals(df_flat)
    assert len(res) == 50
    for col in res.columns:
        assert set(res[col].unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_connors_rsi2_regime():
    df = make_synthetic_ohlcv(250)
    sig = _calc_connors_rsi2_regime(df["close"])
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_ou_process_spread_reversion():
    df = make_synthetic_ohlcv(100)
    sig = _calc_ou_process_spread_reversion(df["close"], window=30)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vwap_distance_zscore():
    df = make_synthetic_ohlcv(100)
    sig = _calc_vwap_distance_zscore(df["high"], df["low"], df["close"], df["volume"], window=20)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_bb_pct_b_hook_reversion():
    df = make_synthetic_ohlcv(100)
    sig = _calc_bb_pct_b_hook_reversion(df["close"], length=20, std=2.0)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_keltner_atr_stretch_reentry():
    df = make_synthetic_ohlcv(100)
    sig = _calc_keltner_atr_stretch_reentry(df["high"], df["low"], df["close"], length=20, mult=3.0)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_kurtosis_fat_tail_exhaustion():
    df = make_synthetic_ohlcv(100)
    sig = _calc_kurtosis_fat_tail_exhaustion(df["close"], window=20)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_dual_ma_disparity_index():
    df = make_synthetic_ohlcv(100)
    sig = _calc_dual_ma_disparity_index(df["close"], length=20, threshold=3.5)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_linreg_residual_zscore():
    df = make_synthetic_ohlcv(100)
    sig = _calc_linreg_residual_zscore(df["close"], length=20)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_wr_cci_double_oversold():
    df = make_synthetic_ohlcv(100)
    sig = _calc_wr_cci_double_oversold(df["high"], df["low"], df["close"], wr_len=14, cci_len=20)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_session_range_fade():
    df = make_synthetic_ohlcv(100)
    sig = _calc_session_range_fade(df, df["open"], df["high"], df["low"], df["close"])
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_volume_climax_absorption_reversion():
    df = make_synthetic_ohlcv(100)
    sig = _calc_volume_climax_absorption_reversion(
        df["open"], df["high"], df["low"], df["close"], df["volume"], window=20
    )
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_multi_period_stretch_consensus():
    df = make_synthetic_ohlcv(100)
    sig = _calc_multi_period_stretch_consensus(df["close"])
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_show_progress_flag():
    df = make_synthetic_ohlcv(50)
    res = generate_mean_reversion_signals(df, show_progress=True)
    assert len(res) == 50
    assert len(res.columns) == 25


def test_calc_lehmann_short_term_reversal():
    df = make_synthetic_ohlcv(120)
    sig = _calc_lehmann_short_term_reversal(df["open"], df["close"])
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_lo_mackinlay_variance_ratio():
    df = make_synthetic_ohlcv(120)
    sig = _calc_lo_mackinlay_variance_ratio(df["close"])
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_ehlers_roofing_filter_reversion():
    df = make_synthetic_ohlcv(150)
    sig = _calc_ehlers_roofing_filter_reversion(df["close"])
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_amihud_liquidity_exhaustion():
    df = make_synthetic_ohlcv(120)
    sig = _calc_amihud_liquidity_exhaustion(
        df["open"], df["high"], df["low"], df["close"], df["volume"]
    )
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_bb_w_bottom_m_top():
    df = make_synthetic_ohlcv(120)
    sig = _calc_bb_w_bottom_m_top(df["open"], df["high"], df["low"], df["close"])
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vn30_morning_gap_fade():
    df = make_synthetic_ohlcv(150)
    ctx = extract_session_context(df)
    sig = _calc_vn30_morning_gap_fade(df["open"], df["high"], df["low"], df["close"], ctx)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vn30_opening_drive_reversal():
    df = make_synthetic_ohlcv(150)
    ctx = extract_session_context(df)
    sig = _calc_vn30_opening_drive_reversal(df["open"], df["high"], df["low"], df["close"], ctx)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vn30_afternoon_reversal_trap():
    df = make_synthetic_ohlcv(150)
    ctx = extract_session_context(df)
    sig = _calc_vn30_afternoon_reversal_trap(df["open"], df["high"], df["low"], df["close"], ctx)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vn30_pre_atc_vwap_snapback():
    df = make_synthetic_ohlcv(150)
    ctx = extract_session_context(df)
    sig = _calc_vn30_pre_atc_vwap_snapback(
        df["open"], df["high"], df["low"], df["close"], df["volume"], ctx
    )
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vn30_pdh_pdl_false_break_fade():
    df = make_synthetic_ohlcv(150)
    ctx = extract_session_context(df)
    sig = _calc_vn30_pdh_pdl_false_break_fade(df["open"], df["high"], df["low"], df["close"], ctx)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vn30_midday_lunch_range_fade():
    df = make_synthetic_ohlcv(150)
    ctx = extract_session_context(df)
    sig = _calc_vn30_midday_lunch_range_fade(
        df["open"], df["high"], df["low"], df["close"], df["volume"], ctx
    )
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vn30_intraday_exhaustion_fade():
    df = make_synthetic_ohlcv(150)
    sig = _calc_vn30_intraday_exhaustion_fade(df["open"], df["high"], df["low"], df["close"])
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_vn30_session_vwap_band_fade():
    df = make_synthetic_ohlcv(150)
    ctx = extract_session_context(df)
    sig = _calc_vn30_session_vwap_band_fade(
        df["open"], df["high"], df["low"], df["close"], df["volume"], ctx
    )
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
