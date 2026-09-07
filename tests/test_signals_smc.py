from __future__ import annotations

import numpy as np
import pandas as pd

from signalx.constants import ALL_SIGNAL_STATES
from signalx.signals.session_helper import extract_session_context
from signalx.signals.smc import (
    SMC_SIGNAL_COLUMNS,
    _calc_break_of_structure,
    _calc_change_of_character,
    _calc_equal_high_low_sweep,
    _calc_fvg_bearish_mitigation,
    _calc_fvg_bullish_mitigation,
    _calc_inducement_sweep,
    _calc_judas_swing,
    _calc_liquidity_sweep,
    _calc_market_structure_break,
    _calc_morning_midpoint_acceptance,
    _calc_order_block_retest,
    _calc_pdh_pdl_sweep,
    generate_smc_signals,
)


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


def test_smc_signals_all_12_columns_present():
    df = make_synthetic_ohlcv(100)
    res = generate_smc_signals(df)
    assert len(res.columns) == 12
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
    assert len(res.columns) == 12

    short_df = make_synthetic_ohlcv(3)
    res_short = generate_smc_signals(short_df)
    assert len(res_short) == 3
    assert len(res_short.columns) == 12


def test_smc_fvg_mitigation_direct():
    df = make_synthetic_ohlcv(60)
    sig_bull = _calc_fvg_bullish_mitigation(df["open"], df["high"], df["low"], df["close"])
    sig_bear = _calc_fvg_bearish_mitigation(df["open"], df["high"], df["low"], df["close"])
    assert set(sig_bull.unique()).issubset(ALL_SIGNAL_STATES)
    assert set(sig_bear.unique()).issubset(ALL_SIGNAL_STATES)


def test_smc_all_direct_calculators():
    df = make_synthetic_ohlcv(120)
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]

    funcs = [
        _calc_order_block_retest(o, h, lo, c),
        _calc_break_of_structure(h, lo, c),
        _calc_change_of_character(h, lo, c),
        _calc_market_structure_break(h, lo, c),
        _calc_liquidity_sweep(h, lo, c),
        _calc_equal_high_low_sweep(h, lo, c),
        _calc_judas_swing(o, h, lo, c),
        _calc_inducement_sweep(o, h, lo, c),
        _calc_pdh_pdl_sweep(df, o, h, lo, c),
    ]
    for s in funcs:
        assert isinstance(s, pd.Series)
        assert len(s) == len(df)
        assert set(s.unique()).issubset(ALL_SIGNAL_STATES)


def test_calc_morning_midpoint_acceptance_direct():
    df = make_synthetic_ohlcv(150)
    ctx = extract_session_context(df)
    sig = _calc_morning_midpoint_acceptance(df["high"], df["low"], df["close"], ctx)
    assert isinstance(sig, pd.Series)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_smc_signals_progress():
    df = make_synthetic_ohlcv(50)
    res = generate_smc_signals(df, show_progress=True)
    assert len(res.columns) == 12
    assert list(res.columns) == SMC_SIGNAL_COLUMNS
