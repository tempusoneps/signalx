from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.volatility import (
    VOLATILITY_SIGNAL_COLUMNS,
    _calc_atr_ratio_fast_slow,
    _calc_atr_trailing_stop,
    _calc_bb_bounce,
    _calc_chaikin_volatility,
    _calc_chandelier_exit,
    _calc_dual_thrust,
    _calc_garman_klass,
    _calc_hv_ratio_breakout,
    _calc_keltner_width_squeeze,
    _calc_mass_index_reversal,
    _calc_natr_stretch,
    _calc_parkinson_surge,
    _calc_pct_b_reversal,
    _calc_rvi_ob_os,
    _calc_squeeze_momentum_pro,
    _calc_ttm_squeeze,
    generate_volatility_signals,
)


def make_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataset with volatility regimes for unit testing."""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.02
    # Add a volatility surge / breakout regime in the middle
    if n >= 100:
        returns[60:80] = np.abs(returns[60:80]) * 3.0  # upward surge
        returns[80:100] = -np.abs(returns[80:100]) * 3.0  # downward surge

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.random.uniform(0.005, 0.03, n))
    low = close * (1.0 - np.random.uniform(0.005, 0.03, n))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.randint(1000, 50000, n).astype(float)

    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


EXPECTED_VOLATILITY_SIGNALS = VOLATILITY_SIGNAL_COLUMNS


def test_volatility_signals_all_42_columns_present():
    """Verify generate_volatility_signals produces exactly 42 volatility signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volatility_signals(df)

    assert len(EXPECTED_VOLATILITY_SIGNALS) == 42
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 42
    assert list(res.index) == list(df.index)

    for col in EXPECTED_VOLATILITY_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"


def test_volatility_signals_all_states_valid():
    """Verify that every value in every signal column is strictly within ALL_SIGNAL_STATES and has no NaNs."""
    df = make_synthetic_ohlcv(250)
    res = generate_volatility_signals(df)

    for col in res.columns:
        assert not res[col].isna().any(), f"Column {col} contains unexpected NaN values"
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), (
            f"Column {col} contains invalid states: {unique_vals - ALL_SIGNAL_STATES}"
        )


def test_volatility_signals_state_occurrences():
    """Verify that BUY, SELL, HOLD, and NONE states actually occur in the output across signals."""
    df = make_synthetic_ohlcv(300)
    res = generate_volatility_signals(df)

    all_values = set()
    for col in res.columns:
        all_values.update(res[col].unique())

    assert SignalState.BUY in all_values
    assert SignalState.SELL in all_values
    assert SignalState.HOLD in all_values
    assert SignalState.NONE in all_values


def test_volatility_signals_short_dataframe():
    """Verify graceful execution without exceptions when given a short dataframe (warmup edge case)."""
    df_short = make_synthetic_ohlcv(10)
    res = generate_volatility_signals(df_short)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert len(res.columns) == 42

    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_volatility_signals_empty_dataframe():
    """Verify handling of empty dataframe."""
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_volatility_signals(df_empty)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert len(res.columns) == 42
    for col in res.columns:
        assert col.endswith("_signal")


def test_volatility_signals_normalization():
    """Verify that uppercase column names are properly normalized."""
    df = make_synthetic_ohlcv(50)
    df_upper = df.rename(
        columns={
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
            "volume": "VOLUME",
        }
    )
    res = generate_volatility_signals(df_upper)

    assert len(res) == 50
    assert len(res.columns) == 42


def test_volatility_signals_missing_columns():
    """Verify that missing OHLCV columns raise ValueError."""
    df_invalid = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        generate_volatility_signals(df_invalid)


def test_volatility_signals_datetime_index_preserved():
    """Verify that custom DatetimeIndex is preserved across all output columns."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = make_synthetic_ohlcv(100)
    df.index = dates
    res = generate_volatility_signals(df)

    assert isinstance(res.index, pd.DatetimeIndex)
    assert res.index.equals(dates)


def test_volatility_signals_bullish_and_bearish_breakouts():
    """Verify breakout signals trigger BUY during strong upsurges and SELL during strong downsurges."""
    n = 100
    # Clear explosive upward surge after flat consolidation
    bull_close = np.array([100.0] * 50 + list(np.linspace(101.0, 200.0, 50)))
    df_bull = pd.DataFrame(
        {
            "open": bull_close - 0.5,
            "high": bull_close + 1.0,
            "low": bull_close - 0.5,
            "close": bull_close,
            "volume": np.full(n, 1000.0),
        }
    )
    res_bull = generate_volatility_signals(df_bull)
    assert (res_bull["vol_bb_breakout_20_20_signal"].iloc[51:60] == SignalState.BUY).all()
    assert (res_bull["vol_donchian_breakout_10_signal"].iloc[51:60] == SignalState.BUY).all()
    assert (res_bull["vol_keltner_breakout_20_15_signal"].iloc[51:60] == SignalState.BUY).all()
    assert (res_bull["vol_atr_trailing_stop_2x_signal"].iloc[51:60] == SignalState.BUY).all()

    # Clear explosive downward plunge after flat consolidation
    bear_close = np.array([200.0] * 50 + list(np.linspace(199.0, 100.0, 50)))
    df_bear = pd.DataFrame(
        {
            "open": bear_close + 0.5,
            "high": bear_close + 0.5,
            "low": bear_close - 1.0,
            "close": bear_close,
            "volume": np.full(n, 1000.0),
        }
    )
    res_bear = generate_volatility_signals(df_bear)
    assert (res_bear["vol_bb_breakout_20_20_signal"].iloc[51:60] == SignalState.SELL).all()
    assert (res_bear["vol_donchian_breakout_10_signal"].iloc[51:60] == SignalState.SELL).all()
    assert (res_bear["vol_keltner_breakout_20_15_signal"].iloc[51:60] == SignalState.SELL).all()
    assert (res_bear["vol_atr_trailing_stop_2x_signal"].iloc[51:60] == SignalState.SELL).all()


def test_bb_bounce_signals_direct():
    """Verify _calc_bb_bounce helper on synthetic bounce and rejection scenarios."""
    close = pd.Series([100.0, 90.0, 95.0, 110.0, 105.0])
    lband = pd.Series([92.0, 92.0, 92.0, 92.0, 92.0])
    hband = pd.Series([108.0, 108.0, 108.0, 108.0, 108.0])

    # Bar 1: close 90 <= lband 92 (below band)
    # Bar 2: close 95 > lband 92 (bounced back inside -> BUY)
    # Bar 3: close 110 >= hband 108 (above band)
    # Bar 4: close 105 < hband 108 (rejected back inside -> SELL)
    res = _calc_bb_bounce(close, lband, hband)
    assert res.iloc[2] == SignalState.BUY
    assert res.iloc[4] == SignalState.SELL


def test_pct_b_reversal_signals_direct():
    """Verify _calc_pct_b_reversal helper on boundary crossings."""
    pct_b = pd.Series([-0.2, 0.1, 0.5, 1.2, 0.9])
    res = _calc_pct_b_reversal(pct_b)
    # Bar 1: crosses back above 0.0 -> BUY
    # Bar 4: crosses back below 1.0 -> SELL
    assert res.iloc[1] == SignalState.BUY
    assert res.iloc[4] == SignalState.SELL


def test_atr_trailing_stop_direct():
    """Verify _calc_atr_trailing_stop on trending and reversing series."""
    close = pd.Series([100, 102, 104, 106, 95, 93, 91, 105])
    high = close + 1.0
    low = close - 1.0

    res = _calc_atr_trailing_stop(close, high, low, window=3, multiplier=2.0)
    assert res.iloc[3] == SignalState.BUY
    assert res.iloc[5] == SignalState.SELL
    assert res.iloc[7] == SignalState.BUY


def test_ttm_squeeze_direct():
    """Verify _calc_ttm_squeeze helper."""
    # Squeeze ON: BB inside KC
    bb_h = pd.Series([105.0, 104.0, 108.0])
    bb_l = pd.Series([95.0, 96.0, 92.0])
    kc_h = pd.Series([106.0, 106.0, 106.0])
    kc_l = pd.Series([94.0, 94.0, 94.0])
    close = pd.Series([100.0, 101.0, 107.0])
    sma20 = pd.Series([100.0, 100.0, 100.0])

    res = _calc_ttm_squeeze(bb_h, bb_l, kc_h, kc_l, close, sma20)
    assert res.iloc[0] == SignalState.HOLD  # Squeeze is ON
    assert res.iloc[1] == SignalState.HOLD  # Squeeze is ON
    assert res.iloc[2] == SignalState.BUY  # Squeeze fired with close > sma20


def test_chaikin_volatility_direct():
    """Verify _calc_chaikin_volatility helper."""
    n = 30
    high = pd.Series(np.linspace(10, 50, n))
    low = pd.Series(np.full(n, 5.0))
    res = _calc_chaikin_volatility(high, low, length=5, roc_length=5)
    assert isinstance(res, pd.Series)
    assert res.iloc[15] == SignalState.BUY


def test_hv_ratio_breakout_direct():
    """Verify _calc_hv_ratio_breakout helper."""
    # Flat price then high volatility price jump
    close = pd.Series([100.0] * 30 + [102.0, 105.0, 110.0, 120.0, 130.0])
    res = _calc_hv_ratio_breakout(close, window_fast=5, window_slow=20, threshold=1.2)
    assert isinstance(res, pd.Series)
    assert (res.iloc[30:33] == SignalState.BUY).all()


def test_rvi_ob_os_direct():
    """Verify _calc_rvi_ob_os helper crosses above 30 (BUY) and below 70 (SELL)."""
    close = pd.Series(
        [
            100.0,
            95.0,
            90.0,
            85.0,
            80.0,
            75.0,
            70.0,
            65.0,
            60.0,
            55.0,
            50.0,
            48.0,
            47.0,
            46.0,
            50.0,
            55.0,
            60.0,
        ]
    )
    res = _calc_rvi_ob_os(close, length=5)
    assert isinstance(res, pd.Series)
    assert set(res.unique()).issubset(ALL_SIGNAL_STATES)


def test_garman_klass_expansion_direct():
    """Verify _calc_garman_klass helper on volatility surge."""
    open_p = pd.Series([100.0] * 25 + [100.0, 101.0, 102.0, 103.0, 104.0])
    high = pd.Series([101.0] * 25 + [115.0, 120.0, 125.0, 130.0, 135.0])
    low = pd.Series([99.0] * 25 + [90.0, 91.0, 92.0, 93.0, 94.0])
    close = pd.Series([100.0] * 25 + [110.0, 115.0, 120.0, 125.0, 130.0])
    res = _calc_garman_klass(open_p, high, low, close, window=20, quantile_threshold=0.90)
    assert isinstance(res, pd.Series)
    assert res.iloc[26] == SignalState.BUY


def test_parkinson_surge_direct():
    """Verify _calc_parkinson_surge helper on high-low surge."""
    open_p = pd.Series([100.0] * 25 + [100.0, 100.0])
    high = pd.Series([101.0] * 25 + [120.0, 120.0])
    low = pd.Series([99.0] * 25 + [90.0, 90.0])
    close = pd.Series([100.0] * 25 + [115.0, 95.0])
    res = _calc_parkinson_surge(open_p, high, low, close, window=20, multiplier=1.8)
    assert isinstance(res, pd.Series)
    assert res.iloc[25] == SignalState.BUY
    assert res.iloc[26] == SignalState.SELL


def test_squeeze_momentum_pro_direct():
    """Verify _calc_squeeze_momentum_pro helper."""
    # Squeeze ON then releases with positive and negative slope
    bb_w = pd.Series([1.0, 1.0, 3.0, 3.0])
    kc_w = pd.Series([2.0, 2.0, 2.0, 2.0])
    mom_slope = pd.Series([0.5, -0.5, 1.0, -1.0])
    res = _calc_squeeze_momentum_pro(bb_w, kc_w, mom_slope)
    assert isinstance(res, pd.Series)
    assert res.iloc[0] == SignalState.HOLD
    assert res.iloc[1] == SignalState.HOLD
    assert res.iloc[2] == SignalState.BUY
    assert res.iloc[3] == SignalState.SELL


def test_keltner_width_squeeze_direct():
    """Verify _calc_keltner_width_squeeze helper."""
    kc_w = pd.Series([10.0] * 20 + [5.0, 5.0])
    close = pd.Series([100.0] * 20 + [105.0, 95.0])
    sma20 = pd.Series([100.0] * 22)
    res = _calc_keltner_width_squeeze(kc_w, close, sma20, window=20, ratio=0.70)
    assert isinstance(res, pd.Series)
    assert res.iloc[20] == SignalState.BUY
    assert res.iloc[21] == SignalState.SELL


def test_atr_ratio_fast_slow_direct():
    """Verify _calc_atr_ratio_fast_slow helper."""
    high = pd.Series([101.0] * 25 + [110.0, 115.0, 90.0])
    low = pd.Series([99.0] * 25 + [95.0, 90.0, 70.0])
    close = pd.Series([100.0] * 25 + [108.0, 114.0, 75.0])
    res = _calc_atr_ratio_fast_slow(high, low, close, fast=5, slow=20, threshold=1.40)
    assert isinstance(res, pd.Series)
    assert set(res.unique()).issubset(ALL_SIGNAL_STATES)


def test_chandelier_exit_direct():
    """Verify _calc_chandelier_exit helper."""
    n = 35
    close = pd.Series(np.linspace(100, 150, n))
    high = close + 2.0
    low = close - 2.0
    res = _calc_chandelier_exit(high, low, close, length=22, multiplier=3.0)
    assert isinstance(res, pd.Series)
    assert set(res.unique()).issubset(ALL_SIGNAL_STATES)


def test_mass_index_reversal_direct():
    """Verify _calc_mass_index_reversal helper."""
    n = 45
    high = pd.Series(np.linspace(100, 150, n))
    low = high - 5.0
    close = high - 2.0
    res = _calc_mass_index_reversal(high, low, close)
    assert isinstance(res, pd.Series)
    assert set(res.unique()).issubset(ALL_SIGNAL_STATES)


def test_natr_stretch_direct():
    """Verify _calc_natr_stretch helper."""
    high = pd.Series([101.0] * 30 + [160.0, 160.0])
    low = pd.Series([99.0] * 30 + [40.0, 40.0])
    open_p = pd.Series([100.0] * 30 + [60.0, 140.0])
    close = pd.Series([100.0] * 30 + [150.0, 50.0])
    res = _calc_natr_stretch(open_p, high, low, close, window_atr=14, window_sma=20, multiplier=2.0)
    assert isinstance(res, pd.Series)
    assert res.iloc[30] == SignalState.BUY
    assert res.iloc[31] == SignalState.SELL


def test_dual_thrust_direct():
    """Verify _calc_dual_thrust helper."""
    n = 20
    open_p = pd.Series([100.0] * n)
    high = pd.Series([102.0] * n)
    low = pd.Series([98.0] * n)
    close = pd.Series([100.0] * (n - 2) + [105.0, 95.0])
    res = _calc_dual_thrust(open_p, high, low, close, length=5, k=0.5)
    assert isinstance(res, pd.Series)
    assert res.iloc[-2] == SignalState.BUY
    assert res.iloc[-1] == SignalState.SELL
