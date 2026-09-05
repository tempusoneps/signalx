from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.momentum import (
    MOMENTUM_SIGNAL_COLUMNS,
    _bound_signal,
    _calc_afternoon_open_breakout,
    _calc_ao_saucer,
    _calc_cmo,
    _calc_cmo_divergence,
    _calc_coppock,
    _calc_demarker,
    _calc_dmi,
    _calc_fisher,
    _calc_kst,
    _calc_rmi,
    _calc_schaff_tc,
    _calc_smi,
    _calc_smi_cross,
    _calc_threshold_reversal,
    _crossover_signal,
    generate_momentum_signals,
)


def make_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataset for fast, isolated unit testing."""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.02
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


EXPECTED_MOMENTUM_SIGNALS = MOMENTUM_SIGNAL_COLUMNS


def test_momentum_signals_all_39_columns_present():
    """Verify generate_momentum_signals produces exactly the 39 expected momentum signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_momentum_signals(df)

    assert len(EXPECTED_MOMENTUM_SIGNALS) == 39
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 39
    assert list(res.index) == list(df.index)

    for col in EXPECTED_MOMENTUM_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"

    assert "mom_afternoon_open_breakout_signal" in res.columns


def test_momentum_signals_all_states_valid():
    """Verify that every value in every signal column is strictly within ALL_SIGNAL_STATES and has no NaNs."""
    df = make_synthetic_ohlcv(250)
    res = generate_momentum_signals(df)

    for col in res.columns:
        assert not res[col].isna().any(), f"Column {col} contains unexpected NaN values"
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), (
            f"Column {col} contains invalid states: {unique_vals - ALL_SIGNAL_STATES}"
        )


def test_momentum_signals_state_occurrences():
    """Verify that BUY, SELL, HOLD, and NONE states actually occur in the output across signals."""
    df = make_synthetic_ohlcv(300)
    res = generate_momentum_signals(df)

    all_values = set()
    for col in res.columns:
        all_values.update(res[col].unique())

    assert SignalState.BUY in all_values
    assert SignalState.SELL in all_values
    assert SignalState.HOLD in all_values
    assert SignalState.NONE in all_values


def test_momentum_signals_short_dataframe():
    """Verify graceful execution without exceptions when given a short dataframe (warmup edge case)."""
    df_short = make_synthetic_ohlcv(10)
    res = generate_momentum_signals(df_short)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert len(res.columns) == 39

    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_momentum_signals_empty_dataframe():
    """Verify handling of empty dataframe."""
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_momentum_signals(df_empty)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert len(res.columns) == 39
    for col in res.columns:
        assert col.endswith("_signal")


def test_momentum_signals_normalization():
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
    res = generate_momentum_signals(df_upper)

    assert len(res) == 50
    assert len(res.columns) == 39


def test_momentum_signals_missing_columns():
    """Verify that missing OHLCV columns raise ValueError."""
    df_invalid = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        generate_momentum_signals(df_invalid)


def test_momentum_signals_datetime_index_preserved():
    """Verify that custom DatetimeIndex is preserved across all output columns."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = make_synthetic_ohlcv(100)
    df.index = dates
    res = generate_momentum_signals(df)

    assert isinstance(res.index, pd.DatetimeIndex)
    assert res.index.equals(dates)


def test_momentum_signals_bullish_and_bearish_trends():
    """Verify momentum signals respond intuitively to clear unidirectional price movements."""
    n = 100
    # Clear bullish trend
    bull_close = np.linspace(100.0, 200.0, n)
    df_bull = pd.DataFrame(
        {
            "open": bull_close - 0.5,
            "high": bull_close + 1.0,
            "low": bull_close - 1.0,
            "close": bull_close,
            "volume": np.full(n, 1000.0),
        }
    )
    res_bull = generate_momentum_signals(df_bull)
    assert (res_bull["mom_rsi_50_cross_14_signal"].iloc[30:] == SignalState.HOLD).all()
    assert (res_bull["mom_roc_zero_cross_5_signal"].iloc[30:] == SignalState.HOLD).all()
    assert (res_bull["mom_roc_zero_cross_10_signal"].iloc[30:] == SignalState.HOLD).all()

    # Clear bearish trend
    bear_close = np.linspace(200.0, 100.0, n)
    df_bear = pd.DataFrame(
        {
            "open": bear_close + 0.5,
            "high": bear_close + 1.0,
            "low": bear_close - 1.0,
            "close": bear_close,
            "volume": np.full(n, 1000.0),
        }
    )
    res_bear = generate_momentum_signals(df_bear)
    assert (res_bear["mom_rsi_50_cross_14_signal"].iloc[30:] == SignalState.NONE).all()
    assert (res_bear["mom_roc_zero_cross_5_signal"].iloc[30:] == SignalState.NONE).all()


def test_crossover_signal_edge_cases():
    """Verify _crossover_signal helper on empty and all-nan inputs."""
    empty = pd.Series([], dtype=float)
    assert _crossover_signal(empty, empty).empty

    nan_s = pd.Series([np.nan, np.nan], dtype=float)
    res = _crossover_signal(nan_s, nan_s)
    assert (res == SignalState.NONE).all()


def test_bound_signal_direct():
    """Verify _bound_signal helper with buy_below=True and False."""
    vals = pd.Series([10.0, 50.0, 90.0, np.nan])
    res_buy_below = _bound_signal(vals, 20.0, 80.0, buy_below=True)
    assert list(res_buy_below) == [
        SignalState.BUY,
        SignalState.HOLD,
        SignalState.SELL,
        SignalState.NONE,
    ]

    res_breakout = _bound_signal(vals, 20.0, 80.0, buy_below=False)
    assert list(res_breakout) == [
        SignalState.SELL,
        SignalState.HOLD,
        SignalState.BUY,
        SignalState.NONE,
    ]


def test_fisher_and_cmo_edge_cases():
    """Verify _calc_fisher and _calc_cmo on small datasets and zero lengths."""
    high = pd.Series([10.0, 11.0, 12.0])
    low = pd.Series([9.0, 10.0, 11.0])
    close = pd.Series([9.5, 10.5, 11.5])

    f, s = _calc_fisher(high, low, length=9)
    assert f.isna().all()
    assert s.isna().all()

    cmo = _calc_cmo(close, length=14)
    assert isinstance(cmo, pd.Series)

    f0, s0 = _calc_fisher(high, low, length=0)
    assert f0.isna().all()


def test_ao_saucer_patterns():
    """Verify Awesome Oscillator Saucer bullish and bearish recognition."""
    # Bullish saucer: 3 bars > 0, bar1 > bar2, bar3 > bar2
    ao_bull = pd.Series([1.0, 2.0, 1.5, 1.8, 1.9])
    res_bull = _calc_ao_saucer(ao_bull)
    assert res_bull.iloc[3] == SignalState.BUY

    # Bearish saucer: 3 bars < 0, bar1 < bar2, bar3 < bar2
    ao_bear = pd.Series([-1.0, -2.0, -1.5, -1.8, -1.9])
    res_bear = _calc_ao_saucer(ao_bear)
    assert res_bear.iloc[3] == SignalState.SELL


def test_threshold_reversal_direct():
    """Verify _calc_threshold_reversal helper on boundary bounce conditions."""
    vals = pd.Series([25.0, 35.0, 50.0, 75.0, 65.0, np.nan])
    res = _calc_threshold_reversal(vals, lower=30.0, upper=70.0)
    assert res.iloc[1] == SignalState.BUY
    assert res.iloc[2] == SignalState.HOLD
    assert res.iloc[4] == SignalState.SELL
    assert res.iloc[5] == SignalState.NONE


def test_rmi_ob_os_direct():
    """Verify Relative Momentum Index (RMI) calculation and threshold reversal."""
    df = make_synthetic_ohlcv(100)
    rmi = _calc_rmi(df["close"], length=14, mom=5)
    assert isinstance(rmi, pd.Series)
    valid_rmi = rmi.dropna()
    assert (valid_rmi >= 0.0).all() and (valid_rmi <= 100.0).all()

    # Synthetic RMI bounce
    synthetic_rmi = pd.Series([20.0, 32.0, 55.0, 75.0, 68.0])
    sig = _calc_threshold_reversal(synthetic_rmi, lower=30.0, upper=70.0)
    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[4] == SignalState.SELL


def test_dmi_variable_lookback_direct():
    """Verify Dynamic Momentum Index (DMI) variable lookback calculation."""
    df = make_synthetic_ohlcv(150)
    dmi = _calc_dmi(df["close"], base_length=14, min_len=5, max_len=30)
    assert isinstance(dmi, pd.Series)
    valid_dmi = dmi.dropna()
    assert len(valid_dmi) > 50
    assert (valid_dmi >= 0.0).all() and (valid_dmi <= 100.0).all()

    # Check edge case short series
    short_dmi = _calc_dmi(df["close"].iloc[:10])
    assert short_dmi.isna().all()


def test_coppock_curve_direct():
    """Verify Coppock Curve calculation and zero centerline crossover."""
    df = make_synthetic_ohlcv(100)
    cop = _calc_coppock(df["close"])
    assert isinstance(cop, pd.Series)

    cop_syn = pd.Series([-5.0, 2.0, 3.0, -1.0])
    sig = _crossover_signal(cop_syn, pd.Series(0.0, index=cop_syn.index))
    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[3] == SignalState.SELL


def test_stoch_momentum_index_cross_direct():
    """Verify Stochastic Momentum Index (SMI) and extreme crossover signal."""
    df = make_synthetic_ohlcv(100)
    smi, smi_sig = _calc_smi(df["high"], df["low"], df["close"], fast=13, slow=25, signal=2)
    assert isinstance(smi, pd.Series)
    assert isinstance(smi_sig, pd.Series)

    # Bullish cross in oversold (< -40): prev smi <= sig, smi > sig and smi < -40
    s_smi = pd.Series([-50.0, -45.0, 50.0, 45.0])
    s_sig = pd.Series([-48.0, -48.0, 42.0, 48.0])
    res = _calc_smi_cross(s_smi, s_sig, lower=-40.0, upper=40.0)
    assert res.iloc[1] == SignalState.BUY
    assert res.iloc[3] == SignalState.SELL


def test_schaff_trend_cycle_cross_direct():
    """Verify Schaff Trend Cycle (STC) and 25/75 threshold crossings."""
    df = make_synthetic_ohlcv(150)
    stc = _calc_schaff_tc(df["close"], fast=23, slow=50, tc_length=10)
    assert isinstance(stc, pd.Series)
    valid_stc = stc.dropna()
    assert len(valid_stc) > 50

    stc_syn = pd.Series([15.0, 28.0, 50.0, 80.0, 72.0])
    sig = _calc_threshold_reversal(stc_syn, lower=25.0, upper=75.0)
    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[4] == SignalState.SELL


def test_cmo_divergence_direct():
    """Verify Chande Momentum Oscillator (CMO) 5-bar regular divergence."""
    # Bullish divergence: Low < Low[5], but CMO > CMO[5]
    lows = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 95.0])
    highs = pd.Series([110.0, 110.0, 110.0, 110.0, 110.0, 105.0])
    cmos = pd.Series([-40.0, -30.0, -20.0, -10.0, 0.0, -20.0])  # CMO[5]=-20 > CMO[0]=-40
    sig_bull = _calc_cmo_divergence(highs, lows, cmos, lookback=5)
    assert sig_bull.iloc[5] == SignalState.BUY

    # Bearish divergence: High > High[5], but CMO < CMO[5]
    lows_bear = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 105.0])
    highs_bear = pd.Series([110.0, 110.0, 110.0, 110.0, 110.0, 120.0])
    cmos_bear = pd.Series([40.0, 30.0, 20.0, 10.0, 0.0, 20.0])  # CMO[5]=20 < CMO[0]=40
    sig_bear = _calc_cmo_divergence(highs_bear, lows_bear, cmos_bear, lookback=5)
    assert sig_bear.iloc[5] == SignalState.SELL


def test_kst_oscillator_cross_direct():
    """Verify Know Sure Thing (KST) line and signal line crossover."""
    df = make_synthetic_ohlcv(100)
    kst, kst_sig = _calc_kst(df["close"])
    assert isinstance(kst, pd.Series)
    assert isinstance(kst_sig, pd.Series)

    kst_syn = pd.Series([10.0, 15.0, 12.0])
    sig_syn = pd.Series([12.0, 12.0, 14.0])
    sig = _crossover_signal(kst_syn, sig_syn)
    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[2] == SignalState.SELL


def test_demarker_indicator_cross_direct():
    """Verify Tom DeMarker Indicator (DeM 14) and 0.30/0.70 threshold crossing."""
    df = make_synthetic_ohlcv(100)
    dem = _calc_demarker(df["high"], df["low"], length=14)
    assert isinstance(dem, pd.Series)
    valid_dem = dem.dropna()
    assert (valid_dem >= 0.0).all() and (valid_dem <= 1.0).all()

    dem_syn = pd.Series([0.25, 0.35, 0.50, 0.75, 0.65])
    sig = _calc_threshold_reversal(dem_syn, lower=0.30, upper=0.70)
    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[4] == SignalState.SELL


def test_afternoon_open_breakout_direct():
    """Verify Afternoon Session Open (13:00 - 13:30) Directional Breakout."""
    # Edge cases: empty series / empty DataFrame
    empty_df = pd.DataFrame(columns=["high", "low", "close"])
    empty_s = pd.Series([], dtype=float)
    assert len(_calc_afternoon_open_breakout(empty_df, empty_s, empty_s, empty_s)) == 0
    assert len(_calc_afternoon_open_breakout(None, empty_s, empty_s, empty_s)) == 0

    # Test with timestamps
    dates = [
        # Morning session (09:00 - 11:30)
        "2026-09-01 09:00",
        "2026-09-01 10:00",
        "2026-09-01 11:00",
        "2026-09-01 11:25",
        # Afternoon open session (13:00 - 13:30)
        "2026-09-01 13:00",
        "2026-09-01 13:05",
        "2026-09-01 13:15",
        "2026-09-01 13:25",
        # Post afternoon open session (> 13:30)
        "2026-09-01 13:35",
        "2026-09-01 14:00",
    ]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [100.0, 101.0, 95.0, 93.0, 108.0, 95.0, 98.0, 105.0, 114.0, 85.0],
            "high": [105.0, 110.0, 104.0, 102.0, 113.0, 96.0, 102.0, 116.0, 122.0, 88.0],
            "low": [95.0, 92.0, 90.0, 93.0, 107.0, 87.0, 97.0, 104.0, 112.0, 78.0],
            "close": [100.0, 108.0, 91.0, 98.0, 112.0, 88.0, 100.0, 115.0, 120.0, 80.0],
            "volume": [1000.0] * 10,
        }
    )

    sig = _calc_afternoon_open_breakout(df, df["high"], df["low"], df["close"])

    # Morning bars (0..3) must be NONE
    for i in range(4):
        assert sig.iloc[i] == SignalState.NONE, f"Morning bar {i} must be NONE"

    # Afternoon open bars
    assert sig.iloc[4] == SignalState.BUY
    assert sig.iloc[5] == SignalState.SELL
    assert sig.iloc[6] == SignalState.NONE
    assert sig.iloc[7] == SignalState.BUY

    # Outside afternoon open bars
    assert sig.iloc[8] == SignalState.NONE
    assert sig.iloc[9] == SignalState.NONE

    # Synthetic fallback (no timestamps, 50 bars per session)
    synth_n = 50
    synth_df = pd.DataFrame(
        {
            "open": [100.0] * synth_n,
            "high": [105.0] * synth_n,
            "low": [95.0] * synth_n,
            "close": [100.0] * synth_n,
            "volume": [1000.0] * synth_n,
        }
    )
    synth_df.loc[10, "high"] = 110.0
    synth_df.loc[15, "low"] = 90.0

    synth_df.loc[30, "close"] = 112.0  # Buy
    synth_df.loc[31, "close"] = 88.0  # Sell
    synth_df.loc[32, "close"] = 100.0  # None
    synth_df.loc[36, "close"] = 115.0  # None (post afternoon open)

    sig_synth = _calc_afternoon_open_breakout(
        synth_df, synth_df["high"], synth_df["low"], synth_df["close"]
    )
    assert sig_synth.iloc[30] == SignalState.BUY
    assert sig_synth.iloc[31] == SignalState.SELL
    assert sig_synth.iloc[32] == SignalState.NONE
    assert sig_synth.iloc[36] == SignalState.NONE
