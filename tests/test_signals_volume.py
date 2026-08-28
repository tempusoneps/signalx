from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.volume import (
    VOLUME_SIGNAL_COLUMNS,
    _calc_volume_spike_direction,
    _calc_vwap_band_reversal,
    generate_volume_signals,
)


def make_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataset with volume and price regimes for unit testing."""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.02
    # Add trending regimes with corresponding volume surges
    if n >= 120:
        returns[40:70] = np.abs(returns[40:70]) * 2.5  # strong uptrend
        returns[80:110] = -np.abs(returns[80:110]) * 2.5  # strong downtrend

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.random.uniform(0.005, 0.03, n))
    low = close * (1.0 - np.random.uniform(0.005, 0.03, n))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n)

    base_volume = np.random.randint(1000, 20000, n).astype(float)
    if n >= 120:
        base_volume[40:70] *= 3.0  # volume expansion during uptrend
        base_volume[80:110] *= 3.0  # volume expansion during downtrend

    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": base_volume,
        }
    )


EXPECTED_VOLUME_SIGNALS = VOLUME_SIGNAL_COLUMNS


def test_volume_signals_all_12_columns_present():
    """Verify generate_volume_signals produces exactly the expected volume signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)

    assert len(EXPECTED_VOLUME_SIGNALS) == 18
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 18
    assert list(res.index) == list(df.index)

    for col in EXPECTED_VOLUME_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"


def test_volume_signals_all_states_valid():
    """Verify that every value in every signal column is strictly within ALL_SIGNAL_STATES and has no NaNs."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)

    for col in res.columns:
        assert not res[col].isna().any(), f"Column {col} contains unexpected NaN values"
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), (
            f"Column {col} contains invalid states: {unique_vals - ALL_SIGNAL_STATES}"
        )


def test_volume_signals_state_occurrences():
    """Verify that BUY, SELL, HOLD, and NONE states actually occur in the output across signals."""
    df = make_synthetic_ohlcv(300)
    res = generate_volume_signals(df)

    all_values = set()
    for col in res.columns:
        all_values.update(res[col].unique())

    assert SignalState.BUY in all_values
    assert SignalState.SELL in all_values
    assert SignalState.HOLD in all_values
    assert SignalState.NONE in all_values


def test_volume_signals_short_dataframe():
    """Verify graceful execution without exceptions when given a short dataframe (warmup edge case)."""
    df_short = make_synthetic_ohlcv(10)
    res = generate_volume_signals(df_short)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert len(res.columns) == 18

    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_volume_signals_empty_dataframe():
    """Verify handling of empty dataframe."""
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_volume_signals(df_empty)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert len(res.columns) == 18
    for col in res.columns:
        assert col.endswith("_signal")


def test_volume_signals_normalization():
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
    res = generate_volume_signals(df_upper)

    assert len(res) == 50
    assert len(res.columns) == 18


def test_volume_signals_missing_columns():
    """Verify that missing OHLCV columns raise ValueError."""
    df_invalid = pd.DataFrame({"close": [100.0, 101.0, 102.0], "volume": [10, 20, 30]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        generate_volume_signals(df_invalid)


def test_volume_signals_zero_volume():
    """Verify graceful execution with zero volume across all bars."""
    df = make_synthetic_ohlcv(50)
    df["volume"] = 0.0
    res = generate_volume_signals(df)

    assert len(res) == 50
    assert len(res.columns) == 18
    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_volume_signals_constant_volume():
    """Verify graceful execution with constant volume across all bars."""
    df = make_synthetic_ohlcv(50)
    df["volume"] = 1000.0
    res = generate_volume_signals(df)

    assert len(res) == 50
    assert len(res.columns) == 18
    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_volume_signals_datetime_index_preserved():
    """Verify that custom DatetimeIndex is preserved across all output columns."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = make_synthetic_ohlcv(100)
    df.index = dates
    res = generate_volume_signals(df)

    assert isinstance(res.index, pd.DatetimeIndex)
    assert res.index.equals(dates)


def test_vwap_band_reversal_direct():
    """Verify _calc_vwap_band_reversal helper on bounce and rejection scenarios."""
    close = pd.Series([100.0, 90.0, 95.0, 110.0, 105.0])
    lband = pd.Series([92.0, 92.0, 92.0, 92.0, 92.0])
    hband = pd.Series([108.0, 108.0, 108.0, 108.0, 108.0])

    # Bar 1: close 90 <= lband 92 (below lower band)
    # Bar 2: close 95 > lband 92 (bounces back above lower band -> BUY)
    # Bar 3: close 110 >= hband 108 (above upper band)
    # Bar 4: close 105 < hband 108 (rejects back below upper band -> SELL)
    res = _calc_vwap_band_reversal(close, lband, hband)
    assert res.iloc[2] == SignalState.BUY
    assert res.iloc[4] == SignalState.SELL


def test_volume_spike_direction_direct():
    """Verify _calc_volume_spike_direction triggers BUY on bullish candle and SELL on bearish candle during spike."""
    n = 25
    open_p = pd.Series([100.0] * n)
    # Candle 21: bullish close (105 > 100) with volume spike
    # Candle 22: bearish close (95 < 100) with volume spike
    # Other candles: normal volume (1000), close == open
    close = pd.Series([100.0] * n)
    close.iloc[21] = 105.0
    close.iloc[22] = 95.0

    volume = pd.Series([1000.0] * n)
    volume.iloc[21] = 5000.0  # 5x SMA -> spike
    volume.iloc[22] = 5000.0  # 5x SMA -> spike

    res = _calc_volume_spike_direction(open_p, close, volume, window=20, threshold=2.0)
    assert res.iloc[21] == SignalState.BUY
    assert res.iloc[22] == SignalState.SELL
    assert res.iloc[10] == SignalState.HOLD or res.iloc[10] == SignalState.NONE


def test_obv_ema_cross_signal_direct():
    """Verify OBV EMA 20 cross produces BUY on cross up and SELL on cross down."""
    n = 60
    # Create price going steadily up, then steadily down
    close = pd.Series([100.0 + i for i in range(30)] + [130.0 - i for i in range(30)])
    volume = pd.Series([1000.0] * n)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": volume,
        }
    )
    res = generate_volume_signals(df)
    sig = res["volume_obv_ema_cross_20_signal"]
    assert SignalState.BUY in sig.values
    assert SignalState.SELL in sig.values


def test_cmf_signals_direct():
    """Verify CMF zero cross and threshold signals trigger appropriately."""
    n = 60
    # Consistently closing near high (strong positive accumulation)
    high = pd.Series([100.0 + i for i in range(n)])
    low = high - 2.0
    close = high  # close at high -> positive CLV
    open_p = low
    volume = pd.Series([1000.0] * n)

    df_bull = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    res_bull = generate_volume_signals(df_bull)
    assert (res_bull["volume_cmf_zero_cross_20_signal"].iloc[25:] == SignalState.BUY).all() or (
        res_bull["volume_cmf_threshold_cross_20_signal"].iloc[25:] == SignalState.BUY
    ).all()

    # Consistently closing near low (strong negative distribution)
    close_bear = low
    df_bear = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close_bear,
            "volume": volume,
        }
    )
    res_bear = generate_volume_signals(df_bear)
    assert (res_bear["volume_cmf_zero_cross_20_signal"].iloc[25:] == SignalState.SELL).all() or (
        res_bear["volume_cmf_threshold_cross_20_signal"].iloc[25:] == SignalState.SELL
    ).all()


def test_vwap_crossover_signals():
    """Verify VWAP 20, 50, 100 crossover signals on actual price crossovers."""
    n = 150
    # Price stays flat at 100 during warmup, then dips below, then surges above VWAP
    close = [100.0] * 110 + [90.0, 92.0] + [110.0, 115.0] + [100.0] * 36
    df = pd.DataFrame(
        {
            "open": pd.Series(close) - 0.5,
            "high": pd.Series(close) + 0.5,
            "low": pd.Series(close) - 0.5,
            "close": pd.Series(close),
            "volume": pd.Series([1000.0] * n),
        }
    )
    res = generate_volume_signals(df)
    assert SignalState.BUY in res["volume_vwap_cross_20_signal"].values
    assert SignalState.BUY in res["volume_vwap_cross_50_signal"].values
    assert SignalState.BUY in res["volume_vwap_cross_100_signal"].values


def test_pvt_adl_force_eom_signals_direct():
    """Verify PVT, ADL, Force Index, and EoM signals generate valid signals on trending data."""
    n = 120
    # Oscillating price wave with non-midpoint close so CLV and ADL fluctuate
    t = np.linspace(0, 4 * np.pi, n)
    close = pd.Series(100.0 + 20.0 * np.sin(t))
    high = close + np.where(np.cos(t) > 0, 2.0, 0.5)
    low = close - np.where(np.cos(t) < 0, 2.0, 0.5)
    volume = pd.Series([2000.0] * n)

    df = pd.DataFrame(
        {
            "open": close - 0.5 * np.cos(t),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    res = generate_volume_signals(df)
    assert SignalState.BUY in res["volume_pvt_ma_cross_14_signal"].values
    assert SignalState.SELL in res["volume_pvt_ma_cross_14_signal"].values
    assert SignalState.BUY in res["volume_adl_ma_cross_signal"].values
    assert SignalState.SELL in res["volume_adl_ma_cross_signal"].values
    assert SignalState.BUY in res["volume_force_index_13_signal"].values
    assert SignalState.SELL in res["volume_force_index_13_signal"].values
    assert SignalState.BUY in res["volume_eom_zero_14_signal"].values
    assert SignalState.SELL in res["volume_eom_zero_14_signal"].values


def test_volume_helpers_edge_cases():
    """Verify edge case handling in _crossover_signal and _bound_signal helpers."""
    from signalx.signals.volume import _bound_signal, _crossover_signal

    empty_s = pd.Series([], dtype=float)
    res_cross_empty = _crossover_signal(empty_s, empty_s)
    assert len(res_cross_empty) == 0

    s = pd.Series([10.0, 50.0, 90.0])
    res_bound = _bound_signal(s, lower=20.0, upper=80.0, buy_below=True)
    assert res_bound.iloc[0] == SignalState.BUY
    assert res_bound.iloc[1] == SignalState.HOLD
    assert res_bound.iloc[2] == SignalState.SELL


def test_signals_package_export():
    """Verify generate_volume_signals is properly exported from signalx.signals."""
    from signalx.signals import generate_volume_signals as exported_func

    assert callable(exported_func)
