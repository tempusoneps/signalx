from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.session_helper import extract_session_context
from signalx.signals.volume import (
    VOLUME_SIGNAL_COLUMNS,
    _calc_rolling_volume_shelf_zscore,
    _calc_vn30_late_session_vwap_momentum,
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


def test_volume_signals_all_34_columns_present():
    """Verify generate_volume_signals produces exactly the expected 34 volume signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)

    assert len(EXPECTED_VOLUME_SIGNALS) == 34
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 34
    assert list(res.index) == list(df.index)

    for col in EXPECTED_VOLUME_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"

    assert "volume_session_vwap_cross_signal" in res.columns
    assert "volume_rvol_time_bucket_signal" in res.columns
    assert "volume_cvd_divergence_signal" in res.columns
    assert "volume_stopping_climax_signal" in res.columns
    assert "volume_rolling_shelf_zscore_signal" in res.columns
    assert "volume_vn30_late_session_vwap_momentum_signal" in res.columns


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
    assert len(res.columns) == 34

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
    assert len(res.columns) == 34
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
    assert len(res.columns) == 34


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
    assert len(res.columns) == 34
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
    assert len(res.columns) == 34
    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_volume_signals_flat_data():
    """Verify that flat data (zero price movement) executes without RuntimeWarning (e.g. division by zero)."""
    import warnings

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
        res = generate_volume_signals(df_flat)

    assert len(res) == 50
    assert len(res.columns) == 34
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


def test_klinger_osc_cross_signal():
    """Verify KVO signal cross produces valid states and triggers on trend change."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)
    sig = res["volume_klinger_osc_cross_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert SignalState.BUY in sig.values or SignalState.HOLD in sig.values


def test_elder_ray_bull_bear_signal():
    """Verify Elder Ray Bull/Bear Power produces BUY/SELL signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)
    sig = res["volume_elder_ray_bull_bear_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert SignalState.BUY in sig.values or SignalState.SELL in sig.values


def test_volume_climax_absorption_signal():
    """Verify volume climax absorption signals on extreme volume wicks."""
    n = 30
    open_p = pd.Series([100.0] * n)
    high = pd.Series([102.0] * n)
    low = pd.Series([98.0] * n)
    close = pd.Series([100.0] * n)
    volume = pd.Series([1000.0] * n)

    # Candle 25: volume spike (5000 >= 3x 1000), long lower wick (low=90, open=98, close=101, high=102 -> lower wick=8, range=12, 8/12=66% >= 40%), bullish close
    high.iloc[25] = 102.0
    low.iloc[25] = 90.0
    open_p.iloc[25] = 98.0
    close.iloc[25] = 101.0
    volume.iloc[25] = 5000.0

    # Candle 26: volume spike, long upper wick (high=112, open=102, close=99, low=98 -> upper wick=10, range=14, 10/14=71% >= 40%), bearish close
    high.iloc[26] = 112.0
    low.iloc[26] = 98.0
    open_p.iloc[26] = 102.0
    close.iloc[26] = 99.0
    volume.iloc[26] = 5000.0

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume})
    res = generate_volume_signals(df)
    sig = res["volume_climax_absorption_signal"]
    assert sig.iloc[25] == SignalState.BUY
    assert sig.iloc[26] == SignalState.SELL


def test_twiggs_money_flow_cross_signal():
    """Verify Twiggs Money Flow zero centerline cross produces signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)
    sig = res["volume_twiggs_money_flow_cross_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert SignalState.BUY in sig.values or SignalState.SELL in sig.values


def test_nvi_ema_cross_signal():
    """Verify Negative Volume Index EMA cross produces signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)
    sig = res["volume_nvi_pvi_cross_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert SignalState.BUY in sig.values or SignalState.SELL in sig.values


def test_vwap_anchored_dev1_signal():
    """Verify VWAP +-1 std band bounce and rejection."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)
    sig = res["volume_vwap_anchored_dev1_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert SignalState.BUY in sig.values or SignalState.SELL in sig.values


def test_vwap_anchored_dev3_signal():
    """Verify VWAP +-3 std extreme bands trigger signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)
    sig = res["volume_vwap_anchored_dev3_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert SignalState.BUY in sig.values or SignalState.HOLD in sig.values


def test_volume_delta_proxy_surge_signal():
    """Verify delta proxy volume surge on strong directional close near extremes."""
    n = 20
    open_p = pd.Series([100.0] * n)
    high = pd.Series([110.0] * n)
    low = pd.Series([90.0] * n)
    close = pd.Series([100.0] * n)
    volume = pd.Series([1000.0] * n)

    # Bar 10: Close = 108 (close near high 110, low 90 -> (108-90)/(110-90) = 18/20 = 0.90 > 0.70, close > open) -> BUY
    open_p.iloc[10] = 95.0
    close.iloc[10] = 108.0

    # Bar 11: Close = 92 (close near low 90, high 110 -> (110-92)/(110-90) = 18/20 = 0.90 > 0.70, close < open) -> SELL
    open_p.iloc[11] = 105.0
    close.iloc[11] = 92.0

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume})
    res = generate_volume_signals(df)
    sig = res["volume_delta_proxy_surge_signal"]
    assert sig.iloc[10] == SignalState.BUY
    assert sig.iloc[11] == SignalState.SELL


def test_vwma_sma_divergence_signal():
    """Verify VWMA vs SMA divergence produces BUY/SELL signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)
    sig = res["volume_vwma_sma_divergence_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert SignalState.BUY in sig.values or SignalState.SELL in sig.values


def test_volume_weighted_rsi_signal():
    """Verify Volume-Weighted RSI (14) overbought/oversold boundaries."""
    df = make_synthetic_ohlcv(250)
    res = generate_volume_signals(df)
    sig = res["volume_volume_weighted_rsi_14_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert SignalState.BUY in sig.values or SignalState.SELL in sig.values


def test_signals_package_export():
    """Verify generate_volume_signals is properly exported from signalx.signals."""
    from signalx.signals import generate_volume_signals as exported_func

    assert callable(exported_func)


def test_session_vwap_cross_signal_direct():
    """Verify session VWAP crossover produces BUY on cross up, SELL on cross down, NONE on bar 0."""
    dates = []
    for d in ["2026-01-05", "2026-01-06"]:
        dates.extend(pd.date_range(f"{d} 09:00", periods=10, freq="5min"))
    dates = pd.DatetimeIndex(dates)
    n = len(dates)

    close = [100.0] * 10 + [100.0, 90.0, 91.0, 92.0, 93.0, 110.0, 112.0, 111.0, 85.0, 84.0]
    volume = [1000.0] * n
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    open_p = [c for c in close]

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "date": dates,
        },
        index=dates,
    )
    res = generate_volume_signals(df)
    sig = res["volume_session_vwap_cross_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert sig.iloc[0] == SignalState.NONE
    assert sig.iloc[10] == SignalState.NONE
    assert SignalState.BUY in sig.iloc[10:].values
    assert SignalState.SELL in sig.iloc[10:].values


def test_rvol_time_bucket_signal_direct():
    """Verify RVOL time bucket signal triggers BUY on bullish candle and SELL on bearish candle when RVOL >= 2.0."""
    dates = []
    for d in ["2026-01-05", "2026-01-06", "2026-01-07"]:
        dates.extend(pd.date_range(f"{d} 09:00", periods=5, freq="5min"))
    dates = pd.DatetimeIndex(dates)
    n = len(dates)

    volume = [1000.0] * 10 + [1000.0, 3000.0, 1000.0, 3000.0, 1000.0]
    open_p = [100.0] * n
    close = [100.0] * n
    close[11] = 105.0
    close[13] = 95.0

    high = [max(o, c) + 1.0 for o, c in zip(open_p, close, strict=False)]
    low = [min(o, c) - 1.0 for o, c in zip(open_p, close, strict=False)]

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "date": dates,
        },
        index=dates,
    )
    res = generate_volume_signals(df)
    sig = res["volume_rvol_time_bucket_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert (sig.iloc[:5] == SignalState.NONE).all()
    assert sig.iloc[11] == SignalState.BUY
    assert sig.iloc[13] == SignalState.SELL
    assert sig.iloc[10] == SignalState.NONE


def test_cvd_divergence_signal_direct():
    """Verify CVD divergence signal detects bullish and bearish divergence."""
    n = 25
    open_p = [100.0] * n
    high = [102.0] * n
    low = [98.0] * n
    close = [100.0] * n
    volume = [1000.0] * n

    # Bar 5: strong push up
    high[5] = 110.0
    close[5] = 110.0
    volume[5] = 5000.0

    # Bars 6, 7: drop with negative CVD
    close[6] = 95.0
    open_p[6] = 100.0
    high[6] = 101.0
    low[6] = 94.0
    volume[6] = 3000.0

    close[7] = 95.0
    open_p[7] = 96.0
    high[7] = 97.0
    low[7] = 94.0
    volume[7] = 2000.0

    # Bar 8: price makes new 10-bar high close=111 (> 110), but red candle (open=112, close=111)
    # CVD peak was at bar 5 (5000), while at bar 8 CVD is ~2690 < 5000 -> Bearish divergence -> SELL
    open_p[8] = 112.0
    high[8] = 113.0
    low[8] = 105.0
    close[8] = 111.0
    volume[8] = 1000.0

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    res = generate_volume_signals(df)
    sig = res["volume_cvd_divergence_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert sig.iloc[8] == SignalState.SELL


def test_stopping_climax_signal_direct():
    """Verify stopping volume and climax exhaustion signal on extreme volume wicks."""
    n = 25
    open_p = [100.0] * n
    high = [102.0] * n
    low = [98.0] * n
    close = [100.0] * n
    volume = [1000.0] * n

    high[21] = 102.0
    low[21] = 90.0
    open_p[21] = 99.0
    close[21] = 100.0
    volume[21] = 3000.0

    high[22] = 110.0
    low[22] = 98.0
    open_p[22] = 101.0
    close[22] = 100.0
    volume[22] = 3000.0

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    res = generate_volume_signals(df)
    sig = res["volume_stopping_climax_signal"]
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert sig.iloc[21] == SignalState.BUY
    assert sig.iloc[22] == SignalState.SELL
    assert sig.iloc[10] == SignalState.NONE


def test_calc_rolling_volume_shelf_zscore_direct():
    """Verify rolling volume shelf Z-score triggers buy, sell, hold, and none."""
    n = 35
    # 1. Bullish scenario: steady price then sharp rally with volume expansion
    close_up = pd.Series([100.0] * n)
    volume_up = pd.Series([1000.0] * n)
    for i in range(20, 26):
        close_up.iloc[i] = 100.0 + (i - 19) * 3.0
        volume_up.iloc[i] = 3000.0

    sig_up = _calc_rolling_volume_shelf_zscore(close_up, volume_up, shelf_len=12)
    assert isinstance(sig_up, pd.Series)
    assert len(sig_up) == n
    assert set(sig_up.unique()).issubset(ALL_SIGNAL_STATES)
    # At bar 25: Z_shelf >= 0.5, RSI8 > 56, slope5 > 0, vol > vol_ma20 -> BUY
    assert sig_up.iloc[25] == SignalState.BUY

    # 2. Bearish scenario: steady price then sharp decline with volume expansion
    close_down = pd.Series([100.0] * n)
    volume_down = pd.Series([1000.0] * n)
    for i in range(20, 26):
        close_down.iloc[i] = 100.0 - (i - 19) * 3.0
        volume_down.iloc[i] = 3000.0

    sig_down = _calc_rolling_volume_shelf_zscore(close_down, volume_down, shelf_len=12)
    assert sig_down.iloc[25] == SignalState.SELL

    # 3. Flat scenario -> NONE
    close_flat = pd.Series([100.0] * n)
    volume_flat = pd.Series([1000.0] * n)
    sig_flat = _calc_rolling_volume_shelf_zscore(close_flat, volume_flat, shelf_len=12)
    assert sig_flat.iloc[25] == SignalState.NONE

    # 4. Moderate stretch with slope5 > 0 but volume <= vol_ma20 -> HOLD
    close_hold = pd.Series([100.0] * n)
    volume_hold = pd.Series([1000.0] * n)
    for i in range(20, 26):
        close_hold.iloc[i] = 100.0 + (i - 19) * 0.8
        volume_hold.iloc[i] = 800.0  # volume below vol_ma20 (~980)

    sig_hold = _calc_rolling_volume_shelf_zscore(close_hold, volume_hold, shelf_len=12)
    assert sig_hold.iloc[25] == SignalState.HOLD


def test_calc_vn30_late_session_vwap_momentum_direct():
    """Verify VN30 late session VWAP momentum triggers buy, sell, hold, and none."""
    n = 50
    # Synthetic session: 50 bars per session, late session is bar 40..47
    # 1. Bullish breakout in late session
    open_p = pd.Series([100.0] * n)
    high = pd.Series([101.0] * n)
    low = pd.Series([99.0] * n)
    close = pd.Series([100.0] * n)
    volume = pd.Series([1000.0] * n)

    # Establish baseline VWAP around 100
    # From bar 35 onwards, price moves up with high volume
    for i in range(35, 48):
        close.iloc[i] = 100.0 + (i - 34) * 0.8
        open_p.iloc[i] = close.iloc[i] - 0.3
        high.iloc[i] = close.iloc[i] + 0.5
        low.iloc[i] = close.iloc[i] - 0.5
        volume.iloc[i] = 2000.0

    df_synth = pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": volume}
    )
    ctx = extract_session_context(df_synth)

    sig = _calc_vn30_late_session_vwap_momentum(open_p, high, low, close, volume, ctx)
    assert isinstance(sig, pd.Series)
    assert len(sig) == n
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    # Outside active window (e.g. bar 15) must be NONE
    assert sig.iloc[15] == SignalState.NONE
    # Inside active window (bar 42..45) with strong VWAP stretch -> BUY
    assert sig.iloc[45] == SignalState.BUY

    # 2. Bearish breakdown in late session
    open_d = pd.Series([100.0] * n)
    high_d = pd.Series([101.0] * n)
    low_d = pd.Series([99.0] * n)
    close_d = pd.Series([100.0] * n)
    volume_d = pd.Series([1000.0] * n)

    for i in range(35, 48):
        close_d.iloc[i] = 100.0 - (i - 34) * 0.8
        open_d.iloc[i] = close_d.iloc[i] + 0.3
        high_d.iloc[i] = close_d.iloc[i] + 0.5
        low_d.iloc[i] = close_d.iloc[i] - 0.5
        volume_d.iloc[i] = 2000.0

    df_synth_d = pd.DataFrame(
        {"open": open_d, "high": high_d, "low": low_d, "close": close_d, "volume": volume_d}
    )
    ctx_d = extract_session_context(df_synth_d)

    sig_d = _calc_vn30_late_session_vwap_momentum(open_d, high_d, low_d, close_d, volume_d, ctx_d)
    assert sig_d.iloc[45] == SignalState.SELL

    # 3. Inside window with moderate stretch (|Z_vwap| > 0.30) but range_pct < 0.0012 -> HOLD
    open_h = pd.Series([100.0] * n)
    high_h = pd.Series([100.05] * n)
    low_h = pd.Series([99.95] * n)
    close_h = pd.Series([100.0] * n)
    volume_h = pd.Series([1000.0] * n)
    # Small drift, tiny range_pct < 0.0012, but std will be small so |Z| > 0.3
    for i in range(38, 48):
        close_h.iloc[i] = 100.03
        high_h.iloc[i] = 100.05
        low_h.iloc[i] = 99.95
    df_synth_h = pd.DataFrame(
        {"open": open_h, "high": high_h, "low": low_h, "close": close_h, "volume": volume_h}
    )
    ctx_h = extract_session_context(df_synth_h)
    sig_h = _calc_vn30_late_session_vwap_momentum(open_h, high_h, low_h, close_h, volume_h, ctx_h)
    assert sig_h.iloc[45] in (SignalState.HOLD, SignalState.NONE)
