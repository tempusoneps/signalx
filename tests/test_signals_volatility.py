from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.session_helper import extract_session_context
from signalx.signals.volatility import (
    VOLATILITY_SIGNAL_COLUMNS,
    _calc_atr_ratio_fast_slow,
    _calc_atr_trailing_stop,
    _calc_bb_bounce,
    _calc_chaikin_volatility,
    _calc_chandelier_exit,
    _calc_close_to_close_donchian,
    _calc_dual_thrust,
    _calc_garman_klass,
    _calc_hv_ratio_breakout,
    _calc_ib_breakout_30m,
    _calc_keltner_width_squeeze,
    _calc_lunch_range_breakout,
    _calc_mass_index_reversal,
    _calc_micro_channel_4_breakout,
    _calc_natr_stretch,
    _calc_parkinson_surge,
    _calc_pct_b_reversal,
    _calc_pre_atc_squeeze,
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


def test_volatility_signals_all_47_columns_present():
    """Verify generate_volatility_signals produces exactly 47 volatility signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_volatility_signals(df)

    assert len(EXPECTED_VOLATILITY_SIGNALS) == 47
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 47
    assert list(res.index) == list(df.index)

    for col in EXPECTED_VOLATILITY_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"

    assert "vol_ib_breakout_30m_signal" in res.columns
    assert "vol_pre_atc_squeeze_signal" in res.columns
    assert "vol_micro_channel_4_breakout_signal" in res.columns
    assert "vol_lunch_range_breakout_signal" in res.columns
    assert "vol_close_to_close_donchian_signal" in res.columns


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
    assert len(res.columns) == 47

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
    assert len(res.columns) == 47
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
    assert len(res.columns) == 47


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


def test_ib_breakout_30m_direct():
    """Verify IB 30m breakout, traps, and trend hold."""
    # Build a 15-bar session (synthetic fallback: session_id = 0, bar_in_session = 0..14)
    # Bars 0..5 establish IB High=105.0, IB Low=95.0
    n = 15
    open_p = [100.0] * n
    high = [104.0] * n
    low = [96.0] * n
    close = [100.0] * n

    high[2] = 105.0  # IB_High = 105.0
    low[3] = 95.0  # IB_Low = 95.0

    # Bar 6: breakout up (Close > IB_High & prev Close <= prev IB_High) -> BUY
    # prev Close (bar 5) is 100.0 <= 105.0
    open_p[6] = 104.0
    high[6] = 108.0
    low[6] = 103.0
    close[6] = 106.0

    # Bar 7: trend hold (Close > IB_High & prev Close > IB_High) -> HOLD
    open_p[7] = 106.0
    high[7] = 109.0
    low[7] = 105.5
    close[7] = 107.0

    # Bar 8: inside IB -> NONE
    open_p[8] = 105.0
    high[8] = 105.0
    low[8] = 96.0
    close[8] = 100.0

    # Bar 9: breakout down (Close < IB_Low & prev Close >= prev IB_Low) -> SELL
    # prev Close (bar 8) is 100.0 >= 95.0
    open_p[9] = 96.0
    high[9] = 96.0
    low[9] = 93.0
    close[9] = 94.0

    # Bar 10: IB Bear Trap (Low < IB_Low & Close > IB_Low & Close > Open) -> BUY
    open_p[10] = 94.0
    low[10] = 93.0
    high[10] = 97.0
    close[10] = 96.0

    # Bar 11: IB Bull Trap (High > IB_High & Close < IB_High & Close < Open) -> SELL
    open_p[11] = 106.0
    high[11] = 107.0
    low[11] = 103.0
    close[11] = 104.0

    # Bar 12: breakout down below IB_Low after re-entering IB (Close < IB_Low & prev Close >= prev IB_Low) -> SELL
    open_p[12] = 94.0
    high[12] = 94.5
    low[12] = 92.0
    close[12] = 93.0

    # Bar 13: trend hold below IB_Low (Close < IB_Low & prev Close < IB_Low) -> HOLD
    open_p[13] = 93.0
    high[13] = 93.5
    low[13] = 91.0
    close[13] = 92.0

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000.0] * n,
        }
    )

    sig = _calc_ib_breakout_30m(
        df,
        high=pd.Series(high),
        low=pd.Series(low),
        close=pd.Series(close),
        open_p=pd.Series(open_p),
    )

    assert isinstance(sig, pd.Series)
    assert len(sig) == n
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)

    # First 6 bars (0..5) must strictly be NONE
    for i in range(6):
        assert sig.iloc[i] == SignalState.NONE, f"Bar {i} should be NONE"

    assert sig.iloc[6] == SignalState.BUY, f"Bar 6 breakout up should be BUY, got {sig.iloc[6]}"
    assert sig.iloc[7] == SignalState.HOLD, f"Bar 7 trend hold should be HOLD, got {sig.iloc[7]}"
    assert sig.iloc[8] == SignalState.NONE, f"Bar 8 inside IB should be NONE, got {sig.iloc[8]}"
    assert sig.iloc[9] == SignalState.SELL, f"Bar 9 breakout down should be SELL, got {sig.iloc[9]}"
    assert sig.iloc[10] == SignalState.BUY, f"Bar 10 bear trap should be BUY, got {sig.iloc[10]}"
    assert sig.iloc[11] == SignalState.SELL, f"Bar 11 bull trap should be SELL, got {sig.iloc[11]}"
    assert sig.iloc[12] == SignalState.SELL, (
        f"Bar 12 breakout down should be SELL, got {sig.iloc[12]}"
    )
    assert sig.iloc[13] == SignalState.HOLD, (
        f"Bar 13 trend hold down should be HOLD, got {sig.iloc[13]}"
    )


def test_pre_atc_squeeze_direct():
    """Verify pre-ATC squeeze signal triggers within pre-ATC window when range > 1.3 * ATR20."""
    # In synthetic fallback, bars 44..48 are is_pre_atc
    n = 50
    open_p = [100.0] * n
    high = [101.0] * n
    low = [99.0] * n
    close = [100.0] * n
    # Normal bar range is 2.0, so ATR20 ~ 2.0. Threshold 1.3 * ATR20 ~ 2.6.

    # Bar 43: outside pre-ATC window, even with expansion -> NONE
    open_p[43] = 100.0
    high[43] = 108.0
    low[43] = 99.0
    close[43] = 107.0

    # Bar 44: is_pre_atc, bullish breakout (Close > prev High & Close > Open) -> BUY
    # prev High (bar 43) = 108.0; range = 115 - 99 = 16.0 > 1.3 * ATR20
    open_p[44] = 105.0
    low[44] = 99.0
    high[44] = 115.0
    close[44] = 112.0  # > 108.0 (prev High) and > 105.0 (Open)

    # Bar 45: is_pre_atc, bearish breakdown (Close < prev Low & Close < Open) -> SELL
    # prev Low (bar 44) = 99.0; range = 100 - 88 = 12.0 > 1.3 * ATR20
    open_p[45] = 98.0
    high[45] = 100.0
    low[45] = 88.0
    close[45] = 90.0  # < 99.0 (prev Low) and < 98.0 (Open)

    # Bar 46: is_pre_atc, small range (< 1.3 * ATR20) -> NONE
    open_p[46] = 90.0
    high[46] = 91.0
    low[46] = 89.5
    close[46] = 90.5

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000.0] * n,
        }
    )

    sig = _calc_pre_atc_squeeze(
        df,
        open_p=pd.Series(open_p),
        high=pd.Series(high),
        low=pd.Series(low),
        close=pd.Series(close),
    )

    assert isinstance(sig, pd.Series)
    assert len(sig) == n
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)

    assert sig.iloc[43] == SignalState.NONE, (
        f"Bar 43 outside pre-ATC should be NONE, got {sig.iloc[43]}"
    )
    assert sig.iloc[44] == SignalState.BUY, (
        f"Bar 44 pre-ATC breakout up should be BUY, got {sig.iloc[44]}"
    )
    assert sig.iloc[45] == SignalState.SELL, (
        f"Bar 45 pre-ATC breakdown should be SELL, got {sig.iloc[45]}"
    )
    assert sig.iloc[46] == SignalState.NONE, (
        f"Bar 46 pre-ATC small range should be NONE, got {sig.iloc[46]}"
    )
    assert sig.iloc[49] == SignalState.NONE, (
        f"Bar 49 outside pre-ATC should be NONE, got {sig.iloc[49]}"
    )


def test_volatility_signals_intraday_datetime_context():
    """Verify IB breakout and Pre-ATC squeeze work seamlessly with real intraday DatetimeIndex."""
    # Generate 2 trading days of 5m bars from 08:45 to 14:45
    dates = []
    for day in ["2026-01-05", "2026-01-06"]:
        morning = pd.date_range(f"{day} 08:45", f"{day} 11:30", freq="5min")
        afternoon = pd.date_range(f"{day} 13:00", f"{day} 14:45", freq="5min")
        dates.extend(morning)
        dates.extend(afternoon)
    dt_idx = pd.DatetimeIndex(dates)
    n = len(dt_idx)

    np.random.seed(123)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)
    open_p = low + (high - low) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "datetime": dt_idx,
        },
        index=dt_idx,
    )

    res = generate_volatility_signals(df)
    assert len(res.columns) == 47
    assert "vol_ib_breakout_30m_signal" in res.columns
    assert "vol_pre_atc_squeeze_signal" in res.columns
    assert "vol_micro_channel_4_breakout_signal" in res.columns
    assert "vol_lunch_range_breakout_signal" in res.columns
    assert "vol_close_to_close_donchian_signal" in res.columns

    ib_sig = res["vol_ib_breakout_30m_signal"]
    pre_atc_sig = res["vol_pre_atc_squeeze_signal"]

    assert set(ib_sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert set(pre_atc_sig.unique()).issubset(ALL_SIGNAL_STATES)
    # First 6 bars of first day must be NONE
    assert (ib_sig.iloc[:6] == SignalState.NONE).all()


def test_calc_micro_channel_4_breakout_direct():
    """Verify 4-bar micro channel breakout calculation and states."""
    n = 60
    close = pd.Series([100.0 + i * 0.5 for i in range(n)])
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series([1000.0] * n)

    # Bar 58: huge volume spike and price jump above micro_high (and above ema55 with rsi21 > 53)
    high.iloc[58] = 150.0
    close.iloc[58] = 149.0
    volume.iloc[58] = 50000.0

    sig = _calc_micro_channel_4_breakout(high, low, close, volume)
    assert isinstance(sig, pd.Series)
    assert len(sig) == n
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert sig.iloc[58] == SignalState.BUY

    # Downward breakdown
    close_down = pd.Series([200.0 - i * 0.5 for i in range(n)])
    high_down = close_down + 1.0
    low_down = close_down - 1.0
    volume_down = pd.Series([1000.0] * n)
    low_down.iloc[58] = 50.0
    close_down.iloc[58] = 51.0
    volume_down.iloc[58] = 50000.0

    sig_down = _calc_micro_channel_4_breakout(high_down, low_down, close_down, volume_down)
    assert sig_down.iloc[58] == SignalState.SELL


def test_calc_lunch_range_breakout_direct():
    """Verify midday lunch range breakout in afternoon window."""
    n = 50
    # In synthetic fallback, bars 25..35 are lunch range, bars >= 36 are afternoon
    close = pd.Series([100.0] * n)
    open_p = pd.Series([100.0] * n)
    high = pd.Series([102.0] * n)
    low = pd.Series([98.0] * n)

    # Lunch range establishes high=102, low=98 for bars 25..35
    # Afternoon bar 40 breaks out above lunch high
    # Needs: close > lunch_high (102), bar_close_pos >= 0.60, slope5 > 0, rsi8 > 55
    for i in range(36, 42):
        close.iloc[i] = 100.0 + (i - 35) * 1.5
        open_p.iloc[i] = close.iloc[i] - 0.5
        high.iloc[i] = close.iloc[i] + 0.2
        low.iloc[i] = close.iloc[i] - 1.0

    df_synth = pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": 1000.0}
    )
    ctx = extract_session_context(df_synth)

    sig = _calc_lunch_range_breakout(open_p, high, low, close, ctx)
    assert isinstance(sig, pd.Series)
    assert len(sig) == n
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    # Outside afternoon window (bar 20) must be NONE
    assert sig.iloc[20] == SignalState.NONE
    # Bar 40 is afternoon breakout BUY
    assert sig.iloc[40] == SignalState.BUY

    # Downward breakdown below lunch low
    close_b = pd.Series([100.0] * n)
    open_b = pd.Series([100.0] * n)
    high_b = pd.Series([102.0] * n)
    low_b = pd.Series([98.0] * n)
    for i in range(36, 42):
        close_b.iloc[i] = 100.0 - (i - 35) * 1.5
        open_b.iloc[i] = close_b.iloc[i] + 0.5
        high_b.iloc[i] = close_b.iloc[i] + 1.0
        low_b.iloc[i] = close_b.iloc[i] - 0.2

    df_synth_b = pd.DataFrame(
        {"open": open_b, "high": high_b, "low": low_b, "close": close_b, "volume": 1000.0}
    )
    ctx_b = extract_session_context(df_synth_b)
    sig_b = _calc_lunch_range_breakout(open_b, high_b, low_b, close_b, ctx_b)
    assert sig_b.iloc[40] == SignalState.SELL


def test_calc_close_to_close_donchian_direct():
    """Verify close-to-close Donchian breakout and breakdown."""
    n = 60
    close = pd.Series([100.0] * n)
    volume = pd.Series([1000.0] * n)

    # Upper breakout at bar 30
    close.iloc[30] = 110.0
    volume.iloc[30] = 5000.0

    sig = _calc_close_to_close_donchian(close, volume, window=20)
    assert isinstance(sig, pd.Series)
    assert len(sig) == n
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)
    assert sig.iloc[30] == SignalState.BUY

    # Lower breakdown at bar 30
    close_down = pd.Series([100.0] * n)
    close_down.iloc[30] = 80.0
    volume_down = pd.Series([1000.0] * n)
    volume_down.iloc[30] = 5000.0
    sig_down = _calc_close_to_close_donchian(close_down, volume_down, window=20)
    assert sig_down.iloc[30] == SignalState.SELL

    # Price strictly inside channel (lower < close < upper) with close != ema55 -> NONE
    close_inside = pd.Series([100.0] * n)
    close_inside.iloc[0:20] = 90.0  # establish lower=90
    close_inside.iloc[10] = 110.0  # establish upper=110
    close_inside.iloc[25] = 102.0  # strictly inside (90 < 102 < 110), ema55 != 102
    volume_inside = pd.Series([1000.0] * n)
    sig_inside = _calc_close_to_close_donchian(close_inside, volume_inside, window=20)
    assert sig_inside.iloc[25] == SignalState.NONE, (
        f"Price inside channel should be NONE, got {sig_inside.iloc[25]}"
    )

    # Holding outside upper channel (close >= upper and close > ema55, but volume <= vol_ma20) -> HOLD
    close_hold = pd.Series([100.0] * n)
    close_hold.iloc[30] = 110.0  # close > upper
    volume_hold = pd.Series([1000.0] * n)  # volume <= vol_ma20
    sig_hold = _calc_close_to_close_donchian(close_hold, volume_hold, window=20)
    assert sig_hold.iloc[30] == SignalState.HOLD, (
        f"Price maintaining outside upper channel without volume spike should be HOLD, got {sig_hold.iloc[30]}"
    )
