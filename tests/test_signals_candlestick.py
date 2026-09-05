from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.candlestick import (
    CANDLESTICK_SIGNAL_COLUMNS,
    _calc_break_of_structure,
    _calc_change_of_character,
    _calc_consecutive_directional,
    _calc_doji_reversal,
    _calc_engulfing,
    _calc_fvg_bearish_mitigation,
    _calc_fvg_bullish_mitigation,
    _calc_hammer_star,
    _calc_harami,
    _calc_inducement_sweep,
    _calc_inside_bar,
    _calc_judas_swing,
    _calc_marubozu,
    _calc_morning_evening_star,
    _calc_narrow_range_7_breakout,
    _calc_order_block_retest,
    _calc_outside_bar,
    _calc_pdh_pdl_sweep,
    _calc_piercing_darkcloud,
    _calc_pinbar,
    _calc_three_soldiers_crows,
    _calc_thrust_bar,
    _calc_tweezer_tops_bottoms,
    _calc_wide_range_reversal,
    generate_candlestick_signals,
)


def make_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataset with realistic candles for candlestick tests."""
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


EXPECTED_CANDLESTICK_SIGNALS = CANDLESTICK_SIGNAL_COLUMNS


def test_candlestick_signals_all_38_columns_present():
    """Verify generate_candlestick_signals produces exactly 38 candlestick signals."""
    df = make_synthetic_ohlcv(250)
    res = generate_candlestick_signals(df)

    assert len(EXPECTED_CANDLESTICK_SIGNALS) == 38
    assert EXPECTED_CANDLESTICK_SIGNALS == CANDLESTICK_SIGNAL_COLUMNS
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 38
    assert list(res.index) == list(df.index)

    for col in EXPECTED_CANDLESTICK_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"

    assert "cdl_pdh_pdl_sweep_signal" in res.columns


def test_candlestick_signals_all_states_valid():
    """Verify that every value in every signal column is strictly within ALL_SIGNAL_STATES and has no NaNs."""
    df = make_synthetic_ohlcv(250)
    res = generate_candlestick_signals(df)

    for col in res.columns:
        assert not res[col].isna().any(), f"Column {col} contains unexpected NaN values"
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), (
            f"Column {col} contains invalid states: {unique_vals - ALL_SIGNAL_STATES}"
        )


def test_candlestick_signals_state_occurrences():
    """Verify that BUY, SELL, HOLD, and NONE states occur in the output across signals."""
    df = make_synthetic_ohlcv(500)
    res = generate_candlestick_signals(df)

    all_values = set()
    for col in res.columns:
        all_values.update(res[col].unique())

    assert SignalState.BUY in all_values
    assert SignalState.SELL in all_values
    assert SignalState.HOLD in all_values
    assert SignalState.NONE in all_values


def test_candlestick_signals_short_dataframe():
    """Verify graceful execution without exceptions when given short dataframes."""
    for n in [1, 2, 3, 5, 10]:
        df_short = make_synthetic_ohlcv(n)
        res = generate_candlestick_signals(df_short)

        assert isinstance(res, pd.DataFrame)
        assert len(res) == n
        assert len(res.columns) == len(CANDLESTICK_SIGNAL_COLUMNS)

        for col in res.columns:
            assert not res[col].isna().any()
            unique_vals = set(res[col].unique())
            assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_candlestick_signals_empty_dataframe():
    """Verify handling of empty dataframe."""
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = generate_candlestick_signals(df_empty)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert len(res.columns) == len(CANDLESTICK_SIGNAL_COLUMNS)
    for col in res.columns:
        assert col.endswith("_signal")


def test_candlestick_signals_normalization():
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
    res = generate_candlestick_signals(df_upper)

    assert len(res) == 50
    assert len(res.columns) == len(CANDLESTICK_SIGNAL_COLUMNS)


def test_candlestick_signals_missing_columns():
    """Verify that missing OHLCV columns raise ValueError."""
    df_invalid = pd.DataFrame({"open": [100.0, 101.0], "close": [102.0, 103.0]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        generate_candlestick_signals(df_invalid)


def test_candlestick_signals_flat_candles():
    """Verify graceful execution when candles are flat (high == low == open == close)."""
    df_flat = pd.DataFrame(
        {
            "open": [100.0] * 20,
            "high": [100.0] * 20,
            "low": [100.0] * 20,
            "close": [100.0] * 20,
            "volume": [1000.0] * 20,
        }
    )
    res = generate_candlestick_signals(df_flat)

    assert len(res) == 20
    assert len(res.columns) == len(CANDLESTICK_SIGNAL_COLUMNS)
    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_candlestick_signals_datetime_index_preserved():
    """Verify that custom DatetimeIndex is preserved across all output columns."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = make_synthetic_ohlcv(100)
    df.index = dates
    res = generate_candlestick_signals(df)

    assert isinstance(res.index, pd.DatetimeIndex)
    assert res.index.equals(dates)


def test_engulfing_signal_direct():
    """Verify Bullish and Bearish Engulfing pattern detection."""
    # Bar 0: baseline
    # Bar 1: red candle (open 105, close 100)
    # Bar 2: green engulfing (open 98, close 107) -> BUY
    # Bar 3: green candle (open 107, close 112)
    # Bar 4: red engulfing (open 114, close 105) -> SELL
    df = pd.DataFrame(
        {
            "open": [100.0, 105.0, 98.0, 107.0, 114.0],
            "high": [102.0, 106.0, 108.0, 113.0, 115.0],
            "low": [99.0, 99.0, 97.0, 106.0, 104.0],
            "close": [101.0, 100.0, 107.0, 112.0, 105.0],
            "volume": [1000.0] * 5,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_engulfing_signal"]

    assert sig.iloc[2] == SignalState.BUY
    assert sig.iloc[4] == SignalState.SELL


def test_hammer_star_signal_direct():
    """Verify Hammer (long lower wick) and Shooting Star (long upper wick) detection."""
    # Bar 0: Hammer (open 100, close 101, high 101.5, low 90.0) -> lower wick = 10, upper wick = 0.5, body = 1 -> BUY
    # Bar 1: Shooting Star (open 100, close 99, high 110.0, low 98.5) -> upper wick = 10, lower wick = 0.5, body = 1 -> SELL
    # Bar 2: Normal candle (open 100, close 102, high 103, low 99) -> HOLD
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [101.5, 110.0, 103.0],
            "low": [90.0, 98.5, 99.0],
            "close": [101.0, 99.0, 102.0],
            "volume": [1000.0] * 3,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_hammer_star_signal"]

    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD


def test_pinbar_signal_direct():
    """Verify Pinbar rejection bar detection (wick >= 60% range)."""
    # Bar 0: Bullish pinbar (open 100, close 102, high 103, low 90) -> range = 13, lower wick = 10 (>= 60%) -> BUY
    # Bar 1: Bearish pinbar (open 100, close 98, high 112, low 97) -> range = 15, upper wick = 12 (>= 60%) -> SELL
    # Bar 2: Neutral bar (open 100, close 101, high 102, low 99) -> HOLD
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [103.0, 112.0, 102.0],
            "low": [90.0, 97.0, 99.0],
            "close": [102.0, 98.0, 101.0],
            "volume": [1000.0] * 3,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_pinbar_signal"]

    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD


def test_marubozu_signal_direct():
    """Verify Bullish and Bearish Marubozu detection (body >= 85% range)."""
    # Bar 0: Bullish Marubozu (open 100, close 110, high 110.5, low 99.5) -> range = 11, body = 10 (90.9% >= 85%) -> BUY
    # Bar 1: Bearish Marubozu (open 110, close 100, high 110.5, low 99.5) -> range = 11, body = 10 (90.9% >= 85%) -> SELL
    # Bar 2: Small body candle (open 100, close 101, high 110, low 90) -> HOLD
    df = pd.DataFrame(
        {
            "open": [100.0, 110.0, 100.0],
            "high": [110.5, 110.5, 110.0],
            "low": [99.5, 99.5, 90.0],
            "close": [110.0, 100.0, 101.0],
            "volume": [1000.0] * 3,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_marubozu_signal"]

    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD


def test_harami_signal_direct():
    """Verify Bullish and Bearish Harami detection (inside body)."""
    # Bar 0: large red body (open 110, close 90)
    # Bar 1: small green body inside (open 95, close 105) -> BUY (Bullish Harami)
    # Bar 2: large green body (open 90, close 110)
    # Bar 3: small red body inside (open 105, close 95) -> SELL (Bearish Harami)
    df = pd.DataFrame(
        {
            "open": [110.0, 95.0, 90.0, 105.0],
            "high": [112.0, 106.0, 112.0, 106.0],
            "low": [88.0, 94.0, 88.0, 94.0],
            "close": [90.0, 105.0, 110.0, 95.0],
            "volume": [1000.0] * 4,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_harami_signal"]

    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[3] == SignalState.SELL


def test_inside_bar_breakout_signal_direct():
    """Verify Inside Bar detection with directional close."""
    # Bar 0: mother bar (high 110, low 90)
    # Bar 1: inside bar, bullish close (open 95, close 105, high 106, low 94) -> BUY
    # Bar 2: mother bar (high 110, low 90)
    # Bar 3: inside bar, bearish close (open 105, close 95, high 106, low 94) -> SELL
    df = pd.DataFrame(
        {
            "open": [95.0, 95.0, 105.0, 105.0],
            "high": [110.0, 106.0, 110.0, 106.0],
            "low": [90.0, 94.0, 90.0, 94.0],
            "close": [105.0, 105.0, 95.0, 95.0],
            "volume": [1000.0] * 4,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_inside_bar_breakout_signal"]

    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[3] == SignalState.SELL


def test_outside_bar_signal_direct():
    """Verify Outside Bar detection with directional close."""
    # Bar 0: inner bar (high 105, low 95)
    # Bar 1: outside bar, bullish close (open 94, close 108, high 110, low 90) -> BUY
    # Bar 2: inner bar (high 105, low 95)
    # Bar 3: outside bar, bearish close (open 108, close 92, high 110, low 90) -> SELL
    df = pd.DataFrame(
        {
            "open": [98.0, 94.0, 102.0, 108.0],
            "high": [105.0, 110.0, 105.0, 110.0],
            "low": [95.0, 90.0, 95.0, 90.0],
            "close": [102.0, 108.0, 98.0, 92.0],
            "volume": [1000.0] * 4,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_outside_bar_signal"]

    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[3] == SignalState.SELL


def test_doji_reversal_signal_direct():
    """Verify Dragonfly Doji (bullish) and Gravestone Doji (bearish) detection."""
    # Bar 0: Dragonfly Doji (open 100, close 100.2, high 100.5, low 90.0) -> range 10.5, body 0.2, lower wick 10.0 -> BUY
    # Bar 1: Gravestone Doji (open 100, close 99.8, high 110.0, low 99.5) -> range 10.5, body 0.2, upper wick 10.0 -> SELL
    # Bar 2: Standard bar (open 100, close 105, high 106, low 99) -> HOLD
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.5, 110.0, 106.0],
            "low": [90.0, 99.5, 99.0],
            "close": [100.2, 99.8, 105.0],
            "volume": [1000.0] * 3,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_doji_reversal_signal"]

    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD


def test_three_soldiers_crows_signal_direct():
    """Verify Three White Soldiers (BUY) and Three Black Crows (SELL) detection."""
    # Bars 0, 1, 2: Three White Soldiers
    # Bars 3, 4, 5: Three Black Crows
    df = pd.DataFrame(
        {
            "open": [100.0, 103.0, 106.0, 110.0, 107.0, 104.0],
            "high": [104.0, 107.0, 110.0, 111.0, 108.0, 105.0],
            "low": [99.0, 102.0, 105.0, 106.0, 103.0, 100.0],
            "close": [103.5, 106.5, 109.5, 106.5, 103.5, 100.5],
            "volume": [1000.0] * 6,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_three_soldiers_crows_signal"]

    assert sig.iloc[2] == SignalState.BUY
    assert sig.iloc[5] == SignalState.SELL


def test_consecutive_3_and_5_signals_direct():
    """Verify Consecutive 3 and 5 directional bar signals."""
    # 5 consecutive green bars, then 5 consecutive red bars
    opens = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 108.0, 106.0, 104.0, 102.0]
    closes = [101.0, 103.0, 105.0, 107.0, 109.0, 109.0, 107.0, 105.0, 103.0, 101.0]
    highs = [max(o, c) + 1.0 for o, c in zip(opens, closes, strict=False)]
    lows = [min(o, c) - 1.0 for o, c in zip(opens, closes, strict=False)]

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * 10,
        }
    )
    res = generate_candlestick_signals(df)

    # Consecutive 3
    sig_3 = res["cdl_consecutive_3_signal"]
    assert sig_3.iloc[0] == SignalState.NONE
    assert sig_3.iloc[1] == SignalState.NONE
    assert sig_3.iloc[2] == SignalState.BUY
    assert sig_3.iloc[3] == SignalState.BUY
    assert sig_3.iloc[4] == SignalState.BUY
    assert sig_3.iloc[7] == SignalState.SELL
    assert sig_3.iloc[8] == SignalState.SELL
    assert sig_3.iloc[9] == SignalState.SELL

    # Consecutive 5
    sig_5 = res["cdl_consecutive_5_signal"]
    assert sig_5.iloc[3] == SignalState.NONE
    assert sig_5.iloc[4] == SignalState.BUY
    assert sig_5.iloc[9] == SignalState.SELL


def test_morning_evening_star_signal_direct():
    """Verify Morning Star (BUY) and Evening Star (SELL) detection."""
    # Bar 0: large red candle (open 110, close 95)
    # Bar 1: small body gapping low (open 92, close 93, high 94, low 91)
    # Bar 2: strong green candle closing > midpoint 102.5 (open 94, close 105) -> Morning Star BUY
    # Bar 3: large green candle (open 95, close 110)
    # Bar 4: small body gapping high (open 112, close 111, high 113, low 110)
    # Bar 5: strong red candle closing < midpoint 102.5 (open 110, close 98) -> Evening Star SELL
    df = pd.DataFrame(
        {
            "open": [110.0, 92.0, 94.0, 95.0, 112.0, 110.0],
            "high": [111.0, 94.0, 106.0, 111.0, 113.0, 111.0],
            "low": [94.0, 91.0, 93.0, 94.0, 110.0, 97.0],
            "close": [95.0, 93.0, 105.0, 110.0, 111.0, 98.0],
            "volume": [1000.0] * 6,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_morning_evening_star_signal"]

    assert sig.iloc[2] == SignalState.BUY
    assert sig.iloc[5] == SignalState.SELL


def test_piercing_darkcloud_signal_direct():
    """Verify Piercing Line (BUY) and Dark Cloud Cover (SELL) detection."""
    # Bar 0: red candle (open 110, close 90, midpoint 100)
    # Bar 1: green candle opening <= 90 and closing > 100 and <= 110 (open 88, close 104) -> BUY
    # Bar 2: green candle (open 90, close 110, midpoint 100)
    # Bar 3: red candle opening >= 110 and closing < 100 and >= 90 (open 112, close 96) -> SELL
    df = pd.DataFrame(
        {
            "open": [110.0, 88.0, 90.0, 112.0],
            "high": [111.0, 105.0, 111.0, 113.0],
            "low": [89.0, 87.0, 89.0, 95.0],
            "close": [90.0, 104.0, 110.0, 96.0],
            "volume": [1000.0] * 4,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_piercing_darkcloud_signal"]

    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[3] == SignalState.SELL


def test_tweezer_tops_bottoms_signal_direct():
    """Verify Tweezer Bottoms (matching lows -> BUY) and Tweezer Tops (matching highs -> SELL)."""
    # Bar 0: red candle (open 105, close 95, low 90.0)
    # Bar 1: green candle (open 95, close 105, low 90.0) -> matching low -> BUY
    # Bar 2: green candle (open 95, close 105, high 110.0)
    # Bar 3: red candle (open 105, close 95, high 110.0) -> matching high -> SELL
    df = pd.DataFrame(
        {
            "open": [105.0, 95.0, 95.0, 105.0],
            "high": [106.0, 106.0, 110.0, 110.0],
            "low": [90.0, 90.0, 94.0, 94.0],
            "close": [95.0, 105.0, 105.0, 95.0],
            "volume": [1000.0] * 4,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_tweezer_tops_bottoms_signal"]

    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[3] == SignalState.SELL


def test_fvg_bullish_mitigation_signal_direct():
    """Verify Bullish FVG creation and mitigation / invalidation."""
    # Bar 0: High = 100, Low = 95
    # Bar 1: Strong up move Open = 101, Close = 108, High = 109, Low = 101
    # Bar 2: Open = 108, Close = 112, High = 114, Low = 105 (Low > High[0]=100 => Bullish FVG [100, 105] created)
    # Bar 3: Retrace into [100, 105], Open = 104, Close = 106, High = 107, Low = 102 (Close > Open => BUY)
    # Bar 4: Breakdown below 100, Open = 103, Close = 98, High = 104, Low = 97 => SELL
    df = pd.DataFrame(
        {
            "open": [96.0, 101.0, 108.0, 104.0, 103.0],
            "high": [100.0, 109.0, 114.0, 107.0, 104.0],
            "low": [95.0, 101.0, 105.0, 102.0, 97.0],
            "close": [99.0, 108.0, 112.0, 106.0, 98.0],
            "volume": [1000.0] * 5,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_fvg_bullish_mitigation_signal"]
    assert sig.iloc[3] == SignalState.BUY
    assert sig.iloc[4] == SignalState.SELL


def test_fvg_bearish_mitigation_signal_direct():
    """Verify Bearish FVG creation and mitigation / invalidation."""
    # Bar 0: High = 115, Low = 110
    # Bar 1: Strong down move Open = 109, Close = 102, High = 109, Low = 101
    # Bar 2: Open = 102, Close = 98, High = 104, Low = 96 (High < Low[0]=110 => Bearish FVG [104, 110] created)
    # Bar 3: Retrace into [104, 110], Open = 108, Close = 105, High = 109, Low = 104 (Close < Open => SELL)
    # Bar 4: Breakout above 110, Open = 107, Close = 113, High = 114, Low = 106 => BUY
    df = pd.DataFrame(
        {
            "open": [114.0, 109.0, 102.0, 108.0, 107.0],
            "high": [115.0, 109.0, 104.0, 109.0, 114.0],
            "low": [110.0, 101.0, 96.0, 104.0, 106.0],
            "close": [111.0, 102.0, 98.0, 105.0, 113.0],
            "volume": [1000.0] * 5,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_fvg_bearish_mitigation_signal"]
    assert sig.iloc[3] == SignalState.SELL
    assert sig.iloc[4] == SignalState.BUY


def test_order_block_retest_signal_direct():
    """Verify Bullish and Bearish Order Block formation and retest."""
    # Bullish OB: Bar 0 is red (Open 102, Close 98). Bars 1, 2, 3 are green.
    # Bar 4: Retraces into [98, 102] with green close (Open 99, Close 101) -> BUY
    df = pd.DataFrame(
        {
            "open": [102.0, 99.0, 104.0, 108.0, 99.0],
            "high": [103.0, 104.0, 108.0, 112.0, 102.0],
            "low": [97.0, 98.0, 103.0, 107.0, 98.5],
            "close": [98.0, 103.0, 107.0, 111.0, 101.0],
            "volume": [1000.0] * 5,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_order_block_retest_signal"]
    assert sig.iloc[4] == SignalState.BUY


def test_break_of_structure_signal_direct():
    """Verify BOS breakout with trend alignment."""
    # 60 bars uptrend so SMA20 > SMA50
    np.random.seed(42)
    closes = [100.0 + i * 0.5 for i in range(55)]
    # Bar 54 has max high of past 10 bars
    # Bar 55 breaks above max high of past 10 bars with SMA20 > SMA50
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    opens = [c - 0.2 for c in closes]
    # Breakout bar
    opens.append(closes[-1])
    closes.append(highs[-1] + 2.0)
    highs.append(closes[-1] + 0.5)
    lows.append(opens[-1] - 0.5)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(opens),
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_break_of_structure_signal"]
    assert sig.iloc[-1] == SignalState.BUY


def test_change_of_character_signal_direct():
    """Verify CHoCH structure break against prior trend regime."""
    # Downtrend for 55 bars so SMA20 < SMA50
    closes = [200.0 - i * 0.5 for i in range(55)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    opens = [c + 0.2 for c in closes]

    # Bar 55 breaks above 5-bar high
    opens.append(closes[-1])
    max_h5 = max(highs[-5:])
    closes.append(max_h5 + 2.0)
    highs.append(closes[-1] + 0.5)
    lows.append(opens[-1] - 0.5)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(opens),
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_change_of_character_signal"]
    assert sig.iloc[-1] == SignalState.BUY


def test_judas_swing_signal_direct():
    """Verify Judas Swing false breakout with reversal."""
    # 5 bars baseline
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 101.0, 100.0, 97.0, 105.0],
            "high": [103.0, 103.0, 104.0, 103.0, 102.0, 102.0, 106.0],
            "low": [98.0, 99.0, 100.0, 99.0, 98.0, 94.0, 99.0],  # Bar 5 low=94 < min_5 (98)
            "close": [101.0, 102.0, 101.0, 100.0, 99.0, 101.5, 96.0],  # Bar 5 close in upper half
            "volume": [1000.0] * 7,
        }
    )
    # Bar 5: Low = 94 < min(Low5)=98, Close=101.5 > Open=97 and Close > (102+94)/2=98 -> BUY
    # Bar 6: High = 106 > max(High5)=104, Close=96 < Open=105 and Close < (106+99)/2=102.5 -> SELL
    res = generate_candlestick_signals(df)
    sig = res["cdl_judas_swing_signal"]
    assert sig.iloc[5] == SignalState.BUY
    assert sig.iloc[6] == SignalState.SELL


def test_inducement_sweep_signal_direct():
    """Verify Inducement Sweep minor liquidity grab."""
    # Bar 0: High = 105, Low = 95
    # Bar 1: Low = 93 < Low[0], Lower wick = 98 - 93 = 5, Range = 103 - 93 = 10 (>= 50%), Close 102 > Open 98 -> BUY
    # Bar 2: High = 108 > High[1], Upper wick = 108 - 103 = 5, Range = 108 - 98 = 10 (>= 50%), Close 99 < Open 103 -> SELL
    df = pd.DataFrame(
        {
            "open": [100.0, 98.0, 103.0],
            "high": [105.0, 103.0, 108.0],
            "low": [95.0, 93.0, 98.0],
            "close": [102.0, 102.0, 99.0],
            "volume": [1000.0] * 3,
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_inducement_sweep_signal"]
    assert sig.iloc[1] == SignalState.BUY
    assert sig.iloc[2] == SignalState.SELL


def test_thrust_bar_signal_direct():
    """Verify Thrust Bar large body directional candle."""
    # 20 bars small body (body = 1.0)
    opens = [100.0] * 20
    closes = [101.0] * 20
    highs = [101.5] * 20
    lows = [99.5] * 20

    # Bar 20: large thrust bullish bar (body = 5.0 >= 1.8 * 1.0, body >= 75% of range 5.5)
    opens.append(100.0)
    closes.append(105.0)
    highs.append(105.2)
    lows.append(99.8)

    # Bar 21: large thrust bearish bar
    opens.append(105.0)
    closes.append(100.0)
    highs.append(105.2)
    lows.append(99.8)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(opens),
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_thrust_bar_signal"]
    assert sig.iloc[20] == SignalState.BUY
    assert sig.iloc[21] == SignalState.SELL


def test_narrow_range_7_breakout_signal_direct():
    """Verify NR7 breakout / breakdown."""
    # 6 bars range = 10.0
    opens = [100.0] * 6
    closes = [105.0] * 6
    highs = [110.0] * 6
    lows = [100.0] * 6

    # Bar 6: NR7 bar (range = 2.0)
    opens.append(104.0)
    closes.append(105.0)
    highs.append(106.0)
    lows.append(104.0)

    # Bar 7: Breakout above High of bar 6 (High=106.0) -> Close=108.0 -> BUY
    opens.append(105.0)
    closes.append(108.0)
    highs.append(109.0)
    lows.append(104.5)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(opens),
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_narrow_range_7_breakout_signal"]
    assert sig.iloc[7] == SignalState.BUY


def test_wide_range_reversal_signal_direct():
    """Verify Wide Range Reversal bar."""
    # 20 bars range = 2.0
    opens = [100.0] * 20
    closes = [101.0] * 20
    highs = [102.0] * 20
    lows = [100.0] * 20

    # Bar 20: Wide range (range = 10.0 >= 2.5 * 2.0), low = 90, high = 100, finishes at 99 (top 30%) -> BUY
    opens.append(98.0)
    closes.append(99.0)
    highs.append(100.0)
    lows.append(90.0)

    # Bar 21: Wide range, finishes at bottom 30% -> SELL
    opens.append(102.0)
    closes.append(101.0)
    highs.append(110.0)
    lows.append(100.0)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(opens),
        }
    )
    res = generate_candlestick_signals(df)
    sig = res["cdl_wide_range_reversal_signal"]
    assert sig.iloc[20] == SignalState.BUY
    assert sig.iloc[21] == SignalState.SELL


def test_pdh_pdl_sweep_signal_direct():
    """Verify Prior Day High / Low (PDH/PDL) liquidity sweep signal logic and zero lookahead bias."""
    dates = [
        "2026-09-01 09:00",
        "2026-09-01 09:05",
        "2026-09-01 09:10",
        "2026-09-01 09:15",
        "2026-09-01 09:20",
    ] + [
        "2026-09-02 09:00",
        "2026-09-02 09:05",
        "2026-09-02 09:10",
        "2026-09-02 09:15",
        "2026-09-02 09:20",
    ]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [94.0, 95.0, 96.0, 97.0, 98.0, 95.0, 91.0, 99.0, 100.0, 89.0],
            "high": [96.0, 98.0, 100.0, 99.0, 98.0, 97.0, 93.0, 102.0, 103.0, 90.0],
            "low": [92.0, 90.0, 95.0, 94.0, 93.0, 93.0, 88.0, 97.0, 99.0, 86.0],
            "close": [95.0, 96.0, 97.0, 98.0, 97.0, 96.0, 92.0, 98.0, 101.0, 87.0],
            "volume": [1000.0] * 10,
        }
    )

    res = generate_candlestick_signals(df)
    sig = res["cdl_pdh_pdl_sweep_signal"]

    # Session 1 (bars 0..4) has no prior session -> MUST BE ALL NONE (zero lookahead!)
    for i in range(5):
        assert sig.iloc[i] == SignalState.NONE, f"Session 1 bar {i} should be NONE"

    # Session 2: PDH is 100.0, PDL is 90.0
    # Bar 5 (index 5): within range -> NONE
    assert sig.iloc[5] == SignalState.NONE
    # Bar 6 (index 6): low 88 < 90, close 92 > 90, close 92 > open 91 -> BUY
    assert sig.iloc[6] == SignalState.BUY
    # Bar 7 (index 7): high 102 > 100, close 98 < 100, close 98 < open 99 -> SELL
    assert sig.iloc[7] == SignalState.SELL
    # Bar 8 (index 8): high 103 > 100, close 101 > 100 -> NONE
    assert sig.iloc[8] == SignalState.NONE
    # Bar 9 (index 9): low 86 < 90, close 87 < 90 -> NONE
    assert sig.iloc[9] == SignalState.NONE


def test_helpers_edge_cases():
    """Verify edge case handling in helper functions."""
    empty_s = pd.Series([], dtype=float)
    res_empty = _calc_engulfing(empty_s, empty_s)
    assert len(res_empty) == 0

    res_ham_empty = _calc_hammer_star(empty_s, empty_s, empty_s, empty_s)
    assert len(res_ham_empty) == 0

    res_pin_empty = _calc_pinbar(empty_s, empty_s, empty_s, empty_s)
    assert len(res_pin_empty) == 0

    res_maru_empty = _calc_marubozu(empty_s, empty_s, empty_s, empty_s)
    assert len(res_maru_empty) == 0

    res_harami_empty = _calc_harami(empty_s, empty_s)
    assert len(res_harami_empty) == 0

    res_inside_empty = _calc_inside_bar(empty_s, empty_s, empty_s, empty_s)
    assert len(res_inside_empty) == 0

    res_outside_empty = _calc_outside_bar(empty_s, empty_s, empty_s, empty_s)
    assert len(res_outside_empty) == 0

    res_doji_empty = _calc_doji_reversal(empty_s, empty_s, empty_s, empty_s)
    assert len(res_doji_empty) == 0

    res_soldiers_empty = _calc_three_soldiers_crows(empty_s, empty_s)
    assert len(res_soldiers_empty) == 0

    res_consec_empty = _calc_consecutive_directional(empty_s, empty_s, window=3)
    assert len(res_consec_empty) == 0

    res_star_empty = _calc_morning_evening_star(empty_s, empty_s)
    assert len(res_star_empty) == 0

    res_piercing_empty = _calc_piercing_darkcloud(empty_s, empty_s)
    assert len(res_piercing_empty) == 0

    res_tweezer_empty = _calc_tweezer_tops_bottoms(empty_s, empty_s, empty_s, empty_s)
    assert len(res_tweezer_empty) == 0

    # New 10 helpers
    assert len(_calc_fvg_bullish_mitigation(empty_s, empty_s, empty_s, empty_s)) == 0
    assert len(_calc_fvg_bearish_mitigation(empty_s, empty_s, empty_s, empty_s)) == 0
    assert len(_calc_order_block_retest(empty_s, empty_s, empty_s, empty_s)) == 0
    assert len(_calc_break_of_structure(empty_s, empty_s, empty_s)) == 0
    assert len(_calc_change_of_character(empty_s, empty_s, empty_s)) == 0
    assert len(_calc_judas_swing(empty_s, empty_s, empty_s, empty_s)) == 0
    assert len(_calc_inducement_sweep(empty_s, empty_s, empty_s, empty_s)) == 0
    assert len(_calc_thrust_bar(empty_s, empty_s, empty_s, empty_s)) == 0
    assert len(_calc_narrow_range_7_breakout(empty_s, empty_s, empty_s)) == 0
    assert len(_calc_wide_range_reversal(empty_s, empty_s, empty_s)) == 0
    assert len(_calc_pdh_pdl_sweep(None, empty_s, empty_s, empty_s, empty_s)) == 0


def test_signals_package_export():
    """Verify generate_candlestick_signals is properly exported from signalx.signals."""
    from signalx.signals import generate_candlestick_signals as exported_func

    assert callable(exported_func)
