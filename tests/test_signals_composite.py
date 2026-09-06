from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.signals.candlestick import generate_candlestick_signals
from signalx.signals.composite import (
    COMPOSITE_SIGNAL_COLUMNS,
    _calc_breakout_volume_confirmed,
    _calc_ma_consensus,
    _calc_master_ensemble,
    _calc_master_ensemble_v2,
    _calc_mean_reversion_confluence,
    _calc_momentum_consensus,
    _calc_smc_trend_volume_confluence,
    _calc_squeeze_momentum_volume_surge,
    _calc_trend_consensus,
    _calc_trend_momentum_align,
    _calc_triple_screen_trading_system,
    _calc_vn30_intraday_confluence,
    generate_composite_signals,
)
from signalx.signals.momentum import generate_momentum_signals
from signalx.signals.smc import generate_smc_signals
from signalx.signals.statistical import generate_statistical_signals
from signalx.signals.trend import generate_trend_signals
from signalx.signals.volatility import generate_volatility_signals
from signalx.signals.volume import generate_volume_signals


def make_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataset with strong trends and mean reversion regimes."""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.02

    if n >= 150:
        # Bullish regime
        returns[30:60] = 0.02 + np.abs(np.random.randn(30)) * 0.005
        # Bearish regime
        returns[80:110] = -0.02 - np.abs(np.random.randn(30)) * 0.005

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.random.uniform(0.005, 0.03, n))
    low = close * (1.0 - np.random.uniform(0.005, 0.03, n))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n)

    base_volume = np.random.randint(1000, 20000, n).astype(float)
    if n >= 150:
        base_volume[30:60] *= 3.0
        base_volume[80:110] *= 3.0

    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": base_volume,
        }
    )


EXPECTED_COMPOSITE_SIGNALS = COMPOSITE_SIGNAL_COLUMNS


def generate_all_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to generate all 7 family signal DataFrames concatenated."""
    trend = generate_trend_signals(df)
    mom = generate_momentum_signals(df)
    vol = generate_volatility_signals(df)
    volume = generate_volume_signals(df)
    cdl = generate_candlestick_signals(df)
    smc = generate_smc_signals(df)
    stat = generate_statistical_signals(df)
    return pd.concat([trend, mom, vol, volume, cdl, smc, stat], axis=1)


def test_composite_signals_all_13_columns_present():
    """Verify generate_composite_signals produces exactly 13 expected composite signals."""
    df = make_synthetic_ohlcv(250)
    intermediate = generate_all_intermediate(df)
    res = generate_composite_signals(df, intermediate)

    assert len(EXPECTED_COMPOSITE_SIGNALS) == 13
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 250
    assert len(res.columns) == 13
    assert list(res.index) == list(df.index)

    for col in EXPECTED_COMPOSITE_SIGNALS:
        assert col in res.columns, f"Expected column {col} missing from output"
        assert col.endswith("_signal"), f"Column {col} must end with '_signal'"
    assert "comp_vn30_intraday_confluence_signal" in res.columns


def test_composite_signals_all_states_valid():
    """Verify that every value in every signal column is strictly within ALL_SIGNAL_STATES and has no NaNs."""
    df = make_synthetic_ohlcv(250)
    intermediate = generate_all_intermediate(df)
    res = generate_composite_signals(df, intermediate)

    for col in res.columns:
        assert not res[col].isna().any(), f"Column {col} contains unexpected NaN values"
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES), (
            f"Column {col} contains invalid states: {unique_vals - ALL_SIGNAL_STATES}"
        )


def test_composite_signals_state_occurrences():
    """Verify that BUY, SELL, HOLD, and NONE states occur in the output across composite signals."""
    df = make_synthetic_ohlcv(300)
    intermediate = generate_all_intermediate(df)
    res = generate_composite_signals(df, intermediate)

    all_values = set()
    for col in res.columns:
        all_values.update(res[col].unique())

    assert SignalState.BUY in all_values
    assert SignalState.SELL in all_values
    assert SignalState.HOLD in all_values
    assert SignalState.NONE in all_values


def test_composite_signals_without_intermediate():
    """Verify generate_composite_signals auto-computes intermediate signals if None is passed."""
    df = make_synthetic_ohlcv(100)
    res = generate_composite_signals(df, None)

    assert len(res.columns) == 13
    assert len(res) == 100
    for col in EXPECTED_COMPOSITE_SIGNALS:
        assert col in res.columns
        assert set(res[col].unique()).issubset(ALL_SIGNAL_STATES)


def test_composite_signals_short_dataframe():
    """Verify graceful execution without exceptions when given a short dataframe."""
    df_short = make_synthetic_ohlcv(10)
    intermediate = generate_all_intermediate(df_short)
    res = generate_composite_signals(df_short, intermediate)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert len(res.columns) == 13

    for col in res.columns:
        assert not res[col].isna().any()
        unique_vals = set(res[col].unique())
        assert unique_vals.issubset(ALL_SIGNAL_STATES)


def test_composite_signals_empty_dataframe():
    """Verify handling of empty dataframe."""
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    empty_intermediate = pd.DataFrame()
    res = generate_composite_signals(df_empty, empty_intermediate)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert len(res.columns) == 13
    for col in res.columns:
        assert col.endswith("_signal")


def test_composite_signals_normalization():
    """Verify uppercase OHLCV columns normalization in composite signals."""
    df = make_synthetic_ohlcv(50)
    df_upper = df.rename(
        columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "volume": "VOLUME"}
    )
    res = generate_composite_signals(df_upper, None)

    assert len(res) == 50
    assert len(res.columns) == 13


def test_composite_signals_missing_columns():
    """Verify missing OHLCV columns raise ValueError."""
    df_invalid = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5]})
    with pytest.raises(ValueError, match="Missing required OHLCV column"):
        generate_composite_signals(df_invalid, None)


def test_composite_signals_datetime_index_preserved():
    """Verify DatetimeIndex is preserved in output."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = make_synthetic_ohlcv(100)
    df.index = dates
    intermediate = generate_all_intermediate(df)
    res = generate_composite_signals(df, intermediate)

    assert isinstance(res.index, pd.DatetimeIndex)
    assert (res.index == dates).all()


def test_trend_consensus_direct():
    """Verify trend majority vote consensus logic."""
    idx = range(4)
    # Row 0: 3 BUY, 1 SELL -> BUY (>50%)
    # Row 1: 1 BUY, 3 SELL -> SELL (>50%)
    # Row 2: 2 BUY, 2 SELL -> HOLD (neither >50%)
    # Row 3: 4 NONE -> NONE
    df_trend = pd.DataFrame(
        {
            "trend_sig1_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.BUY,
                SignalState.NONE,
            ],
            "trend_sig2_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.BUY,
                SignalState.NONE,
            ],
            "trend_sig3_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.SELL,
                SignalState.NONE,
            ],
            "trend_sig4_signal": [
                SignalState.SELL,
                SignalState.SELL,
                SignalState.SELL,
                SignalState.NONE,
            ],
        },
        index=idx,
    )
    sig = _calc_trend_consensus(df_trend)
    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD
    assert sig.iloc[3] == SignalState.NONE


def test_momentum_consensus_direct():
    """Verify momentum majority vote consensus logic."""
    idx = range(3)
    df_mom = pd.DataFrame(
        {
            "mom_sig1_signal": [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            "mom_sig2_signal": [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            "mom_sig3_signal": [SignalState.BUY, SignalState.HOLD, SignalState.NONE],
        },
        index=idx,
    )
    sig = _calc_momentum_consensus(df_mom)
    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD


def test_master_ensemble_direct():
    """Verify master ensemble broad consensus logic."""
    idx = range(4)
    # 10 signal columns
    # Row 0: 4 BUY (40% >= 35%), 0 SELL -> BUY
    # Row 1: 0 BUY, 5 SELL (50% >= 35%) -> SELL
    # Row 2: 2 BUY (20%), 2 SELL (20%) -> HOLD
    # Row 3: all NONE -> NONE
    data = {}
    for i in range(10):
        data[f"sig_{i}_signal"] = [SignalState.HOLD] * 4

    df_signals = pd.DataFrame(data, index=idx)
    # Row 0
    for i in range(4):
        df_signals.iloc[0, i] = SignalState.BUY
    # Row 1
    for i in range(5):
        df_signals.iloc[1, i] = SignalState.SELL
    # Row 2
    df_signals.iloc[2, 0] = SignalState.BUY
    df_signals.iloc[2, 1] = SignalState.BUY
    df_signals.iloc[2, 2] = SignalState.SELL
    df_signals.iloc[2, 3] = SignalState.SELL
    # Row 3
    for i in range(10):
        df_signals.iloc[3, i] = SignalState.NONE

    sig = _calc_master_ensemble(df_signals)
    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD
    assert sig.iloc[3] == SignalState.NONE


def test_ma_consensus_direct():
    """Verify moving average consensus logic."""
    idx = range(3)
    df_ma = pd.DataFrame(
        {
            "trend_sma_cross_5_20_signal": [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            "trend_ema_cross_9_21_signal": [SignalState.BUY, SignalState.SELL, SignalState.HOLD],
            "trend_price_above_sma20_signal": [SignalState.BUY, SignalState.SELL, SignalState.NONE],
        },
        index=idx,
    )
    sig = _calc_ma_consensus(df_ma)
    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD


def test_trend_momentum_align_direct():
    """Verify trend & momentum alignment confluence logic."""
    idx = range(4)
    trend_sig = pd.Series(
        [SignalState.BUY, SignalState.SELL, SignalState.BUY, SignalState.NONE], index=idx
    )
    mom_sig = pd.Series(
        [SignalState.BUY, SignalState.SELL, SignalState.SELL, SignalState.NONE], index=idx
    )

    align_sig = _calc_trend_momentum_align(trend_sig, mom_sig)
    assert align_sig.iloc[0] == SignalState.BUY
    assert align_sig.iloc[1] == SignalState.SELL
    assert align_sig.iloc[2] == SignalState.HOLD
    assert align_sig.iloc[3] == SignalState.NONE


def test_breakout_volume_confirmed_direct():
    """Verify breakout confirmed with volume spike logic."""
    idx = range(4)
    df_vol = pd.DataFrame(
        {
            "vol_donchian_breakout_20_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.BUY,
                SignalState.NONE,
            ],
            "vol_bb_breakout_20_20_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.HOLD,
                SignalState.NONE,
            ],
            "volume_spike_direction_20_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.NONE,
                SignalState.NONE,
            ],
        },
        index=idx,
    )
    sig = _calc_breakout_volume_confirmed(df_vol)
    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD
    assert sig.iloc[3] == SignalState.NONE


def test_mean_reversion_confluence_direct():
    """Verify multi-oscillator mean reversion confluence logic."""
    idx = range(4)
    df_mr = pd.DataFrame(
        {
            "mom_rsi_ob_os_14_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.BUY,
                SignalState.NONE,
            ],
            "vol_bb_bounce_20_20_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.HOLD,
                SignalState.NONE,
            ],
            "stat_price_zscore_20_signal": [
                SignalState.BUY,
                SignalState.SELL,
                SignalState.HOLD,
                SignalState.NONE,
            ],
        },
        index=idx,
    )
    sig = _calc_mean_reversion_confluence(df_mr)
    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD
    assert sig.iloc[3] == SignalState.NONE


def test_smc_trend_volume_confluence_direct():
    """Verify SMC + Trend + Volume confluence logic."""
    n = 60
    close = pd.Series([100.0 + i for i in range(n)])
    open_p = close - 0.5
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series([1000.0] * n)
    volume.iloc[-1] = 5000.0  # Volume spike at last bar

    df_sample = pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": volume}
    )
    intermediate = pd.DataFrame(
        {
            "smc_fvg_bullish_mitigation_signal": [SignalState.HOLD] * (n - 1) + [SignalState.BUY],
            "smc_order_block_retest_signal": [SignalState.NONE] * n,
        }
    )

    sig = _calc_smc_trend_volume_confluence(df_sample, intermediate)
    assert sig.iloc[-1] == SignalState.BUY
    assert sig.iloc[0] == SignalState.NONE or sig.iloc[0] == SignalState.HOLD


def test_triple_screen_trading_system_direct():
    """Verify Elder Triple Screen system logic."""
    n = 70
    # Create trending up series then pullback then breakout
    close = [100.0 + i * 0.5 for i in range(n)]
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    open_p = [c - 0.2 for c in close]
    volume = [1000.0] * n

    df_sample = pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": volume}
    )
    sig = _calc_triple_screen_trading_system(df_sample)
    assert isinstance(sig, pd.Series)
    assert set(sig.unique()).issubset(ALL_SIGNAL_STATES)


def test_squeeze_momentum_volume_surge_direct():
    """Verify Squeeze Momentum breakout with volume surge logic."""
    n = 40
    df_sample = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [102.0] * n,
            "volume": [1000.0] * (n - 1) + [5000.0],
        }
    )
    intermediate = pd.DataFrame(
        {
            "vol_squeeze_momentum_pro_signal": [SignalState.HOLD] * (n - 1) + [SignalState.BUY],
        }
    )
    sig = _calc_squeeze_momentum_volume_surge(df_sample, intermediate)
    assert sig.iloc[-1] == SignalState.BUY


def test_master_ensemble_v2_direct():
    """Verify Master Ensemble v2 consensus logic with >=30% threshold."""
    idx = range(4)
    data = {}
    for i in range(10):
        data[f"sig_{i}_signal"] = [SignalState.HOLD] * 4

    df_signals = pd.DataFrame(data, index=idx)
    # Row 0: 3 BUY out of 10 (30% >= 30%), 0 SELL -> BUY
    for i in range(3):
        df_signals.iloc[0, i] = SignalState.BUY
    # Row 1: 3 SELL out of 10 (30% >= 30%), 0 BUY -> SELL
    for i in range(3):
        df_signals.iloc[1, i] = SignalState.SELL
    # Row 2: 1 BUY, 1 SELL (10% < 30%) -> HOLD
    df_signals.iloc[2, 0] = SignalState.BUY
    df_signals.iloc[2, 1] = SignalState.SELL
    # Row 3: all NONE -> NONE
    for i in range(10):
        df_signals.iloc[3, i] = SignalState.NONE

    sig = _calc_master_ensemble_v2(df_signals)
    assert sig.iloc[0] == SignalState.BUY
    assert sig.iloc[1] == SignalState.SELL
    assert sig.iloc[2] == SignalState.HOLD
    assert sig.iloc[3] == SignalState.NONE


def test_vn30_intraday_confluence_direct():
    """Verify VN30 intraday master confluence (Session VWAP + IB/Sweep + Volume)."""
    n = 10
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [102.0] * n,
            "volume": [1000.0] * n,
        }
    )
    # Row 5: Bullish via IB breakout + volume surge (price above VWAP)
    df.loc[5, "close"] = 103.0
    df.loc[5, "volume"] = 5000.0
    # Row 6: Bullish via PDL sweep + VWAP HOLD + volume surge (price above VWAP)
    df.loc[6, "close"] = 103.0
    df.loc[6, "volume"] = 5000.0
    # Row 7: Bearish via IB breakdown + volume surge (price below VWAP)
    df.loc[7, "close"] = 98.0
    df.loc[7, "volume"] = 5000.0
    # Row 8: Bearish via PDH sweep + VWAP SELL + volume surge (price below VWAP)
    df.loc[8, "close"] = 98.0
    df.loc[8, "volume"] = 5000.0
    # Row 9: IB breakout but volume not confirmed (volume = 100.0 while rolling mean > 1000)
    df.loc[9, "volume"] = 100.0

    intermediate = pd.DataFrame(
        {
            "volume_session_vwap_cross_signal": [
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.BUY,
                SignalState.HOLD,
                SignalState.SELL,
                SignalState.SELL,
                SignalState.BUY,
            ],
            "vol_ib_breakout_30m_signal": [
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.BUY,
                SignalState.NONE,
                SignalState.SELL,
                SignalState.NONE,
                SignalState.BUY,
            ],
            "smc_pdh_pdl_sweep_signal": [
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.NONE,
                SignalState.BUY,
                SignalState.NONE,
                SignalState.SELL,
                SignalState.NONE,
            ],
        }
    )

    sig = _calc_vn30_intraday_confluence(df, intermediate)
    assert sig.iloc[5] == SignalState.BUY
    assert sig.iloc[6] == SignalState.BUY
    assert sig.iloc[7] == SignalState.SELL
    assert sig.iloc[8] == SignalState.SELL
    assert sig.iloc[9] == SignalState.NONE

    # Test coded column names fallback (VLM029, VOL043, SMC011)
    df_coded = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [105.0] * 6,
            "low": [95.0] * 6,
            "close": [102.0] * 6,
            "volume": [100.0] * 5 + [1000.0],
        }
    )
    intermediate_coded = pd.DataFrame(
        {
            "VLM029_signal": [SignalState.NONE] * 5 + [SignalState.BUY],
            "VOL043_signal": [SignalState.NONE] * 5 + [SignalState.BUY],
            "SMC011_signal": [SignalState.NONE] * 6,
        }
    )
    sig_coded = _calc_vn30_intraday_confluence(df_coded, intermediate_coded)
    assert sig_coded.iloc[-1] == SignalState.BUY

    # Test missing intermediate columns fallback (returns all NONE)
    sig_empty_intermediate = _calc_vn30_intraday_confluence(df_coded, pd.DataFrame())
    assert (sig_empty_intermediate == SignalState.NONE).all()

    # Test series index alignment in _get_series
    df_idx = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [105.0] * 6,
            "low": [95.0] * 6,
            "close": [103.0] * 6,
            "volume": [100.0] * 5 + [5000.0],
        },
        index=[10, 20, 30, 40, 50, 60],
    )
    inter_idx = pd.DataFrame(
        {
            "volume_session_vwap_cross_signal": [SignalState.BUY],
            "vol_ib_breakout_30m_signal": [SignalState.BUY],
            "smc_pdh_pdl_sweep_signal": [SignalState.NONE],
        },
        index=[60],
    )
    sig_reindexed = _calc_vn30_intraday_confluence(df_idx, inter_idx)
    assert sig_reindexed.loc[10] == SignalState.NONE
    assert sig_reindexed.loc[60] == SignalState.BUY


def test_composite_helpers_edge_cases():
    """Verify helper functions handle empty dataframes cleanly."""
    empty_df = pd.DataFrame()
    assert len(_calc_trend_consensus(empty_df)) == 0
    assert len(_calc_momentum_consensus(empty_df)) == 0
    assert len(_calc_master_ensemble(empty_df)) == 0
    assert len(_calc_master_ensemble_v2(empty_df)) == 0
    assert len(_calc_ma_consensus(empty_df)) == 0
    assert len(_calc_trend_momentum_align(pd.Series(dtype=str), pd.Series(dtype=str))) == 0
    assert len(_calc_breakout_volume_confirmed(empty_df)) == 0
    assert len(_calc_mean_reversion_confluence(empty_df)) == 0
    assert len(_calc_smc_trend_volume_confluence(empty_df, empty_df)) == 0
    assert len(_calc_triple_screen_trading_system(empty_df)) == 0
    assert len(_calc_squeeze_momentum_volume_surge(empty_df, empty_df)) == 0
    assert len(_calc_vn30_intraday_confluence(empty_df, empty_df)) == 0


def test_signals_package_export():
    """Verify generate_composite_signals is properly exported in signalx.signals."""
    import signalx.signals

    assert hasattr(signalx.signals, "generate_composite_signals")
    assert callable(signalx.signals.generate_composite_signals)
