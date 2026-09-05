from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from signalx.signals.session_helper import SessionContext, extract_session_context


def test_extract_session_context_with_datetime_column():
    dates = pd.date_range("2026-01-05 09:00", periods=55, freq="5min")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.ones(55),
            "high": np.ones(55),
            "low": np.ones(55),
            "close": np.ones(55),
            "volume": np.ones(55),
        }
    )
    ctx = extract_session_context(df)
    assert isinstance(ctx, SessionContext)
    assert len(ctx.session_id) == 55
    assert ctx.bar_in_session.iloc[0] == 0
    assert ctx.bar_in_session.iloc[5] == 5
    assert ctx.is_morning_open.iloc[0] is True or ctx.is_morning_open.iloc[0] == 1
    assert bool(ctx.is_morning_open.iloc[0]) is True
    assert bool(ctx.is_morning_open.iloc[10]) is False


def test_extract_session_context_synthetic_fallback():
    df = pd.DataFrame(
        {
            "open": np.ones(120),
            "high": np.ones(120),
            "low": np.ones(120),
            "close": np.ones(120),
            "volume": np.ones(120),
        }
    )
    ctx = extract_session_context(df)
    assert isinstance(ctx, SessionContext)
    assert len(ctx.session_id) == 120
    assert ctx.session_id.iloc[0] == 0
    assert ctx.session_id.iloc[50] == 1
    assert ctx.bar_in_session.iloc[0] == 0
    assert ctx.bar_in_session.iloc[50] == 0
    assert bool(ctx.is_morning_open.iloc[0]) is True
    assert bool(ctx.is_morning_open.iloc[10]) is False
    assert bool(ctx.is_afternoon_open.iloc[30]) is True
    assert bool(ctx.is_afternoon_open.iloc[35]) is True
    assert bool(ctx.is_afternoon_open.iloc[36]) is False
    assert bool(ctx.is_pre_atc.iloc[44]) is True
    assert bool(ctx.is_pre_atc.iloc[48]) is True
    assert bool(ctx.is_pre_atc.iloc[49]) is False


def test_extract_session_context_with_datetime_index_and_windows():
    dates = pd.to_datetime(
        [
            "2026-01-05 09:05:00",
            "2026-01-05 09:30:00",
            "2026-01-05 10:00:00",
            "2026-01-05 13:15:00",
            "2026-01-05 14:10:00",
            "2026-01-05 14:26:00",
            "2026-01-06 09:00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "open": np.ones(len(dates)),
            "high": np.ones(len(dates)),
            "low": np.ones(len(dates)),
            "close": np.ones(len(dates)),
            "volume": np.ones(len(dates)),
        },
        index=dates,
    )
    ctx = extract_session_context(df)
    assert isinstance(ctx, SessionContext)
    assert ctx.session_id.iloc[0] == "2026-01-05"
    assert ctx.session_id.iloc[5] == "2026-01-05"
    assert ctx.session_id.iloc[6] == "2026-01-06"
    assert ctx.bar_in_session.iloc[0] == 0
    assert ctx.bar_in_session.iloc[5] == 5
    assert ctx.bar_in_session.iloc[6] == 0

    assert bool(ctx.is_morning_open.iloc[0]) is True
    assert bool(ctx.is_morning_open.iloc[1]) is True
    assert bool(ctx.is_morning_open.iloc[2]) is False

    assert bool(ctx.is_afternoon_open.iloc[3]) is True
    assert bool(ctx.is_afternoon_open.iloc[2]) is False

    assert bool(ctx.is_pre_atc.iloc[4]) is True
    assert bool(ctx.is_pre_atc.iloc[5]) is False


def test_extract_session_context_column_aliases():
    for col in ["timestamp", "DATETIME", "Time", " Date "]:
        df = pd.DataFrame(
            {
                col: pd.date_range("2026-01-05 09:00", periods=5, freq="5min"),
                "close": np.ones(5),
            }
        )
        ctx = extract_session_context(df)
        assert len(ctx.session_id) == 5
        assert ctx.session_id.iloc[0] == "2026-01-05"


def test_session_context_frozen_immutability():
    df = pd.DataFrame({"close": np.ones(10)})
    ctx = extract_session_context(df)
    with pytest.raises(FrozenInstanceError):
        ctx.session_id = pd.Series([1, 2, 3])
