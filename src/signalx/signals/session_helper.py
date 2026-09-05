from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionContext:
    session_id: pd.Series
    bar_in_session: pd.Series
    time_minutes: pd.Series
    is_morning_open: pd.Series
    is_afternoon_open: pd.Series
    is_pre_atc: pd.Series


def extract_session_context(df: pd.DataFrame) -> SessionContext:
    """Extract intraday session IDs and time windows from DataFrame with synthetic fallback."""
    time_col = None
    for candidate in ["date", "datetime", "timestamp", "time"]:
        matches = [c for c in df.columns if str(c).strip().lower() == candidate]
        if matches:
            time_col = matches[0]
            break

    idx = df.index
    n = len(df)

    if time_col is not None or isinstance(idx, pd.DatetimeIndex):
        raw_dt = df[time_col] if time_col is not None else idx
        dt_series = pd.Series(pd.to_datetime(raw_dt), index=idx)
        session_id = pd.Series(dt_series.dt.strftime("%Y-%m-%d"), index=idx)
        bar_in_session = pd.Series(df.groupby(session_id).cumcount(), index=idx)
        time_minutes = pd.Series(dt_series.dt.hour * 60 + dt_series.dt.minute, index=idx)

        is_morning_open = pd.Series(
            (time_minutes >= 8 * 60 + 45) & (time_minutes <= 9 * 60 + 30),
            index=idx,
        )
        is_afternoon_open = pd.Series(
            (time_minutes >= 13 * 60) & (time_minutes <= 13 * 60 + 30),
            index=idx,
        )
        is_pre_atc = pd.Series(
            (time_minutes >= 14 * 60) & (time_minutes <= 14 * 60 + 25),
            index=idx,
        )
    else:
        session_id = pd.Series(np.arange(n) // 50, index=idx)
        bar_in_session = pd.Series(np.arange(n) % 50, index=idx)
        time_minutes = pd.Series(540 + (np.arange(n) % 50) * 5, index=idx)
        is_morning_open = pd.Series(bar_in_session < 6, index=idx)
        is_afternoon_open = pd.Series((bar_in_session >= 30) & (bar_in_session < 36), index=idx)
        is_pre_atc = pd.Series((bar_in_session >= 44) & (bar_in_session < 49), index=idx)

    return SessionContext(
        session_id=session_id,
        bar_in_session=bar_in_session,
        time_minutes=time_minutes,
        is_morning_open=is_morning_open,
        is_afternoon_open=is_afternoon_open,
        is_pre_atc=is_pre_atc,
    )
