from __future__ import annotations

from enum import StrEnum


class SignalState(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NONE = "none"


ALL_SIGNAL_STATES: frozenset[str] = frozenset(
    {
        SignalState.BUY,
        SignalState.SELL,
        SignalState.HOLD,
        SignalState.NONE,
    }
)
