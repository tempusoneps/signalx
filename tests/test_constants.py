from enum import Enum

from signalx import ALL_SIGNAL_STATES, SignalState, __version__
from signalx.constants import (
    ALL_SIGNAL_STATES as CONST_ALL_STATES,
)
from signalx.constants import (
    SignalState as ConstSignalState,
)


def test_version():
    assert __version__ == "0.1.0"


def test_signal_state_values():
    assert SignalState.BUY == "buy"
    assert SignalState.SELL == "sell"
    assert SignalState.HOLD == "hold"
    assert SignalState.NONE == "none"
    assert SignalState.BUY.value == "buy"
    assert SignalState.SELL.value == "sell"
    assert SignalState.HOLD.value == "hold"
    assert SignalState.NONE.value == "none"


def test_signal_state_is_str_enum():
    assert issubclass(SignalState, (str, Enum))
    assert isinstance(SignalState.BUY, str)
    assert isinstance(SignalState.SELL, str)
    assert isinstance(SignalState.HOLD, str)
    assert isinstance(SignalState.NONE, str)


def test_all_signal_states():
    expected = frozenset({"buy", "sell", "hold", "none"})
    assert ALL_SIGNAL_STATES == expected
    assert isinstance(ALL_SIGNAL_STATES, frozenset)
    assert CONST_ALL_STATES == expected


def test_signal_state_exports_identical():
    assert SignalState is ConstSignalState
    assert ALL_SIGNAL_STATES is CONST_ALL_STATES


def test_signal_state_membership():
    for state in SignalState:
        assert state.value in ALL_SIGNAL_STATES
        assert state in ALL_SIGNAL_STATES
