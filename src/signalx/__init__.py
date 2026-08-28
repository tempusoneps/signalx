from __future__ import annotations

from signalx.constants import ALL_SIGNAL_STATES, SignalState
from signalx.core import generate_signals
from signalx.metadata import (
    SIGNAL_CATALOG,
    SIGNAL_CODE_CATALOG,
    get_code_to_name_map,
    get_name_to_code_map,
    get_signal_by_code,
    get_signal_by_name,
    get_signal_metadata,
    to_code_names,
    to_semantic_names,
)

__version__ = "0.1.0"
__all__ = [
    "ALL_SIGNAL_STATES",
    "SIGNAL_CATALOG",
    "SIGNAL_CODE_CATALOG",
    "SignalState",
    "__version__",
    "generate_signals",
    "get_code_to_name_map",
    "get_name_to_code_map",
    "get_signal_by_code",
    "get_signal_by_name",
    "get_signal_metadata",
    "to_code_names",
    "to_semantic_names",
]
