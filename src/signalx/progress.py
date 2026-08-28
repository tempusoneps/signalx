from __future__ import annotations

import sys
from types import TracebackType

from tqdm.auto import tqdm


class GroupProgressBar:
    """A clean wrapper around tqdm to render per-group signal progress bars in terminal."""

    def __init__(
        self,
        group_name: str,
        total: int,
        unit: str = "sig",
        enabled: bool = True,
        miniters: int = 1,
    ) -> None:
        self.group_name = group_name
        self.total = total
        self.enabled = bool(enabled)
        # Pad group name for aligned terminal rendering
        padded_desc = f"{group_name:<20}"
        self._bar = tqdm(
            total=total,
            desc=padded_desc,
            unit=unit,
            disable=not self.enabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            miniters=miniters,
        )

    def __enter__(self) -> GroupProgressBar:
        self._bar.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._bar.__exit__(exc_type, exc, traceback)

    def update(self, n: int = 1) -> None:
        """Advance the progress bar by n steps."""
        if hasattr(self._bar, "update"):
            self._bar.update(n)

    def set_postfix_str(self, s: str) -> None:
        """Set a postfix string on the progress bar."""
        if hasattr(self._bar, "set_postfix_str"):
            self._bar.set_postfix_str(s)
