from __future__ import annotations

from signalx.progress import GroupProgressBar


def test_group_progress_bar_context_manager():
    with GroupProgressBar("Trend Signals", total=30, enabled=False) as pbar:
        assert pbar.total == 30
        assert pbar.group_name == "Trend Signals"
        assert not pbar.enabled
        pbar.update(1)
        pbar.update(5)
        pbar.set_postfix_str("SMA")


def test_group_progress_bar_enabled():
    with GroupProgressBar("Momentum", total=10, enabled=True) as pbar:
        assert pbar.enabled
        pbar.update(2)
        pbar.set_postfix_str("RSI")
