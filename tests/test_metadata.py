from __future__ import annotations

import dataclasses

import pytest

from signalx.metadata import (
    SIGNAL_CATALOG,
    VALID_CATEGORIES,
    SignalMetadata,
    get_signal_metadata,
    get_signals_by_category,
    list_categories,
    register_signal,
)


def test_valid_categories_set():
    expected = {
        "trend",
        "momentum",
        "volatility",
        "volume",
        "candlestick",
        "statistical",
        "composite",
    }
    assert VALID_CATEGORIES == expected


def test_signal_metadata_frozen():
    meta = SignalMetadata(
        name="test_dummy_signal",
        category="trend",
        description="A test signal",
        library="signalx_native",
        buy_trigger="Bullish cross",
        sell_trigger="Bearish cross",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.name = "new_name_signal"  # type: ignore[misc]


def test_catalog_richness_and_validity():
    assert len(SIGNAL_CATALOG) >= 80, f"Expected >= 80 signals, found {len(SIGNAL_CATALOG)}"

    for name, meta in SIGNAL_CATALOG.items():
        # Name convention rule
        assert name.endswith("_signal"), f"Signal {name} does not end with '_signal'"
        assert isinstance(meta, SignalMetadata)
        assert meta.name == name

        # Valid category
        assert meta.category in VALID_CATEGORIES, (
            f"Signal {name} has invalid category {meta.category}"
        )

        # Non-empty fields
        assert isinstance(meta.description, str) and meta.description.strip(), (
            f"Empty description for {name}"
        )
        assert isinstance(meta.library, str) and meta.library.strip(), f"Empty library for {name}"
        assert isinstance(meta.buy_trigger, str) and meta.buy_trigger.strip(), (
            f"Empty buy_trigger for {name}"
        )
        assert isinstance(meta.sell_trigger, str) and meta.sell_trigger.strip(), (
            f"Empty sell_trigger for {name}"
        )


def test_list_categories():
    cats = list_categories()
    assert isinstance(cats, list)
    assert set(cats) == VALID_CATEGORIES
    assert cats == sorted(list(VALID_CATEGORIES))


def test_get_signals_by_category():
    for cat in VALID_CATEGORIES:
        signals = get_signals_by_category(cat)
        assert isinstance(signals, list)
        assert len(signals) > 0, f"Category {cat} should have registered signals"
        for s in signals:
            assert s.category == cat

    # Category counts validation
    assert len(get_signals_by_category("trend")) >= 20
    assert len(get_signals_by_category("momentum")) >= 15
    assert len(get_signals_by_category("volatility")) >= 10
    assert len(get_signals_by_category("volume")) >= 10
    assert len(get_signals_by_category("candlestick")) >= 10
    assert len(get_signals_by_category("statistical")) >= 8
    assert len(get_signals_by_category("composite")) >= 3

    # Unknown category returns empty list
    assert get_signals_by_category("non_existent_category") == []


def test_get_signal_metadata():
    meta = get_signal_metadata("trend_sma_cross_5_20_signal")
    assert meta is not None
    assert meta.name == "trend_sma_cross_5_20_signal"
    assert meta.category == "trend"
    assert "SMA" in meta.description

    assert get_signal_metadata("non_existent_signal") is None


def test_register_signal_success():
    custom_name = "trend_custom_test_mock_signal"
    register_signal(
        name=custom_name,
        category="trend",
        description="Custom registered test signal",
        library="signalx_native",
        buy_trigger="Custom buy rule",
        sell_trigger="Custom sell rule",
    )
    assert custom_name in SIGNAL_CATALOG
    registered = get_signal_metadata(custom_name)
    assert registered is not None
    assert registered.name == custom_name
    assert registered.category == "trend"


def test_register_signal_invalid_name():
    with pytest.raises(ValueError, match="must end with '_signal'"):
        register_signal(
            name="invalid_name_without_suffix",
            category="trend",
            description="Invalid",
            library="ta",
            buy_trigger="Buy",
            sell_trigger="Sell",
        )


def test_register_signal_invalid_category():
    with pytest.raises(ValueError, match="Invalid category"):
        register_signal(
            name="trend_invalid_cat_signal",
            category="invalid_category",
            description="Invalid cat",
            library="ta",
            buy_trigger="Buy",
            sell_trigger="Sell",
        )


@pytest.mark.parametrize(
    "field,kwargs",
    [
        (
            "description",
            {
                "name": "trend_test_1_signal",
                "category": "trend",
                "description": "  ",
                "library": "ta",
                "buy_trigger": "b",
                "sell_trigger": "s",
            },
        ),
        (
            "library",
            {
                "name": "trend_test_2_signal",
                "category": "trend",
                "description": "d",
                "library": "",
                "buy_trigger": "b",
                "sell_trigger": "s",
            },
        ),
        (
            "buy_trigger",
            {
                "name": "trend_test_3_signal",
                "category": "trend",
                "description": "d",
                "library": "ta",
                "buy_trigger": " ",
                "sell_trigger": "s",
            },
        ),
        (
            "sell_trigger",
            {
                "name": "trend_test_4_signal",
                "category": "trend",
                "description": "d",
                "library": "ta",
                "buy_trigger": "b",
                "sell_trigger": "\t",
            },
        ),
    ],
)
def test_register_signal_empty_fields(field: str, kwargs: dict[str, str]):
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        register_signal(**kwargs)
