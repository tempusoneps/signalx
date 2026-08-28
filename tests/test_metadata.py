from __future__ import annotations

import dataclasses
import re

import pandas as pd
import pytest

from signalx.metadata import (
    SIGNAL_CATALOG,
    SIGNAL_CODE_CATALOG,
    VALID_CATEGORIES,
    SignalMetadata,
    get_code_to_name_map,
    get_name_to_code_map,
    get_signal_by_code,
    get_signal_by_name,
    get_signal_metadata,
    get_signals_by_category,
    list_categories,
    register_signal,
    to_code_names,
    to_semantic_names,
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


def test_signal_metadata_attributes_and_frozen():
    meta = SignalMetadata(
        code="TRD999_signal",
        name="trend_dummy_signal",
        category="trend",
        description="A test signal",
        library="signalx_native",
        buy_trigger="Bullish cross",
        sell_trigger="Bearish cross",
    )
    assert meta.code == "TRD999_signal"
    assert meta.name == "trend_dummy_signal"
    assert meta.category == "trend"
    assert meta.description == "A test signal"
    assert meta.library == "signalx_native"
    assert meta.buy_trigger == "Bullish cross"
    assert meta.sell_trigger == "Bearish cross"

    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.name = "new_name_signal"  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.code = "TRD000_signal"  # type: ignore[misc]


def test_catalog_richness_and_validity():
    assert len(SIGNAL_CATALOG) == 172, f"Expected 172 signals, found {len(SIGNAL_CATALOG)}"
    assert len(SIGNAL_CODE_CATALOG) == 172, (
        f"Expected 172 signals in SIGNAL_CODE_CATALOG, found {len(SIGNAL_CODE_CATALOG)}"
    )

    code_pattern = re.compile(r"^[A-Z]{3}\d{3}_signal$")
    all_codes = set()

    for name, meta in SIGNAL_CATALOG.items():
        # Name convention rule
        assert name.endswith("_signal"), f"Signal {name} does not end with '_signal'"
        assert isinstance(meta, SignalMetadata)
        assert meta.name == name

        # Code convention rule
        assert code_pattern.match(meta.code), f"Signal code {meta.code} does not match pattern"
        assert meta.code not in all_codes, f"Duplicate code detected: {meta.code}"
        all_codes.add(meta.code)

        # Code catalog consistency
        assert meta.code in SIGNAL_CODE_CATALOG
        assert SIGNAL_CODE_CATALOG[meta.code] is meta

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


def test_deterministic_category_codes():
    expected_category_prefixes = {
        "trend": ("TRD", 41),
        "momentum": ("MOM", 30),
        "volatility": ("VOL", 32),
        "volume": ("VLM", 18),
        "candlestick": ("CDL", 27),
        "statistical": ("STA", 16),
        "composite": ("CMP", 8),
    }

    for cat, (prefix, expected_count) in expected_category_prefixes.items():
        signals = get_signals_by_category(cat)
        assert len(signals) == expected_count, (
            f"Category {cat} expected {expected_count} signals, got {len(signals)}"
        )
        for i, meta in enumerate(signals, start=1):
            expected_code = f"{prefix}{i:03d}_signal"
            assert meta.code == expected_code, (
                f"Expected {expected_code} for {meta.name}, got {meta.code}"
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

    # Unknown category returns empty list
    assert get_signals_by_category("non_existent_category") == []


def test_get_signal_by_code():
    meta = get_signal_by_code("TRD001_signal")
    assert meta is not None
    assert meta.code == "TRD001_signal"
    assert meta.name == "trend_sma_cross_5_20_signal"
    assert meta.category == "trend"

    assert get_signal_by_code("NONEXISTENT_signal") is None
    assert get_signal_by_code("") is None


def test_get_signal_by_name():
    meta = get_signal_by_name("trend_sma_cross_5_20_signal")
    assert meta is not None
    assert meta.name == "trend_sma_cross_5_20_signal"
    assert meta.code == "TRD001_signal"
    assert meta.category == "trend"

    assert get_signal_by_name("non_existent_signal") is None


def test_get_signal_metadata_lookup_both_code_and_name():
    # Lookup by code
    meta_by_code = get_signal_metadata("TRD001_signal")
    assert meta_by_code is not None
    assert meta_by_code.code == "TRD001_signal"
    assert meta_by_code.name == "trend_sma_cross_5_20_signal"

    # Lookup by name
    meta_by_name = get_signal_metadata("trend_sma_cross_5_20_signal")
    assert meta_by_name is not None
    assert meta_by_name.code == "TRD001_signal"
    assert meta_by_name.name == "trend_sma_cross_5_20_signal"

    # Should refer to same instance
    assert meta_by_code is meta_by_name

    # Nonexistent
    assert get_signal_metadata("non_existent_signal") is None


def test_bidirectional_mappings():
    code_to_name = get_code_to_name_map()
    name_to_code = get_name_to_code_map()

    assert len(code_to_name) == 172
    assert len(name_to_code) == 172

    assert code_to_name["TRD001_signal"] == "trend_sma_cross_5_20_signal"
    assert name_to_code["trend_sma_cross_5_20_signal"] == "TRD001_signal"

    for code, name in code_to_name.items():
        assert name_to_code[name] == code

    for name, code in name_to_code.items():
        assert code_to_name[code] == name


def test_to_code_names_and_to_semantic_names():
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1200.0],
            "date": ["2026-01-01", "2026-01-02"],
            "trend_sma_cross_5_20_signal": ["buy", "hold"],
            "mom_rsi_ob_os_14_signal": ["none", "sell"],
            "custom_feature": [1, 2],
        }
    )

    coded_df = to_code_names(df)
    expected_coded_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "date",
        "TRD001_signal",
        "MOM001_signal",
        "custom_feature",
    ]
    assert list(coded_df.columns) == expected_coded_cols
    assert (coded_df["TRD001_signal"] == df["trend_sma_cross_5_20_signal"]).all()
    assert (coded_df["MOM001_signal"] == df["mom_rsi_ob_os_14_signal"]).all()

    # Converting back to semantic
    semantic_df = to_semantic_names(coded_df)
    assert list(semantic_df.columns) == list(df.columns)
    assert (semantic_df["trend_sma_cross_5_20_signal"] == df["trend_sma_cross_5_20_signal"]).all()
    assert (semantic_df["mom_rsi_ob_os_14_signal"] == df["mom_rsi_ob_os_14_signal"]).all()


def test_to_code_names_idempotent_and_passthrough():
    df = pd.DataFrame(
        {
            "open": [100.0],
            "TRD001_signal": ["buy"],
            "already_coded_signal": ["none"],
        }
    )
    # Applying to_code_names on already coded DataFrame should pass through untouched
    coded_df = to_code_names(df)
    assert list(coded_df.columns) == ["open", "TRD001_signal", "already_coded_signal"]


def test_register_signal_success():
    custom_code = "TRD999_signal"
    custom_name = "trend_custom_test_mock_signal"
    register_signal(
        code=custom_code,
        name=custom_name,
        category="trend",
        description="Custom registered test signal",
        library="signalx_native",
        buy_trigger="Custom buy rule",
        sell_trigger="Custom sell rule",
    )
    assert custom_name in SIGNAL_CATALOG
    assert custom_code in SIGNAL_CODE_CATALOG
    registered = get_signal_metadata(custom_name)
    assert registered is not None
    assert registered.code == custom_code
    assert registered.name == custom_name
    assert registered.category == "trend"

    # Also retrievable by code
    assert get_signal_by_code(custom_code) is registered
    assert get_signal_metadata(custom_code) is registered


def test_register_signal_invalid_code():
    with pytest.raises(ValueError, match="Signal code .* must end with '_signal'"):
        register_signal(
            code="invalid_code_no_suffix",
            name="trend_valid_name_signal",
            category="trend",
            description="Valid",
            library="ta",
            buy_trigger="Buy",
            sell_trigger="Sell",
        )


def test_register_signal_invalid_name():
    with pytest.raises(ValueError, match="Signal name .* must end with '_signal'"):
        register_signal(
            code="TRD998_signal",
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
            code="TRD997_signal",
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
            "code",
            {
                "code": "  ",
                "name": "trend_test_0_signal",
                "category": "trend",
                "description": "d",
                "library": "ta",
                "buy_trigger": "b",
                "sell_trigger": "s",
            },
        ),
        (
            "description",
            {
                "code": "TRD996_signal",
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
                "code": "TRD995_signal",
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
                "code": "TRD994_signal",
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
                "code": "TRD993_signal",
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
    with pytest.raises(ValueError, match=f"{field} cannot be empty|must end with '_signal'"):
        register_signal(**kwargs)
