from __future__ import annotations

import pytest

from watchgpu.units import CapacityParseError, parse_capacity_mib, parse_user_capacity_mib


@pytest.mark.parametrize(
    ("value", "expected_mib"),
    [
        (2048, 2048),
        ("2048MiB", 2048),
        ("2GiB", 2048),
        ("2G", 2048),
        ("1.5GiB", 1536),
        ("500M", 500),
    ],
)
def test_user_capacity_values_are_normalized_to_mib(
    value: int | str, expected_mib: int
) -> None:
    assert parse_capacity_mib(value) == expected_mib


@pytest.mark.parametrize("value", ["", "0", 0, -1, "-2GiB", "12watts", object()])
def test_invalid_capacity_values_are_rejected(value: object) -> None:
    with pytest.raises(CapacityParseError):
        parse_capacity_mib(value)


@pytest.mark.parametrize(
    ("value", "expected_mib"),
    [
        ("2", 2048),
        ("2.5", 2560),
        ("0.5", 512),
        ("2048MiB", 2048),
        ("2GiB", 2048),
    ],
)
def test_user_facing_bare_capacity_values_default_to_gib(
    value: str, expected_mib: int
) -> None:
    assert parse_user_capacity_mib(value) == expected_mib
