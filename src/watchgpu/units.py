from __future__ import annotations

import re
from decimal import ROUND_CEILING, Decimal, InvalidOperation


class CapacityParseError(ValueError):
    """Raised when a memory capacity cannot be normalized to MiB."""


_CAPACITY_PATTERN = re.compile(
    r"^(?P<number>(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>m|mb|mib|g|gb|gib)?$",
    re.IGNORECASE,
)


def parse_capacity_mib(value: object) -> int:
    """Normalize a positive capacity value to a whole number of MiB.

    Bare integers are MiB. For operator convenience, ``G`` and ``GB`` follow
    the GPU tooling convention used by this project and are treated as GiB.
    Fractional values round up so a requested safety margin is never reduced.
    """

    if isinstance(value, bool):
        raise CapacityParseError("capacity must be a positive number")

    if isinstance(value, int):
        if value <= 0:
            raise CapacityParseError("capacity must be greater than zero")
        return value

    if not isinstance(value, str):
        raise CapacityParseError("capacity must be an integer or a string")

    match = _CAPACITY_PATTERN.fullmatch(value.strip())
    if match is None:
        raise CapacityParseError(f"invalid capacity: {value!r}")

    try:
        number = Decimal(match.group("number"))
    except InvalidOperation as exc:
        raise CapacityParseError(f"invalid capacity: {value!r}") from exc

    if number <= 0:
        raise CapacityParseError("capacity must be greater than zero")

    unit = (match.group("unit") or "mib").lower()
    multiplier = Decimal(1024 if unit in {"g", "gb", "gib"} else 1)
    return int((number * multiplier).to_integral_value(rounding=ROUND_CEILING))


def parse_user_capacity_mib(value: object) -> int:
    """Parse an interactive capacity, treating a bare string as GiB.

    Internal integers and persisted TOML values remain normalized MiB.  This
    wrapper is for CLI/console text such as ``2`` or ``2.5`` where GiB is the
    convenient operator-facing default.
    """

    if isinstance(value, str):
        match = _CAPACITY_PATTERN.fullmatch(value.strip())
        if match is not None and match.group("unit") is None:
            return parse_capacity_mib(f"{match.group('number')}GiB")
    return parse_capacity_mib(value)
