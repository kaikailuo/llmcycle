from __future__ import annotations

from decimal import Decimal
from typing import Any


# Stable project convention: 1 USD is approximately 6.8 CNY.  This is
# deliberately not a live exchange rate, so identical source prices produce
# identical candidates on every run.
CNY_PER_USD = Decimal("6.8")


def cny_to_usd(value: Any) -> Decimal:
    """Convert a CNY amount to USD using the project's fixed approximation."""

    return Decimal(str(value)) / CNY_PER_USD
