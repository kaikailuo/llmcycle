from __future__ import annotations

import unittest
from decimal import Decimal

from src.utils.currency import CNY_PER_USD, cny_to_usd


class CurrencyTests(unittest.TestCase):
    def test_uses_fixed_cny_per_usd_direction(self) -> None:
        self.assertEqual(Decimal("6.8"), CNY_PER_USD)
        self.assertEqual(Decimal("1"), cny_to_usd(Decimal("6.8")))


if __name__ == "__main__":
    unittest.main()
