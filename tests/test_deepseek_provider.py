from __future__ import annotations

import copy
import unittest

from src.providers.deepseek import DeepSeekProvider


HTML = """
<table>
  <tr><td colspan="3">MODEL</td><td>deepseek-flash<sup>(1)</sup></td><td>deepseek-v4-pro</td></tr>
  <tr><td rowspan="6">PRICING</td><td rowspan="2">1M INPUT TOKENS<br>(CACHE HIT)</td><td>OFF-PEAK</td><td>$0.003</td><td>$0.022</td></tr>
  <tr><td>PEAK</td><td>$0.006</td><td>$0.044</td></tr>
  <tr><td rowspan="2">1M INPUT TOKENS<br>(CACHE MISS)</td><td>OFF-PEAK</td><td>$0.15</td><td>$0.66</td></tr>
  <tr><td>PEAK</td><td>$0.3</td><td>$1.32</td></tr>
  <tr><td rowspan="2">1M OUTPUT TOKENS</td><td>OFF-PEAK</td><td>$0.6</td><td>$1.98</td></tr>
  <tr><td>PEAK</td><td>$1.2</td><td>$3.96</td></tr>
</table>
"""


def _model(model_id: str) -> dict:
    return {
        "provider": "DeepSeek",
        "model_api_id": model_id,
        "model_region": "Global",
        "model_uid": f"uid-{model_id}",
        "name": model_id,
        "input": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "output": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "cache": [
            {
                "cache_read": [
                    {"context_min": 0, "context_max": None, "per_million": 99}
                ]
            }
        ],
    }


class DeepSeekProviderTests(unittest.TestCase):
    def test_uses_peak_prices_and_explicit_legacy_flash_aliases(self) -> None:
        originals = [
            _model("deepseek-v4-pro"),
            _model("deepseek-v4-flash"),
            _model("deepseek-v4-flash-vision-exp"),
        ]
        untouched = copy.deepcopy(originals)

        results = DeepSeekProvider().parse(HTML, originals)

        self.assertEqual(untouched, originals)
        pro, flash, vision = results
        self.assertEqual(1.32, pro.data["input"][0]["per_million"])
        self.assertEqual(3.96, pro.data["output"][0]["per_million"])
        self.assertEqual(0.044, pro.data["cache"][0]["cache_read"][0]["per_million"])
        for result in (flash, vision):
            self.assertEqual(0.3, result.data["input"][0]["per_million"])
            self.assertEqual(1.2, result.data["output"][0]["per_million"])
            self.assertEqual(
                0.006, result.data["cache"][0]["cache_read"][0]["per_million"]
            )
            self.assertEqual([], result.warnings)
            self.assertEqual(
                f"uid-{result.model_api_id}", result.data["model_uid"]
            )

    def test_missing_peak_field_preserves_only_that_field(self) -> None:
        source = HTML.replace(
            "<tr><td>PEAK</td><td>$1.2</td><td>$3.96</td></tr>",
            "<tr><td>PEAK</td><td>-</td><td>$3.96</td></tr>",
        )

        result = DeepSeekProvider().parse(source, [_model("deepseek-v4-flash")])[0]

        self.assertEqual(0.3, result.data["input"][0]["per_million"])
        self.assertEqual(99, result.data["output"][0]["per_million"])
        self.assertIn("official peak price not found for output", result.warnings[0])

    def test_empty_pricing_table_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "DeepSeek pricing table is empty"):
            DeepSeekProvider().parse("<html></html>", [_model("deepseek-v4-pro")])


if __name__ == "__main__":
    unittest.main()
