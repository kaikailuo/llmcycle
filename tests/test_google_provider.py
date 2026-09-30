from __future__ import annotations

import unittest

from src.providers.google import GoogleProvider


HTML = """
<h2 id="single-model">Single Model</h2>
<h3>Standard</h3>
<table>
  <tr><th></th><th>Free Tier</th><th>Paid Tier</th></tr>
  <tr><td>Input price</td><td>Free</td><td>$1.50</td></tr>
  <tr><td>Output price (including thinking tokens)</td><td>Free</td><td>$9.00</td></tr>
  <tr><td>Context caching price</td><td>Free</td><td>$0.15<br>$1 storage</td></tr>
</table>
<h3>Batch</h3>
<table>
  <tr><th></th><th>Free Tier</th><th>Paid Tier</th></tr>
  <tr><td>Input price</td><td>Unavailable</td><td>$0.75</td></tr>
  <tr><td>Output price (including thinking tokens)</td><td>Unavailable</td><td>$4.50</td></tr>
  <tr><td>Context caching price</td><td>Unavailable</td><td>$0.075<br>$1 storage</td></tr>
</table>

<h2 id="tiered-model">Tiered Model</h2>
<h3>Standard</h3>
<table>
  <tr><th></th><th>Free Tier</th><th>Paid Tier</th></tr>
  <tr><td>Input price</td><td>Unavailable</td><td>$2.00, prompts &lt;= 200k tokens<br>$4.00, prompts &gt; 200k tokens</td></tr>
  <tr><td>Output price (including thinking tokens)</td><td>Unavailable</td><td>$12.00, prompts &lt;= 200k tokens<br>$18.00, prompts &gt; 200k</td></tr>
  <tr><td>Context caching price</td><td>Unavailable</td><td>$0.20, prompts &lt;= 200k tokens<br>$0.40, prompts &gt; 200k</td></tr>
</table>
<h3>Batch</h3>
<table>
  <tr><th></th><th>Free Tier</th><th>Paid Tier</th></tr>
  <tr><td>Input price</td><td>Unavailable</td><td>$1.00, prompts &lt;= 200k tokens<br>$2.00, prompts &gt; 200k tokens</td></tr>
  <tr><td>Output price (including thinking tokens)</td><td>Unavailable</td><td>$6.00, prompts &lt;= 200k tokens<br>$9.00, prompts &gt; 200k</td></tr>
  <tr><td>Context caching price</td><td>Unavailable</td><td>$0.20, prompts &lt;= 200k tokens<br>$0.40, prompts &gt; 200k</td></tr>
</table>
"""


def _rates(value: float = 99) -> list[dict]:
    return [{"context_min": 0, "context_max": None, "per_million": value}]


def _tiered_rates(value: float = 99) -> list[dict]:
    return [
        {"context_min": 0, "context_max": 200000, "per_million": value},
        {"context_min": 200000, "context_max": None, "per_million": value},
    ]


def _model(model_id: str, rates: list[dict]) -> dict:
    return {
        "provider": "Google",
        "model_api_id": model_id,
        "model_region": "Global",
        "input": [dict(rate) for rate in rates],
        "output": [dict(rate) for rate in rates],
        "cache": [{"cache_read": [dict(rate) for rate in rates]}],
        "batch": {
            "input": [dict(rate) for rate in rates],
            "output": [dict(rate) for rate in rates],
            "cache": [{"cache_read": [dict(rate) for rate in rates]}],
        },
    }


class GoogleProviderTests(unittest.TestCase):
    def test_updates_single_tier_standard_batch_and_cache(self) -> None:
        result = GoogleProvider().parse(HTML, [_model("single-model", _rates())])[0]

        self.assertEqual(1.5, result.data["input"][0]["per_million"])
        self.assertEqual(9.0, result.data["output"][0]["per_million"])
        self.assertEqual(
            0.15, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual(0.75, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual(4.5, result.data["batch"]["output"][0]["per_million"])
        self.assertEqual(
            0.075,
            result.data["batch"]["cache"][0]["cache_read"][0]["per_million"],
        )
        self.assertEqual([], result.warnings)

    def test_maps_200k_context_tiers(self) -> None:
        result = GoogleProvider().parse(
            HTML, [_model("tiered-model", _tiered_rates())]
        )[0]

        self.assertEqual([2.0, 4.0], [rate["per_million"] for rate in result.data["input"]])
        self.assertEqual(
            [12.0, 18.0], [rate["per_million"] for rate in result.data["output"]]
        )
        self.assertEqual(
            [0.2, 0.4],
            [rate["per_million"] for rate in result.data["cache"][0]["cache_read"]],
        )
        self.assertEqual(
            [1.0, 2.0],
            [rate["per_million"] for rate in result.data["batch"]["input"]],
        )

    def test_unmatched_context_tier_is_retained_and_warned(self) -> None:
        rates = [
            {"context_min": 0, "context_max": 128000, "per_million": 77},
            {"context_min": 128000, "context_max": None, "per_million": 88},
        ]
        original = _model("tiered-model", rates)

        result = GoogleProvider().parse(HTML, [original])[0]

        self.assertEqual(original, result.data)
        self.assertTrue(
            any("cannot map official context tier" in warning for warning in result.warnings)
        )


if __name__ == "__main__":
    unittest.main()
