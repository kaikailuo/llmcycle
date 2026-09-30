from __future__ import annotations

import unittest

from src.providers.moonshot import MoonshotProvider


HTML = """
<div class="home-card">
  <h3 class="home-card-title">K2.6</h3>
  <div class="home-card-pricing">
    <div><span>Input</span><span>$0.95 / MTok</span></div>
    <div><span>Output</span><span>$4.00 / MTok</span></div>
    <div><span>Cache Hit</span><span>$0.16 / MTok</span></div>
  </div>
</div>
"""


def _model(model_id: str = "kimi-k2.6") -> dict:
    return {
        "provider": "Moonshot AI",
        "model_api_id": model_id,
        "model_region": "Global",
        "input": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "output": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "cache": [
            {
                "cache_read": [
                    {"context_min": 0, "context_max": None, "per_million": 99}
                ]
            }
        ],
        "batch": {
            "input": [
                {"context_min": 0, "context_max": None, "per_million": 7}
            ],
            "output": [
                {"context_min": 0, "context_max": None, "per_million": 8}
            ],
        },
    }


class MoonshotProviderTests(unittest.TestCase):
    def test_updates_input_output_and_cache_but_retains_batch(self) -> None:
        original = _model()

        result = MoonshotProvider().parse(HTML, [original])[0]

        self.assertEqual(0.95, result.data["input"][0]["per_million"])
        self.assertEqual(4.0, result.data["output"][0]["per_million"])
        self.assertEqual(
            0.16, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual(original["batch"], result.data["batch"])
        self.assertIn(
            "official batch price not found; retained existing values",
            result.warnings,
        )

    def test_missing_model_is_unchanged_and_warned(self) -> None:
        original = _model("missing-model")

        result = MoonshotProvider().parse(HTML, [original])[0]

        self.assertEqual(original, result.data)
        self.assertIn("official price not found", result.warnings)


if __name__ == "__main__":
    unittest.main()
