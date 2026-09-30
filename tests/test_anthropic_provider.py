from __future__ import annotations

import unittest
from datetime import date

from src.providers.anthropic import AnthropicProvider


MARKDOWN = """
## Model pricing

| Model | Base Input | Cache Write 5m | Cache Write 1h | Cache Hit | Output |
| --- | --- | --- | --- | --- | --- |
| Claude Haiku 4.5 | $1.00 / MTok | $1.25 / MTok | $2.00 / MTok | $0.10 / MTok | $5.00 / MTok |

Prompt caching multipliers stack with other pricing modifiers, including the Batch API discount.

### Batch processing

| Model | Batch Input | Batch Output |
| --- | --- | --- |
| Claude Haiku 4.5 | $0.50 / MTok | $2.50 / MTok |
"""


def _model(model_id: str = "claude-haiku-4-5") -> dict:
    return {
        "provider": "Anthropic",
        "model_api_id": model_id,
        "model_region": "Global",
        "currency": "USD",
        "input": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "output": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "cache": [
            {
                "cache_read": [
                    {"context_min": 0, "context_max": None, "per_million": 99}
                ],
                "cache_writes": [
                    {
                        "context_min": 0,
                        "context_max": None,
                        "5m_per_million": 99,
                        "1h_per_million": 99,
                    }
                ],
            }
        ],
        "batch": {
            "input": [
                {"context_min": 0, "context_max": None, "per_million": 99}
            ],
            "output": [
                {"context_min": 0, "context_max": None, "per_million": 99}
            ],
            "cache": [
                {
                    "cache_read": [
                        {"context_min": 0, "context_max": None, "per_million": 99}
                    ],
                    "cache_writes": [
                        {
                            "context_min": 0,
                            "context_max": None,
                            "5m_per_million": 99,
                            "1h_per_million": 99,
                        }
                    ],
                }
            ],
        },
    }


class AnthropicProviderTests(unittest.TestCase):
    def test_updates_base_cache_and_batch_prices(self) -> None:
        result = AnthropicProvider().parse(MARKDOWN, [_model()])[0]

        self.assertEqual(1.0, result.data["input"][0]["per_million"])
        self.assertEqual(5.0, result.data["output"][0]["per_million"])
        self.assertEqual(
            0.1, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        writes = result.data["cache"][0]["cache_writes"][0]
        self.assertEqual(1.25, writes["5m_per_million"])
        self.assertEqual(2.0, writes["1h_per_million"])
        self.assertEqual(0.5, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual(2.5, result.data["batch"]["output"][0]["per_million"])
        self.assertEqual(
            0.05,
            result.data["batch"]["cache"][0]["cache_read"][0]["per_million"],
        )
        self.assertEqual([], result.warnings)

    def test_selects_record_active_on_requested_date(self) -> None:
        markdown = """
## Model pricing

| Model | Base Input | Output | Effective Date |
| --- | --- | --- | --- |
| Claude Sonnet 5 | $3 | $15 | 2026-01-01 |
| Claude Sonnet 5 | $2 | $10 | 2026-09-30 |

### Batch processing

| Model | Batch Input | Batch Output |
| --- | --- | --- |
| Claude Sonnet 5 | $1 | $5 |
"""
        records = AnthropicProvider()._parse_table(markdown, "Model pricing")

        selected = AnthropicProvider()._select_active_record(
            records["Claude Sonnet 5"], date(2026, 9, 30)
        )

        self.assertIsNotNone(selected)
        self.assertEqual(2.0, float(selected.input))
        self.assertEqual(10.0, float(selected.output))

    def test_missing_model_is_unchanged_and_warned(self) -> None:
        original = _model("missing-model")

        result = AnthropicProvider().parse(MARKDOWN, [original])[0]

        self.assertEqual(original, result.data)
        self.assertIn("official price not found", result.warnings)


if __name__ == "__main__":
    unittest.main()
