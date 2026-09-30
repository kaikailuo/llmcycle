from __future__ import annotations

import unittest

from src.providers.openai import OpenAIProvider


MARKDOWN = """
### Standard pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact-model | $1.00 | $0.10 | $1.25 | $5.00 | $2.00 | $0.20 | $2.50 | $7.50 |

### Batch pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact-model | $0.50 | $0.05 | $0.625 | $2.50 | - | - | - | - |

Short context: ≤272K input tokens. Long context: >272K input tokens.
"""


class OpenAIProviderTests(unittest.TestCase):
    def test_updates_existing_price_shape_only(self) -> None:
        original = {
            "provider": "OpenAI",
            "model_api_id": "exact-model",
            "model_region": "Global",
            "currency": "USD",
            "input": [
                {"context_min": 0, "context_max": 272000, "per_million": 99},
                {"context_min": 272000, "context_max": None, "per_million": 99},
            ],
            "output": [
                {"context_min": 0, "context_max": 272000, "per_million": 99},
                {"context_min": 272000, "context_max": None, "per_million": 99},
            ],
            "cache": [
                {
                    "cache_read": [
                        {"context_min": 0, "context_max": 272000, "per_million": 99},
                        {"context_min": 272000, "context_max": None, "per_million": 99},
                    ]
                }
            ],
            "batch": {
                "input": [{"context_min": 0, "context_max": None, "per_million": 99}],
                "output": [{"context_min": 0, "context_max": None, "per_million": 99}],
            },
        }

        result = OpenAIProvider().parse(MARKDOWN, [original])[0]

        self.assertEqual([1.0, 2.0], [item["per_million"] for item in result.data["input"]])
        self.assertEqual([5.0, 7.5], [item["per_million"] for item in result.data["output"]])
        self.assertEqual(
            [0.1, 0.2],
            [item["per_million"] for item in result.data["cache"][0]["cache_read"]],
        )
        self.assertEqual(0.5, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual("USD", result.data["currency"])
        self.assertEqual([], result.warnings)

    def test_missing_model_is_unchanged_and_warned(self) -> None:
        original = {
            "provider": "OpenAI",
            "model_api_id": "missing-model",
            "model_region": "Global",
            "input": [{"context_min": 0, "context_max": None, "per_million": 3.0}],
        }

        result = OpenAIProvider().parse(MARKDOWN, [original])[0]

        self.assertEqual(original, result.data)
        self.assertIn("official price not found", result.warnings)

    def test_unavailable_price_keeps_old_value(self) -> None:
        markdown = MARKDOWN.replace("$0.10 | $1.25", "- | $1.25", 1)
        original = {
            "provider": "OpenAI",
            "model_api_id": "exact-model",
            "model_region": "Global",
            "cache": [
                {
                    "cache_read": [
                        {"context_min": 0, "context_max": None, "per_million": 9.0}
                    ]
                }
            ],
        }

        result = OpenAIProvider().parse(markdown, [original])[0]

        self.assertEqual(9.0, result.data["cache"][0]["cache_read"][0]["per_million"])
        self.assertTrue(any("official price unavailable" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
