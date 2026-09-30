from __future__ import annotations

import unittest

from src.providers.minimax import MiniMaxProvider


MARKDOWN = r"""
## LLM

| Model | Input | Output | Prompt caching Read | Prompt caching Write |
| :- | :- | :- | :- | :- |
| **MiniMax-M2.7** | \$0.3 / M tokens | \$1.2 / M tokens | \$0.06 / M tokens | \$0.375 / M tokens |
| **MiniMax-M2.7-highspeed** | \$0.6 / M tokens | \$2.4 / M tokens | \$0.06 / M tokens | \$0.375 / M tokens |

<Accordion title="Legacy Models">
| Model | Input | Output | Prompt caching Read | Prompt caching Write |
| :- | :- | :- | :- | :- |
| **MiniMax-M2.5** | \$0.3 / M tokens | \$1.2 / M tokens | \$0.03 / M tokens | \$0.375 / M tokens |
| **MiniMax-M2.5-highspeed** | \$0.6 / M tokens | \$2.4 / M tokens | \$0.03 / M tokens | \$0.375 / M tokens |
</Accordion>

## Audio
"""


def _model(model_id: str) -> dict:
    return {
        "provider": "MiniMax",
        "model_api_id": model_id,
        "model_region": "Global",
        "input": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "output": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "cache": [
            {
                "cache_read": [
                    {"context_min": 0, "context_max": None, "per_million": 99}
                ],
                "cache_writes": [
                    {"context_min": 0, "context_max": None, "per_million": 99}
                ],
            }
        ],
    }


class MiniMaxProviderTests(unittest.TestCase):
    def test_updates_current_and_legacy_models_without_matching_highspeed(self) -> None:
        results = MiniMaxProvider().parse(
            MARKDOWN, [_model("MiniMax-M2.5"), _model("MiniMax-M2.7")]
        )

        m25, m27 = results
        self.assertEqual(0.3, m25.data["input"][0]["per_million"])
        self.assertEqual(1.2, m25.data["output"][0]["per_million"])
        self.assertEqual(0.03, m25.data["cache"][0]["cache_read"][0]["per_million"])
        self.assertEqual(0.375, m25.data["cache"][0]["cache_writes"][0]["per_million"])
        self.assertEqual(0.06, m27.data["cache"][0]["cache_read"][0]["per_million"])
        self.assertEqual([], m25.warnings)
        self.assertEqual([], m27.warnings)

    def test_missing_cache_write_preserves_only_that_field(self) -> None:
        source = MARKDOWN.replace(
            "| **MiniMax-M2.7** | \\$0.3 / M tokens | \\$1.2 / M tokens | \\$0.06 / M tokens | \\$0.375 / M tokens |",
            "| **MiniMax-M2.7** | \\$0.3 / M tokens | \\$1.2 / M tokens | \\$0.06 / M tokens | - |",
        )
        original = _model("MiniMax-M2.7")

        result = MiniMaxProvider().parse(source, [original])[0]

        self.assertEqual(0.3, result.data["input"][0]["per_million"])
        self.assertEqual(
            99, result.data["cache"][0]["cache_writes"][0]["per_million"]
        )
        self.assertIn(
            "official price not found for cache.cache_writes; retained existing values",
            result.warnings,
        )


if __name__ == "__main__":
    unittest.main()
