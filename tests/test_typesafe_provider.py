from __future__ import annotations

import copy
import unittest

from src.providers.typesafe import TypesafeProvider


def _model() -> dict:
    return {
        "provider": "Typesafe AI",
        "model_api_id": "jev-latest",
        "model_region": "Global",
        "model_uid": "jev-uid",
        "name": "Jev",
        "input": [{"context_min": 0, "context_max": None, "per_million": 9}],
        "output": [{"context_min": 0, "context_max": None, "per_million": 9}],
    }


class TypesafeProviderTests(unittest.TestCase):
    def test_converts_billion_input_tokens_and_parses_free_output(self) -> None:
        source = """
        <main>
          <h1>Jev</h1>
          <h2>$42</h2><p>Per Billion input tokens.</p>
          <p>Output tokens: FREE (too cheap to meter).</p>
        </main>
        """
        original = _model()
        untouched = copy.deepcopy(original)

        result = TypesafeProvider().parse(source, [original])[0]

        self.assertEqual(untouched, original)
        self.assertEqual(0.042, result.data["input"][0]["per_million"])
        self.assertEqual(0.0, result.data["output"][0]["per_million"])
        self.assertEqual("jev-uid", result.data["model_uid"])
        self.assertEqual([], result.warnings)

    def test_missing_output_statement_preserves_output_and_warns(self) -> None:
        source = "Jev costs $42 per Billion input tokens."

        result = TypesafeProvider().parse(source, [_model()])[0]

        self.assertEqual(0.042, result.data["input"][0]["per_million"])
        self.assertEqual(9, result.data["output"][0]["per_million"])
        self.assertEqual(
            ["official price not found for output; retained existing values"],
            result.warnings,
        )

    def test_unrecognizable_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Typesafe Jev pricing is empty"):
            TypesafeProvider().parse("Pricing coming soon", [_model()])


if __name__ == "__main__":
    unittest.main()
