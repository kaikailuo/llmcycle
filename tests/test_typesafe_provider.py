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
    def test_parses_official_models_document_price(self) -> None:
        source = r"""
        ## Current models

        | Jev 1.13 | `jev-1.13.0` |
        | :- | :- |
        | Price (per Btok / per Mtok) | \$42 / \$0.042 |

        * **Price:** Charged per input token. Output tokens are free.
        """
        original = _model()
        untouched = copy.deepcopy(original)

        result = TypesafeProvider().parse(source, [original])[0]

        self.assertEqual(untouched, original)
        self.assertEqual(0.042, result.data["input"][0]["per_million"])
        self.assertEqual(0.0, result.data["output"][0]["per_million"])
        self.assertEqual([], result.warnings)

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

    def test_missing_input_price_preserves_input_and_warns(self) -> None:
        source = "Jev 1.13. Charged per input token. Output tokens are free."

        result = TypesafeProvider().parse(source, [_model()])[0]

        self.assertEqual(9, result.data["input"][0]["per_million"])
        self.assertEqual(0.0, result.data["output"][0]["per_million"])
        self.assertEqual(
            ["official price not found for input; retained existing values"],
            result.warnings,
        )

    def test_missing_output_statement_preserves_output_and_warns(self) -> None:
        source = """
        Jev 1.13 | jev-1.13.0
        Price (per Btok / per Mtok)
        $42 / $0.042
        Charged per input token.
        """

        result = TypesafeProvider().parse(source, [_model()])[0]

        self.assertEqual(0.042, result.data["input"][0]["per_million"])
        self.assertEqual(9, result.data["output"][0]["per_million"])
        self.assertEqual(
            ["official price not found for output; retained existing values"],
            result.warnings,
        )

    def test_output_free_wordings(self) -> None:
        for statement in (
            "Output tokens: FREE",
            "Output tokens FREE",
            "Output tokens are free",
            "Output tokens are currently free of charge",
        ):
            with self.subTest(statement=statement):
                source = f"Jev costs $42 per billion input tokens. {statement}."

                result = TypesafeProvider().parse(source, [_model()])[0]

                self.assertEqual(0.0, result.data["output"][0]["per_million"])
                self.assertEqual([], result.warnings)

    def test_non_price_fields_remain_unchanged(self) -> None:
        original = _model()
        original["type"] = "judgment"
        source = "Jev costs $42 per billion input tokens. Output tokens are free."

        result = TypesafeProvider().parse(source, [original])[0]

        for field in (
            "provider",
            "model_api_id",
            "model_region",
            "model_uid",
            "name",
            "type",
        ):
            self.assertEqual(original[field], result.data[field])

    def test_unrecognizable_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Typesafe Jev pricing is empty"):
            TypesafeProvider().parse("Pricing coming soon", [_model()])


if __name__ == "__main__":
    unittest.main()
