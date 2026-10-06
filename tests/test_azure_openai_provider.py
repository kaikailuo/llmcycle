from __future__ import annotations

import copy
import io
import json
import unittest
from unittest.mock import patch

from src.providers.azure_openai import AzureOpenAIProvider


def _model() -> dict:
    def rates() -> list[dict]:
        return [{"context_min": 0, "context_max": None, "per_million": 99}]

    return {
        "provider": "Azure OpenAI",
        "model_api_id": "gpt-4.1-mini",
        "model_region": "Global",
        "currency": "EUR",
        "input": rates(),
        "output": rates(),
        "cache": [{"cache_read": rates()}],
        "batch": {
            "input": rates(),
            "output": rates(),
        },
    }


def _meter(sku: str, price: float, unit: str = "1K") -> dict:
    return {
        "skuName": sku,
        "retailPrice": price,
        "unitOfMeasure": unit,
        "currencyCode": "USD",
        "type": "Consumption",
        "isPrimaryMeterRegion": True,
    }


METERS = [
    _meter("gpt 4.1 mini Inp glbl", 0.0004),
    _meter("gpt 4.1 mini Outp glbl", 0.0016),
    _meter("gpt 4.1 mini cached Inp glbl", 0.0001),
    _meter("gpt 4.1 mini Batch Inp glbl", 0.0002),
    _meter("gpt 4.1 mini Batch Outp glbl", 0.0008),
]


class AzureOpenAIProviderTests(unittest.TestCase):
    def test_matches_global_dimensions_and_converts_units(self) -> None:
        original = _model()
        untouched = copy.deepcopy(original)

        result = AzureOpenAIProvider().parse(METERS, [original])[0]

        self.assertEqual(untouched, original)
        self.assertEqual(0.4, result.data["input"][0]["per_million"])
        self.assertEqual(1.6, result.data["output"][0]["per_million"])
        self.assertEqual(
            0.1, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual(0.2, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual(0.8, result.data["batch"]["output"][0]["per_million"])
        self.assertEqual("USD", result.data["currency"])
        self.assertEqual([], result.warnings)

    def test_conflicting_or_missing_field_preserves_only_that_field(self) -> None:
        original = _model()
        meters = [
            *METERS[:2],
            _meter("gpt 4.1 mini Inp glbl", 0.0005),
        ]

        result = AzureOpenAIProvider().parse(meters, [original])[0]

        self.assertEqual(99, result.data["input"][0]["per_million"])
        self.assertEqual(1.6, result.data["output"][0]["per_million"])
        self.assertEqual(
            99, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual(99, result.data["batch"]["input"][0]["per_million"])
        self.assertIn(
            "multiple official prices found for input; retained existing values",
            result.warnings,
        )
        self.assertIn(
            "official price not found for cache.cache_read; retained existing values",
            result.warnings,
        )

    def test_excludes_regional_data_zone_and_fine_tuning_meters(self) -> None:
        source = [
            _meter("gpt 4.1 mini Inp Data Zone", 0.123),
            _meter("gpt-4.1-mini-ft input global", 0.456),
        ]

        result = AzureOpenAIProvider().parse(source, [_model()])[0]

        self.assertEqual(99, result.data["input"][0]["per_million"])
        self.assertEqual(
            ["official Global price not found; retained existing values"],
            result.warnings,
        )

    def test_fetch_follows_every_next_page_for_each_product(self) -> None:
        pages = [
            {
                "Items": [_meter("gpt 4.1 mini Inp glbl", 0.0004)],
                "NextPageLink": "https://prices.azure.com/api/retail/prices?$skip=1000",
            },
            {
                "Items": [_meter("gpt 4.1 mini Outp glbl", 0.0016)],
                "NextPageLink": None,
            },
            {
                "Items": [_meter("GPT 5 Mini Inpt Glbl", 0.25, "1M")],
                "NextPageLink": "",
            },
        ]
        responses = [io.BytesIO(json.dumps(page).encode()) for page in pages]

        with patch(
            "src.providers.azure_openai.urlopen", side_effect=responses
        ) as mocked:
            items = AzureOpenAIProvider().fetch(
                "https://prices.azure.com/api/retail/prices"
            )

        self.assertEqual(3, len(items))
        self.assertEqual(3, mocked.call_count)
        second_url = mocked.call_args_list[1].args[0].full_url
        self.assertIn("$skip=1000", second_url)

    def test_fetch_retries_only_the_failed_page(self) -> None:
        page_1 = {
            "Items": [_meter("gpt 4.1 mini Inp glbl", 0.0004)],
            "NextPageLink": "https://prices.azure.com/api/retail/prices?$skip=1000",
        }
        page_2 = {
            "Items": [_meter("gpt 4.1 mini Outp glbl", 0.0016)],
            "NextPageLink": None,
        }
        page_3 = {
            "Items": [_meter("GPT 5 Mini Inpt Glbl", 0.25, "1M")],
            "NextPageLink": None,
        }
        responses = [
            io.BytesIO(json.dumps(page_1).encode()),
            TimeoutError("page 2 timed out"),
            io.BytesIO(json.dumps(page_2).encode()),
            io.BytesIO(json.dumps(page_3).encode()),
        ]

        with patch(
            "src.providers.azure_openai.urlopen", side_effect=responses
        ) as mocked:
            items = AzureOpenAIProvider().fetch(
                "https://prices.azure.com/api/retail/prices"
            )

        self.assertEqual(3, len(items))
        self.assertEqual(4, mocked.call_count)
        requested_urls = [item.args[0].full_url for item in mocked.call_args_list]
        self.assertNotEqual(requested_urls[0], requested_urls[1])
        self.assertEqual(requested_urls[1], requested_urls[2])
        self.assertNotEqual(requested_urls[0], requested_urls[3])


if __name__ == "__main__":
    unittest.main()
