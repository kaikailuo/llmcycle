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


def _meter(
    sku: str,
    price: float,
    unit: str = "1K",
    **overrides: object,
) -> dict:
    meter = {
        "skuName": sku,
        "retailPrice": price,
        "unitOfMeasure": unit,
        "currencyCode": "USD",
        "type": "Consumption",
        "isPrimaryMeterRegion": True,
    }
    meter.update(overrides)
    return meter


def _tiered_model(model_id: str = "gpt-5.6-sol") -> dict:
    def rates() -> list[dict]:
        return [
            {"context_min": 0, "context_max": 272000, "per_million": 99},
            {"context_min": 272000, "context_max": None, "per_million": 99},
        ]

    return {
        "provider": "Azure OpenAI",
        "model_api_id": model_id,
        "model_region": "Global",
        "currency": "EUR",
        "input": rates(),
        "output": rates(),
        "cache": [
            {
                "cache_read": rates(),
                "cache_writes": rates(),
            }
        ],
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

    def test_gpt_4o_uses_only_1120_snapshot_for_every_billing_type(self) -> None:
        model = _model()
        model["model_api_id"] = "gpt-4o"
        meters = [
            _meter("gpt 4o 1120 Inp glbl", 0.0025),
            _meter("gpt 4o 1120 Outp glbl", 0.01),
            _meter("gpt 4o 1120 cached Inp glbl", 0.00125),
            _meter("gpt 4o 1120 Batch Inp glbl", 0.00125),
            _meter("gpt 4o 1120 Batch Outp glbl", 0.005),
            _meter("gpt 4o 0513 Inp glbl", 0.005),
            _meter("gpt 4o 0513 Outp glbl", 0.015),
            _meter("gpt 4o 0513 cached Inp glbl", 0.003),
            _meter("gpt 4o 0513 Batch Inp glbl", 0.0025),
            _meter("gpt 4o 0513 Batch Outp glbl", 0.0075),
        ]

        result = AzureOpenAIProvider().parse(meters, [model])[0]

        self.assertEqual(2.5, result.data["input"][0]["per_million"])
        self.assertEqual(10.0, result.data["output"][0]["per_million"])
        self.assertEqual(
            1.25, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual(1.25, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual(5.0, result.data["batch"]["output"][0]["per_million"])
        self.assertEqual([], result.warnings)

    def test_duplicate_global_prices_across_azure_regions_are_deduplicated(self) -> None:
        model = {
            "provider": "Azure OpenAI",
            "model_api_id": "gpt-4o",
            "model_region": "Global",
            "currency": "EUR",
            "batch": {
                "input": [
                    {"context_min": 0, "context_max": None, "per_million": 99}
                ]
            },
        }
        meters = [
            _meter(
                "gpt 4o 1120 Batch Inp glbl",
                0.00125,
                armRegionName=azure_region,
            )
            for azure_region in ("swedencentral", "norwayeast", "eastus2")
        ]

        result = AzureOpenAIProvider().parse(meters, [model])[0]

        self.assertEqual(1.25, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual([], result.warnings)

    def test_short_and_long_context_update_input_output_and_cache_tiers(self) -> None:
        meters = [
            _meter("5.6 sol ShortCo Inp Std Gl", 4.0, "1M"),
            _meter("5.6 sol LongCo Inp Std Gl", 8.0, "1M"),
            _meter("5.6 sol ShortCo Opt Std Gl", 20.0, "1M"),
            _meter("5.6 sol LongCo Opt Std Gl", 30.0, "1M"),
            _meter("5.6 sol ShortCo Cd Inp Std Gl", 0.4, "1M"),
            _meter("5.6 sol LongCo Cd Inp Std Gl", 0.8, "1M"),
            _meter("5.6 sol ShortCo Cd Wr Std Gl", 5.0, "1M"),
            _meter("5.6 sol LongCo Cd Wr Std Gl", 10.0, "1M"),
        ]

        result = AzureOpenAIProvider().parse(meters, [_tiered_model()])[0]

        self.assertEqual([4.0, 8.0], [x["per_million"] for x in result.data["input"]])
        self.assertEqual(
            [20.0, 30.0], [x["per_million"] for x in result.data["output"]]
        )
        self.assertEqual(
            [0.4, 0.8],
            [x["per_million"] for x in result.data["cache"][0]["cache_read"]],
        )
        self.assertEqual(
            [5.0, 10.0],
            [x["per_million"] for x in result.data["cache"][0]["cache_writes"]],
        )
        self.assertEqual([], result.warnings)

    def test_unqualified_gpt_54_meter_is_short_context_fallback(self) -> None:
        model = _tiered_model("gpt-5.4")
        del model["cache"]
        meters = [
            _meter("5.4 inp Gl", 2.5, "1M"),
            _meter("5.4 longco inp Gl", 5.0, "1M"),
            _meter("5.4 opt Gl", 15.0, "1M"),
            _meter("5.4 longco opt Gl", 22.5, "1M"),
        ]

        result = AzureOpenAIProvider().parse(meters, [model])[0]

        self.assertEqual([2.5, 5.0], [x["per_million"] for x in result.data["input"]])
        self.assertEqual(
            [15.0, 22.5], [x["per_million"] for x in result.data["output"]]
        )
        self.assertEqual([], result.warnings)

    def test_only_standard_global_dimensions_are_collected(self) -> None:
        model = _tiered_model()
        del model["output"]
        del model["cache"]
        meters = [
            _meter("5.6 sol ShortCo Inp Std Gl", 4.0, "1M"),
            _meter("5.6 sol LongCo Inp Std Gl", 8.0, "1M"),
            _meter("5.6 sol ShortCo Inp Std DZ", 4.4, "1M"),
            _meter("5.6 sol ShortCo Inp PP Gl", 8.0, "1M"),
            _meter("5.6 sol ShortCo Inp Flex Gl", 3.0, "1M"),
            _meter("5.6 sol LongCo Inp Fl Gl", 6.0, "1M"),
            _meter("5.6 sol ShortCo Inp regional", 4.4, "1M"),
        ]

        result = AzureOpenAIProvider().parse(meters, [model])[0]

        self.assertEqual([4.0, 8.0], [x["per_million"] for x in result.data["input"]])
        self.assertEqual([], result.warnings)

    def test_conflict_in_same_complete_sku_identity_preserves_affected_tier(self) -> None:
        model = _tiered_model()
        del model["output"]
        del model["cache"]
        meters = [
            _meter(
                "5.6 sol ShortCo Inp Std Gl",
                4.0,
                "1M",
                effectiveStartDate="2026-09-01T00:00:00Z",
            ),
            _meter(
                "5.6 sol ShortCo Inp Std Gl",
                4.5,
                "1M",
                effectiveStartDate="2026-10-01T00:00:00Z",
            ),
            _meter("5.6 sol LongCo Inp Std Gl", 8.0, "1M"),
        ]

        result = AzureOpenAIProvider().parse(meters, [model])[0]

        self.assertEqual(99, result.data["input"][0]["per_million"])
        self.assertEqual(8.0, result.data["input"][1]["per_million"])
        self.assertIn(
            "multiple official prices found for input[0]; retained existing values",
            result.warnings,
        )

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
            {
                "Items": [_meter("6-sol ShortCo Inp Std Gl", 2.0, "1M")],
                "NextPageLink": None,
            },
        ]
        responses = [io.BytesIO(json.dumps(page).encode()) for page in pages]

        with patch(
            "src.providers.azure_openai.urlopen", side_effect=responses
        ) as mocked:
            items = AzureOpenAIProvider().fetch(
                "https://prices.azure.com/api/retail/prices"
            )

        self.assertEqual(4, len(items))
        self.assertEqual(4, mocked.call_count)
        second_url = mocked.call_args_list[1].args[0].full_url
        self.assertIn("$skip=1000", second_url)
        gpt6_url = mocked.call_args_list[3].args[0].full_url
        self.assertIn("Azure+OpenAI+GPT6", gpt6_url)

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
        page_4 = {
            "Items": [_meter("6-sol ShortCo Inp Std Gl", 2.0, "1M")],
            "NextPageLink": None,
        }
        responses = [
            io.BytesIO(json.dumps(page_1).encode()),
            TimeoutError("page 2 timed out"),
            io.BytesIO(json.dumps(page_2).encode()),
            io.BytesIO(json.dumps(page_3).encode()),
            io.BytesIO(json.dumps(page_4).encode()),
        ]

        with patch(
            "src.providers.azure_openai.urlopen", side_effect=responses
        ) as mocked:
            items = AzureOpenAIProvider().fetch(
                "https://prices.azure.com/api/retail/prices"
            )

        self.assertEqual(4, len(items))
        self.assertEqual(5, mocked.call_count)
        requested_urls = [item.args[0].full_url for item in mocked.call_args_list]
        self.assertNotEqual(requested_urls[0], requested_urls[1])
        self.assertEqual(requested_urls[1], requested_urls[2])
        self.assertNotEqual(requested_urls[0], requested_urls[3])
        self.assertIn("Azure+OpenAI+GPT6", requested_urls[4])


if __name__ == "__main__":
    unittest.main()
