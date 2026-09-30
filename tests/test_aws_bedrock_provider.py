from __future__ import annotations

import copy
import unittest

from src.providers.aws_bedrock import AWSBedrockProvider


MODEL_NAME = "Claude Haiku 4.5"
SOURCE_NAME = f"{MODEL_NAME} (Amazon Bedrock Edition)"


def _rates() -> list[dict]:
    return [{"context_min": 0, "context_max": None, "per_million": 99}]


def _model(region: str = "Global") -> dict:
    prefix = "global" if region == "Global" else ("jp" if region == "ap-northeast-1" else "us")
    return {
        "provider": "AWS Bedrock",
        "model_api_id": f"{prefix}.anthropic.claude-haiku-4-5-v1:0",
        "model_region": region,
        "name": MODEL_NAME,
        "currency": "EUR",
        "input": _rates(),
        "output": _rates(),
        "cache": [
            {
                "cache_read": _rates(),
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
        "batch": {"input": _rates(), "output": _rates()},
    }


def _entry(
    usage_type: str,
    price: str,
    region: str = "ap-northeast-1",
    *,
    unit: str = "1M tokens",
    name: str = SOURCE_NAME,
) -> dict:
    return {
        "usage_type": usage_type,
        "price": price,
        "region": region,
        "unit": unit,
        "name": name,
    }


def _offer(*entries: dict) -> dict:
    products = {}
    on_demand = {}
    for index, entry in enumerate(entries, 1):
        sku = f"SKU{index}"
        products[sku] = {
            "sku": sku,
            "productFamily": "Amazon Bedrock",
            "attributes": {
                "servicename": entry["name"],
                "regionCode": entry["region"],
                "usagetype": entry["usage_type"],
                "operation": "",
            },
        }
        on_demand[sku] = {
            f"{sku}.JRTCKXETXF": {
                "priceDimensions": {
                    f"{sku}.JRTCKXETXF.6YS6EN2CT7": {
                        "unit": entry["unit"],
                        "pricePerUnit": {"USD": entry["price"]},
                    }
                }
            }
        }
    return {"products": products, "terms": {"OnDemand": on_demand}}


def _global_base_entries() -> list[dict]:
    return [
        _entry("APN1-MP:APN1_InputTokenCount_Global-Units", "1.0"),
        _entry("APN1-MP:APN1_OutputTokenCount_Global-Units", "5.0"),
    ]


def _regional_base_entries(region: str = "ap-northeast-1") -> list[dict]:
    prefix = "APN1" if region == "ap-northeast-1" else "USW2"
    input_price = "1.1" if region == "ap-northeast-1" else "1.2"
    output_price = "5.5" if region == "ap-northeast-1" else "6.0"
    return [
        _entry(
            f"{prefix}-MP:{prefix}_InputTokenCount-Units", input_price, region
        ),
        _entry(
            f"{prefix}-MP:{prefix}_OutputTokenCount-Units", output_price, region
        ),
    ]


class AWSBedrockProviderTests(unittest.TestCase):
    def test_global_model_updates_input_and_output(self) -> None:
        source = _offer(*_global_base_entries(), *_regional_base_entries())

        result = AWSBedrockProvider().parse(source, [_model()])[0]

        self.assertEqual(1.0, result.data["input"][0]["per_million"])
        self.assertEqual(5.0, result.data["output"][0]["per_million"])
        self.assertEqual("USD", result.data["currency"])

    def test_regional_model_uses_matching_region(self) -> None:
        source = _offer(
            *_regional_base_entries("ap-northeast-1"),
            *_regional_base_entries("us-west-2"),
        )

        result = AWSBedrockProvider().parse(source, [_model("ap-northeast-1")])[0]

        self.assertEqual(1.1, result.data["input"][0]["per_million"])
        self.assertEqual(5.5, result.data["output"][0]["per_million"])

    def test_global_and_regional_models_do_not_cross_prices(self) -> None:
        source = _offer(*_global_base_entries(), *_regional_base_entries())

        global_result, regional_result = AWSBedrockProvider().parse(
            source, [_model(), _model("ap-northeast-1")]
        )

        self.assertEqual(1.0, global_result.data["input"][0]["per_million"])
        self.assertEqual(5.0, global_result.data["output"][0]["per_million"])
        self.assertEqual(1.1, regional_result.data["input"][0]["per_million"])
        self.assertEqual(5.5, regional_result.data["output"][0]["per_million"])

    def test_cache_fields_support_real_legacy_and_current_usage_types(self) -> None:
        source = _offer(
            _entry("APN1-MP:APN1_CacheReadInputTokenCount-Units", "0.11"),
            _entry("APN1-MP:APN1_cache_write_tokens_standard-Units", "1.375"),
            _entry("APN1-MP:APN1_cache_write_tokens_1h_standard-Units", "2.2"),
        )

        result = AWSBedrockProvider().parse(source, [_model("ap-northeast-1")])[0]

        cache = result.data["cache"][0]
        self.assertEqual(0.11, cache["cache_read"][0]["per_million"])
        self.assertEqual(1.375, cache["cache_writes"][0]["5m_per_million"])
        self.assertEqual(2.2, cache["cache_writes"][0]["1h_per_million"])

    def test_batch_fields_support_current_usage_types(self) -> None:
        source = _offer(
            _entry("APN1-MP:APN1_input_tokens_batch-Units", "0.55"),
            _entry("APN1-MP:APN1_output_tokens_batch-Units", "2.75"),
        )

        result = AWSBedrockProvider().parse(source, [_model("ap-northeast-1")])[0]

        self.assertEqual(0.55, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual(2.75, result.data["batch"]["output"][0]["per_million"])

    def test_missing_field_preserves_only_that_field_and_warns(self) -> None:
        source = _offer(*_regional_base_entries())

        result = AWSBedrockProvider().parse(source, [_model("ap-northeast-1")])[0]

        self.assertEqual(1.1, result.data["input"][0]["per_million"])
        self.assertEqual(5.5, result.data["output"][0]["per_million"])
        self.assertEqual(
            99,
            result.data["cache"][0]["cache_writes"][0]["1h_per_million"],
        )
        self.assertIn(
            "official price not found for cache.cache_writes.1h_per_million; "
            "retained existing value",
            result.warnings,
        )

    def test_conflicting_prices_preserve_field_and_warn(self) -> None:
        source = _offer(
            *_regional_base_entries(),
            _entry("APN1-MP:APN1_input_tokens_standard-Units", "1.2"),
        )

        result = AWSBedrockProvider().parse(source, [_model("ap-northeast-1")])[0]

        self.assertEqual(99, result.data["input"][0]["per_million"])
        self.assertEqual(5.5, result.data["output"][0]["per_million"])
        self.assertTrue(
            any(
                warning.startswith("multiple official prices matched for input")
                for warning in result.warnings
            )
        )

    def test_equal_prices_from_multiple_skus_are_accepted(self) -> None:
        source = _offer(
            _entry("APN1-MP:APN1_InputTokenCount-Units", "1.1"),
            _entry("APN1-MP:APN1_input_tokens_standard-Units", "1.1000000000"),
        )

        result = AWSBedrockProvider().parse(source, [_model("ap-northeast-1")])[0]

        self.assertEqual(1.1, result.data["input"][0]["per_million"])
        self.assertFalse(any("multiple" in warning for warning in result.warnings))

    def test_reserved_tpm_and_priority_usage_types_are_ignored(self) -> None:
        source = _offer(
            _entry("APN1-MP:APN1_InputTokenCount-Units", "1.1"),
            _entry("APN1-MP:APN1_Reserved_1Month_InputTPM_Geo-Units", "7.0"),
            _entry("APN1-MP:APN1_input_tokens_priority-Units", "8.0"),
        )

        result = AWSBedrockProvider().parse(source, [_model("ap-northeast-1")])[0]

        self.assertEqual(1.1, result.data["input"][0]["per_million"])
        self.assertFalse(any("multiple" in warning for warning in result.warnings))

    def test_unknown_unit_preserves_price_and_warns(self) -> None:
        source = _offer(
            _entry("APN1-MP:APN1_InputTokenCount-Units", "0.0011", unit="1K tokens")
        )

        result = AWSBedrockProvider().parse(source, [_model("ap-northeast-1")])[0]

        self.assertEqual(99, result.data["input"][0]["per_million"])
        self.assertIn(
            "official unit is not per 1M tokens for input; retained existing value",
            result.warnings,
        )

    def test_parse_does_not_modify_original_models(self) -> None:
        original = _model("ap-northeast-1")
        untouched = copy.deepcopy(original)

        AWSBedrockProvider().parse(
            _offer(*_regional_base_entries()), [original]
        )

        self.assertEqual(untouched, original)


if __name__ == "__main__":
    unittest.main()
