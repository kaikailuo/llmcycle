from __future__ import annotations

import copy
import hashlib
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.providers.volcengine import VolcengineProvider


ENDPOINT_ID = "ep-20251029183622-7nd6b"
FOUNDATION_MODEL = "doubao-1.5-pro-32k"
VERSIONED_MODEL_ID = "doubao-seed-2-0-pro-260215"
VERSIONED_FOUNDATION_MODEL = "doubao-seed-2-0-pro"
MODEL_VERSION = "260215"
SOURCE_URL = "https://ark.cn-beijing.volcengineapi.com/"


def _rates(tiers: list[tuple[int, int | None]] | None = None) -> list[dict]:
    return [
        {"context_min": minimum, "context_max": maximum, "per_million": 99}
        for minimum, maximum in (tiers or [(0, None)])
    ]


def _model(model_id: str = ENDPOINT_ID) -> dict:
    return {
        "provider": "Volcengine",
        "model_api_id": model_id,
        "model_region": "Global",
        "currency": "USD",
        "input": _rates(),
        "output": _rates(),
        "cache": [{"cache_read": _rates()}],
        "batch": {
            "input": _rates(),
            "output": _rates(),
            "cache": [{"cache_read": _rates()}],
        },
    }


def _response(foundation_model: str = FOUNDATION_MODEL) -> dict:
    return {
        "Result": {
            "Item": {
                "FoundationModelName": foundation_model,
                "ChargeItems": [
                    {
                        "Price": 0.0008,
                        "OriginalPrice": 0.0012,
                        "UnitCode": "CNY/1KToken",
                        "Type": "InferencePrompt",
                    },
                    {
                        "Price": 0.002,
                        "OriginalPrice": 0.003,
                        "UnitCode": "CNY/1KToken",
                        "Type": "InferenceCompletion",
                    },
                ],
                "MultiChargeItems": [],
            }
        }
    }


def _list_response(items: list[dict]) -> dict:
    return {
        "Result": {
            "TotalCount": len(items),
            "PageNumber": 1,
            "PageSize": 100,
            "Items": items,
        }
    }


def _credentials() -> dict[str, str]:
    return {
        "VOLCENGINE_ACCESS_KEY_ID": "test-ak",
        "VOLCENGINE_SECRET_ACCESS_KEY": "test-sk",
    }


class VolcengineProviderTests(unittest.TestCase):
    def test_parses_documented_charge_items_and_preserves_unavailable_fields(self) -> None:
        original = _model(FOUNDATION_MODEL)
        untouched = copy.deepcopy(original)

        result = VolcengineProvider().parse(_response(), [original])[0]

        self.assertEqual(untouched, original)
        self.assertEqual(0.12, result.data["input"][0]["per_million"])
        self.assertEqual(0.29, result.data["output"][0]["per_million"])
        self.assertEqual(
            99, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual(99, result.data["batch"]["input"][0]["per_million"])
        self.assertTrue(any("cache_read" in warning for warning in result.warnings))
        self.assertTrue(any("batch.input" in warning for warning in result.warnings))

    def test_maps_multi_charge_items_by_max_prompt_tokens(self) -> None:
        model = _model("doubao-tiered")
        model["input"] = _rates([(0, 32000), (32000, None)])
        model["output"] = _rates([(0, 32000), (32000, None)])
        del model["cache"]
        del model["batch"]
        response = {
            "Result": {
                "Item": {
                    "FoundationModelName": "doubao-tiered",
                    "ChargeItems": [],
                    "MultiChargeItems": [
                        {
                            "MaxPromptTokens": 32000,
                            "MaxCompletionTokens": 4096,
                            "ChargeItems": [
                                {
                                    "Price": 6.8,
                                    "UnitCode": "CNY/1MToken",
                                    "Type": "InferencePrompt",
                                },
                                {
                                    "Price": 13.6,
                                    "UnitCode": "CNY/1MToken",
                                    "Type": "InferenceCompletion",
                                },
                            ],
                        },
                        {
                            "MaxPromptTokens": 128000,
                            "MaxCompletionTokens": 4096,
                            "ChargeItems": [
                                {
                                    "Price": 13.6,
                                    "UnitCode": "CNY/1MToken",
                                    "Type": "InferencePrompt",
                                },
                                {
                                    "Price": 27.2,
                                    "UnitCode": "CNY/1MToken",
                                    "Type": "InferenceCompletion",
                                },
                            ],
                        },
                    ],
                }
            }
        }

        result = VolcengineProvider().parse(response, [model])[0]

        self.assertEqual([1.0, 2.0], [x["per_million"] for x in result.data["input"]])
        self.assertEqual([2.0, 4.0], [x["per_million"] for x in result.data["output"]])
        self.assertEqual([], result.warnings)

    def test_maps_batch_and_cache_types_without_using_fast_or_storage_prices(
        self,
    ) -> None:
        model = _model("doubao-flat")
        response = {
            "Result": {
                "Item": {
                    "FoundationModelName": "doubao-flat",
                    "ChargeItems": [
                        {
                            "Price": 0.0068,
                            "UnitCode": "千tokens",
                            "Type": "InferencePrompt",
                        },
                        {
                            "Price": 0.0136,
                            "UnitCode": "千tokens",
                            "Type": "InferenceCompletion",
                        },
                        {
                            "Price": 0.0034,
                            "UnitCode": "千tokens",
                            "Type": "ContextSessionHit",
                        },
                        {
                            "Price": 0.0017,
                            "UnitCode": "千tokens",
                            "Type": "BatchInferencePrompt",
                        },
                        {
                            "Price": 0.0051,
                            "UnitCode": "千tokens",
                            "Type": "BatchInferenceCompletion",
                        },
                        {
                            "Price": 0.00068,
                            "UnitCode": "千tokens",
                            "Type": "BatchInferenceCacheHit",
                        },
                        {
                            "Price": 99,
                            "UnitCode": "千tokens",
                            "Type": "FastInferencePrompt",
                        },
                        {
                            "Price": 98,
                            "UnitCode": "千tokens",
                            "Type": "FastInferenceCompletion",
                        },
                        {
                            "Price": 97,
                            "UnitCode": "千tokens",
                            "Type": "FastInferenceCacheHit",
                        },
                        {
                            "Price": 96,
                            "UnitCode": "千tokens/小时",
                            "Type": "ContextSessionStorage",
                        },
                    ],
                    "MultiChargeItems": [],
                }
            }
        }

        result = VolcengineProvider().parse(response, [model])[0]

        self.assertEqual(1.0, result.data["input"][0]["per_million"])
        self.assertEqual(2.0, result.data["output"][0]["per_million"])
        self.assertEqual(
            0.5, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual(0.25, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual(0.75, result.data["batch"]["output"][0]["per_million"])
        self.assertEqual(
            0.1,
            result.data["batch"]["cache"][0]["cache_read"][0]["per_million"],
        )
        self.assertEqual([], result.warnings)

    def test_maps_open_ended_multi_charge_tier_for_every_supported_price(self) -> None:
        tier_bounds = [(0, 32768), (32768, 131072), (131072, None)]
        model = _model("doubao-tiered-all-prices")
        model["input"] = _rates(tier_bounds)
        model["output"] = _rates(tier_bounds)
        model["cache"][0]["cache_read"] = _rates(tier_bounds)
        model["batch"]["input"] = _rates(tier_bounds)
        model["batch"]["output"] = _rates(tier_bounds)
        model["batch"]["cache"][0]["cache_read"] = _rates(tier_bounds)

        tier_values = [
            (32768, [0.0032, 0.016, 0.00064, 0.0016, 0.008, 0.00064]),
            (131072, [0.0048, 0.024, 0.00096, 0.0024, 0.012, 0.00096]),
            (None, [0.0096, 0.048, 0.00192, 0.0048, 0.024, 0.00192]),
        ]
        types = [
            "InferencePrompt",
            "InferenceCompletion",
            "ContextSessionHit",
            "BatchInferencePrompt",
            "BatchInferenceCompletion",
            "BatchInferenceCacheHit",
        ]
        multi_charge_items = []
        for maximum, prices in tier_values:
            tier = {
                "Name": "official tier",
                "Description": "official description",
                "ChargeItems": [
                    {"Price": price, "UnitCode": "千tokens", "Type": type_name}
                    for type_name, price in zip(types, prices, strict=True)
                ],
            }
            if maximum is not None:
                tier["MaxPromptTokens"] = maximum
            multi_charge_items.append(tier)
        response = {
            "Result": {
                "Item": {
                    "FoundationModelName": "doubao-tiered-all-prices",
                    "ChargeItems": [],
                    "MultiChargeItems": multi_charge_items,
                }
            }
        }

        result = VolcengineProvider().parse(response, [model])[0]

        def prices(rates: list[dict]) -> list[float]:
            return [rate["per_million"] for rate in rates]

        self.assertEqual([0.47, 0.71, 1.41], prices(result.data["input"]))
        self.assertEqual([2.35, 3.53, 7.06], prices(result.data["output"]))
        self.assertEqual(
            [0.09, 0.14, 0.28],
            prices(result.data["cache"][0]["cache_read"]),
        )
        self.assertEqual(
            [0.24, 0.35, 0.71], prices(result.data["batch"]["input"])
        )
        self.assertEqual(
            [1.18, 1.76, 3.53], prices(result.data["batch"]["output"])
        )
        self.assertEqual(
            [0.09, 0.14, 0.28],
            prices(result.data["batch"]["cache"][0]["cache_read"]),
        )
        self.assertEqual([], result.warnings)

    def test_multiple_official_prices_retain_existing_value(self) -> None:
        model = _model("doubao-ambiguous")
        del model["cache"]
        del model["batch"]
        response = _response("doubao-ambiguous")
        response["Result"]["Item"]["ChargeItems"].append(
            {
                "Price": 0.0016,
                "UnitCode": "CNY/1KToken",
                "Type": "InferencePrompt",
            }
        )

        result = VolcengineProvider().parse(response, [model])[0]

        self.assertEqual(99, result.data["input"][0]["per_million"])
        self.assertEqual(0.29, result.data["output"][0]["per_million"])
        self.assertTrue(
            any(
                "multiple official prices found for input" in warning
                for warning in result.warnings
            )
        )

    def test_missing_credentials_returns_unchanged_results_with_warning(self) -> None:
        original = _model()
        with patch.dict(os.environ, {}, clear=True):
            results = VolcengineProvider().run(
                "https://ark.cn-beijing.volcengineapi.com/", [original]
            )

        self.assertEqual(original, results[0].data)
        self.assertIn("VOLCENGINE_ACCESS_KEY_ID", results[0].warnings[0])

    def test_versioned_model_id_uses_official_identity_for_activation(self) -> None:
        provider = VolcengineProvider()
        calls: list[tuple[str, dict]] = []

        def call_api(_source_url: str, action: str, payload: dict) -> dict:
            calls.append((action, payload))
            if action == "ListFoundationModels":
                return _list_response([{"Name": VERSIONED_FOUNDATION_MODEL}])
            if action == "ListFoundationModelVersions":
                return _list_response(
                    [
                        {
                            "FoundationModelName": VERSIONED_FOUNDATION_MODEL,
                            "ModelVersion": MODEL_VERSION,
                        }
                    ]
                )
            if action == "GetModelActivation":
                return _response(VERSIONED_FOUNDATION_MODEL)
            self.fail(f"unexpected action: {action}")

        with patch.dict(os.environ, _credentials(), clear=True), patch.object(
            provider, "_call_api", side_effect=call_api
        ):
            result = provider.run(SOURCE_URL, [_model(VERSIONED_MODEL_ID)])[0]

        activation_payloads = [
            payload for action, payload in calls if action == "GetModelActivation"
        ]
        self.assertEqual(
            [
                {
                    "FoundationModelName": VERSIONED_FOUNDATION_MODEL,
                    "WithPrice": True,
                }
            ],
            activation_payloads,
        )
        self.assertEqual(VERSIONED_MODEL_ID, result.data["model_api_id"])
        self.assertEqual(0.12, result.data["input"][0]["per_million"])

    def test_unknown_model_version_is_retained_without_activation(self) -> None:
        provider = VolcengineProvider()
        calls: list[str] = []

        def call_api(_source_url: str, action: str, _payload: dict) -> dict:
            calls.append(action)
            if action == "ListFoundationModels":
                return _list_response([{"Name": VERSIONED_FOUNDATION_MODEL}])
            if action == "ListFoundationModelVersions":
                return _list_response(
                    [
                        {
                            "FoundationModelName": VERSIONED_FOUNDATION_MODEL,
                            "ModelVersion": "260101",
                        }
                    ]
                )
            self.fail(f"unexpected action: {action}")

        original = _model(VERSIONED_MODEL_ID)
        with patch.dict(os.environ, _credentials(), clear=True), patch.object(
            provider, "_call_api", side_effect=call_api
        ):
            result = provider.run(SOURCE_URL, [original])[0]

        self.assertEqual(original, result.data)
        self.assertNotIn("GetModelActivation", calls)
        self.assertTrue(
            any(
                "model version 260215 was not found under foundation model "
                "doubao-seed-2-0-pro" in warning
                for warning in result.warnings
            )
        )

    def test_endpoint_id_uses_get_endpoint_foundation_model_reference(self) -> None:
        provider = VolcengineProvider()
        calls: list[tuple[str, dict]] = []

        def call_api(_source_url: str, action: str, payload: dict) -> dict:
            calls.append((action, payload))
            if action == "GetEndpoint":
                return {
                    "Result": {
                        "Id": ENDPOINT_ID,
                        "EndpointModelType": "FoundationModel",
                        "ModelReference": {
                            "FoundationModel": {
                                "Name": FOUNDATION_MODEL,
                                "ModelVersion": "250115",
                            }
                        },
                    }
                }
            if action == "GetModelActivation":
                return _response()
            self.fail(f"unexpected action: {action}")

        with patch.dict(os.environ, _credentials(), clear=True), patch.object(
            provider, "_call_api", side_effect=call_api
        ):
            result = provider.run(SOURCE_URL, [_model()])[0]

        self.assertEqual(
            [
                ("GetEndpoint", {"Id": ENDPOINT_ID}),
                (
                    "GetModelActivation",
                    {
                        "FoundationModelName": FOUNDATION_MODEL,
                        "WithPrice": True,
                    },
                ),
            ],
            calls,
        )
        self.assertEqual(ENDPOINT_ID, result.data["model_api_id"])

    def test_one_model_request_failure_does_not_block_other_models(self) -> None:
        provider = VolcengineProvider()
        def call_api(_source_url: str, action: str, payload: dict) -> dict:
            if action == "GetEndpoint":
                raise OSError("temporary failure")
            if action == "ListFoundationModels":
                return _list_response([{"Name": VERSIONED_FOUNDATION_MODEL}])
            if action == "ListFoundationModelVersions":
                return _list_response(
                    [
                        {
                            "FoundationModelName": VERSIONED_FOUNDATION_MODEL,
                            "ModelVersion": MODEL_VERSION,
                        }
                    ]
                )
            if action == "GetModelActivation":
                self.assertEqual(
                    VERSIONED_FOUNDATION_MODEL,
                    payload["FoundationModelName"],
                )
                return _response(VERSIONED_FOUNDATION_MODEL)
            self.fail(f"unexpected action: {action}")

        with patch.dict(os.environ, _credentials(), clear=True), patch.object(
            provider, "_call_api", side_effect=call_api
        ):
            first, second = provider.run(
                SOURCE_URL, [_model(), _model(VERSIONED_MODEL_ID)]
            )

        self.assertEqual(99, first.data["input"][0]["per_million"])
        self.assertIn("temporary failure", first.warnings[0])
        self.assertEqual(0.12, second.data["input"][0]["per_million"])
        self.assertEqual(0.29, second.data["output"][0]["per_million"])

    def test_official_catalog_and_foundation_versions_are_cached_per_run(self) -> None:
        provider = VolcengineProvider()
        second_version = "260216"
        second_model_id = f"{VERSIONED_FOUNDATION_MODEL}-{second_version}"
        calls: list[str] = []

        def call_api(_source_url: str, action: str, _payload: dict) -> dict:
            calls.append(action)
            if action == "ListFoundationModels":
                return _list_response([{"Name": VERSIONED_FOUNDATION_MODEL}])
            if action == "ListFoundationModelVersions":
                return _list_response(
                    [
                        {
                            "FoundationModelName": VERSIONED_FOUNDATION_MODEL,
                            "ModelVersion": version,
                        }
                        for version in (MODEL_VERSION, second_version)
                    ]
                )
            if action == "GetModelActivation":
                return _response(VERSIONED_FOUNDATION_MODEL)
            self.fail(f"unexpected action: {action}")

        with patch.dict(os.environ, _credentials(), clear=True), patch.object(
            provider, "_call_api", side_effect=call_api
        ):
            results = provider.run(
                SOURCE_URL,
                [_model(VERSIONED_MODEL_ID), _model(second_model_id)],
            )

        self.assertEqual(1, calls.count("ListFoundationModels"))
        self.assertEqual(1, calls.count("ListFoundationModelVersions"))
        self.assertEqual(2, calls.count("GetModelActivation"))
        self.assertTrue(
            all(result.data["input"][0]["per_million"] == 0.12 for result in results)
        )

    def test_signed_request_contains_official_action_version_and_body_hash(self) -> None:
        body = json.dumps(
            {"FoundationModelName": FOUNDATION_MODEL, "WithPrice": True},
            separators=(",", ":"),
        ).encode()
        request = VolcengineProvider._signed_request(
            "https://ark.cn-beijing.volcengineapi.com/",
            body,
            "test-ak",
            "test-sk",
            datetime(2026, 9, 30, 1, 2, 3, tzinfo=timezone.utc),
        )

        self.assertEqual(
            "https://ark.cn-beijing.volcengineapi.com/"
            "?Action=GetModelActivation&Version=2024-01-01",
            request.full_url,
        )
        self.assertEqual("POST", request.get_method())
        self.assertEqual(body, request.data)
        self.assertEqual("20260930T010203Z", request.get_header("X-date"))
        self.assertEqual(
            hashlib.sha256(body).hexdigest(), request.get_header("X-content-sha256")
        )
        authorization = request.get_header("Authorization")
        self.assertIn("Credential=test-ak/20260930/cn-beijing/ark/request", authorization)
        self.assertIn(
            "SignedHeaders=content-type;host;x-content-sha256;x-date", authorization
        )

    def test_signed_request_supports_other_ark_openapi_actions(self) -> None:
        body = b'{"Id":"ep-test"}'
        request = VolcengineProvider._signed_request(
            SOURCE_URL,
            body,
            "test-ak",
            "test-sk",
            datetime(2026, 9, 30, 1, 2, 3, tzinfo=timezone.utc),
            "GetEndpoint",
        )

        self.assertEqual(
            f"{SOURCE_URL}?Action=GetEndpoint&Version=2024-01-01",
            request.full_url,
        )


if __name__ == "__main__":
    unittest.main()
