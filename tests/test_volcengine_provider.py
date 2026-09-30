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


class VolcengineProviderTests(unittest.TestCase):
    def test_parses_documented_charge_items_and_preserves_unavailable_fields(self) -> None:
        original = _model()
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

    def test_missing_credentials_returns_unchanged_results_with_warning(self) -> None:
        original = _model()
        with patch.dict(os.environ, {}, clear=True):
            results = VolcengineProvider().run(
                "https://ark.cn-beijing.volcengineapi.com/", [original]
            )

        self.assertEqual(original, results[0].data)
        self.assertIn("VOLCENGINE_ACCESS_KEY_ID", results[0].warnings[0])

    def test_endpoint_id_uses_explicit_foundation_model_mapping(self) -> None:
        provider = VolcengineProvider()
        with patch.dict(
            os.environ,
            {
                "VOLCENGINE_ACCESS_KEY_ID": "test-ak",
                "VOLCENGINE_SECRET_ACCESS_KEY": "test-sk",
            },
            clear=True,
        ), patch.object(provider, "fetch", return_value=_response()) as mocked:
            result = provider.run(
                "https://ark.cn-beijing.volcengineapi.com/", [_model()]
            )[0]

        mocked.assert_called_once_with(
            "https://ark.cn-beijing.volcengineapi.com/", FOUNDATION_MODEL
        )
        self.assertEqual(ENDPOINT_ID, result.data["model_api_id"])

    def test_one_model_request_failure_does_not_block_other_models(self) -> None:
        provider = VolcengineProvider()
        second_id = "doubao-seed-2-0-pro-260215"
        with patch.dict(
            os.environ,
            {
                "VOLCENGINE_ACCESS_KEY_ID": "test-ak",
                "VOLCENGINE_SECRET_ACCESS_KEY": "test-sk",
            },
            clear=True,
        ), patch.object(
            provider,
            "fetch",
            side_effect=[OSError("temporary failure"), _response(second_id)],
        ):
            first, second = provider.run(
                "https://ark.cn-beijing.volcengineapi.com/",
                [_model(), _model(second_id)],
            )

        self.assertEqual(99, first.data["input"][0]["per_million"])
        self.assertIn("temporary failure", first.warnings[0])
        self.assertEqual(0.12, second.data["input"][0]["per_million"])
        self.assertEqual(0.29, second.data["output"][0]["per_million"])

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


if __name__ == "__main__":
    unittest.main()
