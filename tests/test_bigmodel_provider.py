from __future__ import annotations

import copy
import unittest

from src.providers.bigmodel import BigModelProvider


MARKDOWN = r"""
| 模型名称 | 上下文 | 输入单价（元/百万 Tokens） | 输出单价（元/百万 Tokens） | 缓存存储（元/百万 Tokens/小时） | 缓存命中（元/百万 Tokens） |
| - | - | - | - | - | - |
| GLM-5.2 | 1M | 8 | 28 | 限时免费 | 2 |
| GLM-5.1 | 输入长度 \[0, 32K) | 6 | 24 | 限时免费 | 1.3 |
| GLM-5.1 | 输入长度 ≥32K | 8 | 28 | 限时免费 | 2 |

| 模型名称 | 上下文 | 输入单价（元/百万 Tokens） | 输出单价（元/百万 Tokens） | 缓存存储（元/百万 Tokens/小时） | 缓存命中（元/百万 Tokens） |
| - | - | - | - | - | - |
| GLM-4.6V | 输入长度 \[0, 32K) | 1 | 3 | 限时免费 | 0.2 |
| GLM-4.6V | 输入长度 \[32K, 128K) | 2 | 6 | 限时免费 | 0.4 |
"""


def _model(model_id: str, tiers: list[tuple[int, int | None]]) -> dict:
    def rates() -> list[dict]:
        return [
            {"context_min": minimum, "context_max": maximum, "per_million": 99}
            for minimum, maximum in tiers
        ]

    return {
        "provider": "BigModel",
        "model_api_id": model_id,
        "model_region": "Global",
        "model_uid": f"uid-{model_id}",
        "name": model_id,
        "input": rates(),
        "output": rates(),
        "cache": [{"cache_read": rates()}],
    }


class BigModelProviderTests(unittest.TestCase):
    def test_maps_context_tiers_and_converts_cny_with_project_rate(self) -> None:
        originals = [
            _model("glm-4.6v", [(0, 32000), (32000, 128000)]),
            _model("glm-5.1", [(0, 32000), (32000, None)]),
            _model("glm-5.2", [(0, None)]),
        ]
        untouched = copy.deepcopy(originals)

        results = BigModelProvider().parse(MARKDOWN, originals)

        self.assertEqual(untouched, originals)
        glm46, glm51, glm52 = results
        self.assertEqual([0.15, 0.29], self._prices(glm46, "input"))
        self.assertEqual([0.44, 0.88], self._prices(glm46, "output"))
        self.assertEqual([0.03, 0.06], self._cache_prices(glm46))
        self.assertEqual([0.88, 1.18], self._prices(glm51, "input"))
        self.assertEqual([3.53, 4.12], self._prices(glm51, "output"))
        self.assertEqual([0.19, 0.29], self._cache_prices(glm51))
        self.assertEqual([1.18], self._prices(glm52, "input"))
        self.assertEqual([4.12], self._prices(glm52, "output"))
        self.assertEqual([0.29], self._cache_prices(glm52))
        for result in results:
            self.assertEqual([], result.warnings)
            self.assertEqual(f"uid-{result.model_api_id}", result.data["model_uid"])

    def test_unmatched_context_tier_is_preserved_with_warning(self) -> None:
        model = _model("glm-4.6v", [(0, 32000), (32000, None)])

        result = BigModelProvider().parse(MARKDOWN, [model])[0]

        self.assertEqual([0.15, 99], self._prices(result, "input"))
        self.assertEqual([0.44, 99], self._prices(result, "output"))
        self.assertEqual([0.03, 99], self._cache_prices(result))
        self.assertTrue(
            any("input[32000-∞]" in warning for warning in result.warnings)
        )

    def test_conflicting_price_preserves_affected_tier(self) -> None:
        conflicting = MARKDOWN + MARKDOWN.replace(
            "| GLM-5.2 | 1M | 8 | 28 |", "| GLM-5.2 | 1M | 9 | 28 |"
        )

        result = BigModelProvider().parse(
            conflicting, [_model("glm-5.2", [(0, None)])]
        )[0]

        self.assertEqual([99], self._prices(result, "input"))
        self.assertEqual([4.12], self._prices(result, "output"))
        self.assertTrue(
            any("multiple official prices found for input" in warning for warning in result.warnings)
        )

    @staticmethod
    def _prices(result, field: str) -> list[float]:
        return [rate["per_million"] for rate in result.data[field]]

    @staticmethod
    def _cache_prices(result) -> list[float]:
        return [
            rate["per_million"]
            for rate in result.data["cache"][0]["cache_read"]
        ]


if __name__ == "__main__":
    unittest.main()
