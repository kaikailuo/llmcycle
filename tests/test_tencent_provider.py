from __future__ import annotations

import unittest

from src.providers.tencent import TencentProvider


TABLE = """
<table>
  <tr><th>模型名称</th><th>条件（token）</th><th>峰谷计费</th><th>推理输入（元/百万 tokens）</th><th>推理输出（元/百万 tokens）</th><th>缓存命中（元/百万 tokens）</th></tr>
  <tr><td>Hy3 preview</td><td>-</td><td>-</td><td>6</td><td>18</td><td>0.3</td></tr>
  <tr><td>Hy3</td><td>-</td><td>-</td><td>1</td><td>4</td><td>0.25</td></tr>
</table>
"""


def _model() -> dict:
    return {
        "provider": "Tencent Hunyuan",
        "model_api_id": "hy3",
        "model_region": "Global",
        "input": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "output": [{"context_min": 0, "context_max": None, "per_million": 99}],
        "cache": [
            {
                "cache_read": [
                    {"context_min": 0, "context_max": None, "per_million": 99}
                ]
            }
        ],
    }


class TencentProviderTests(unittest.TestCase):
    def test_matches_hy3_exactly_converts_cny_and_ignores_identical_duplicates(self) -> None:
        result = TencentProvider().parse(TABLE + TABLE, [_model()])[0]

        self.assertEqual(0.15, result.data["input"][0]["per_million"])
        self.assertEqual(0.59, result.data["output"][0]["per_million"])
        self.assertEqual(
            0.04, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual([], result.warnings)

    def test_missing_official_field_preserves_only_that_field(self) -> None:
        source = TABLE.replace("<td>0.25</td></tr>", "<td>-</td></tr>")
        original = _model()

        result = TencentProvider().parse(source, [original])[0]

        self.assertEqual(0.15, result.data["input"][0]["per_million"])
        self.assertEqual(
            99, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertIn(
            "official price not found for cache.cache_read; retained existing values",
            result.warnings,
        )


if __name__ == "__main__":
    unittest.main()
