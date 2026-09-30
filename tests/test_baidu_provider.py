from __future__ import annotations

import unittest

from src.providers.baidu import BaiduProvider


HTML = """
<table>
  <tr><th>模型名称</th><th>版本名称</th><th>服务内容</th><th>子项</th><th>在线推理</th><th>批量推理</th><th>单位</th></tr>
  <tr>
    <td rowspan="4">ERNIE 5.1</td><td rowspan="4">ERNIE-5.1</td>
    <td rowspan="4">推理服务</td><td>输入（输入&lt;=32k）</td>
    <td>0.004</td><td>-</td><td rowspan="4">元/千tokens</td>
  </tr>
  <tr><td>输出（输入&lt;=32k）</td><td>0.018</td><td>-</td></tr>
  <tr><td>输入（32k&lt;输入&lt;=128k）</td><td>0.006</td><td>-</td></tr>
  <tr><td>输出（32k&lt;输入&lt;=128k）</td><td>0.022</td><td>-</td></tr>
</table>
"""


def _model() -> dict:
    return {
        "provider": "Baidu ERNIE",
        "model_api_id": "ernie-5.1",
        "model_region": "Global",
        "input": [
            {"context_min": 0, "context_max": 32000, "per_million": 99},
            {"context_min": 32000, "context_max": 128000, "per_million": 99},
        ],
        "output": [
            {"context_min": 0, "context_max": 32000, "per_million": 99},
            {"context_min": 32000, "context_max": 128000, "per_million": 99},
        ],
    }


class BaiduProviderTests(unittest.TestCase):
    def test_maps_context_tiers_and_converts_thousand_cny_to_million_usd(self) -> None:
        result = BaiduProvider().parse(HTML, [_model()])[0]

        self.assertEqual([0.59, 0.88], [rate["per_million"] for rate in result.data["input"]])
        self.assertEqual([2.65, 3.24], [rate["per_million"] for rate in result.data["output"]])
        self.assertEqual([], result.warnings)

    def test_missing_one_field_preserves_only_that_field(self) -> None:
        source = HTML.replace(
            "<tr><td>输出（32k&lt;输入&lt;=128k）</td><td>0.022</td><td>-</td></tr>",
            "",
        )
        original = _model()

        result = BaiduProvider().parse(source, [original])[0]

        self.assertEqual(0.88, result.data["input"][1]["per_million"])
        self.assertEqual(99, result.data["output"][1]["per_million"])
        self.assertIn(
            "official price not found for output[32000-128000]; retained existing value",
            result.warnings,
        )


if __name__ == "__main__":
    unittest.main()
