from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from src.providers.alibaba_bailian import AlibabaBailianProvider


def _table(items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f"<tr><td><p>{label}</p></td><td><p>{price}</p></td>"
        "<td><p>每百万<span></span>tokens</p></td></tr>"
        for label, price in items
    )
    return (
        "<table><thead><tr><th>计费项</th><th>价格（元）</th>"
        f"<th>单位</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _page(
    sections: list[tuple[str | None, list[tuple[str, str]]]],
    suffix: str = "",
) -> str:
    pricing = "".join(
        (f"<p>{context}</p>" if context is not None else "") + _table(items)
        for context, items in sections
    )
    return (
        "<html><body><h2>模型价格</h2>"
        "<section><h4>华北<span></span>2（北京）</h4>"
        f"{pricing}<h4>新加坡</h4>{suffix}</section></body></html>"
    )


def _rates(
    tiers: list[tuple[int, int | None]], thinking: list[bool] | None = None
) -> list[dict]:
    values: list[dict] = []
    for minimum, maximum in tiers:
        modes = thinking if thinking is not None else [None]
        for mode in modes:
            rate = {
                "context_min": minimum,
                "context_max": maximum,
                "per_million": 99,
            }
            if mode is not None:
                rate["thinking"] = mode
            values.append(rate)
    return values


def _model(**fields: object) -> dict:
    model = {
        "provider": "Alibaba Bailian",
        "model_api_id": "qwen-plus",
        "model_region": "Global",
        "currency": "USD",
    }
    model.update(fields)
    return model


def _parse(html: str, model: dict):
    return AlibabaBailianProvider().parse(
        {model["model_api_id"]: html}, [model]
    )[0]


class AlibabaBailianProviderTests(unittest.TestCase):
    def test_multiple_context_tiers_match_exactly(self) -> None:
        tiers = [(0, 128000), (128000, 256000), (256000, 1000000)]
        model = _model(input=_rates(tiers), output=_rates(tiers))
        html = _page(
            [
                ("输入&lt;=128k", [("输入", "6.8"), ("输出", "13.6")]),
                (
                    "128k&lt;输入&lt;=256k",
                    [("输入", "13.6"), ("输出", "27.2")],
                ),
                (
                    "256k&lt;输入&lt;=1m",
                    [("输入", "20.4"), ("输出", "40.8")],
                ),
            ]
        )

        result = _parse(html, model)

        self.assertEqual(
            [1.0, 2.0, 3.0],
            [rate["per_million"] for rate in result.data["input"]],
        )
        self.assertEqual(
            [2.0, 4.0, 6.0],
            [rate["per_million"] for rate in result.data["output"]],
        )
        self.assertEqual([], result.warnings)

    def test_output_thinking_and_non_thinking_map_separately(self) -> None:
        model = _model(output=_rates([(0, None)], [False, True]))
        html = _page(
            [(None, [("输出", "6.8"), ("输出（思考）", "13.6")])]
        )

        result = _parse(html, model)

        self.assertEqual(
            [1.0, 2.0],
            [rate["per_million"] for rate in result.data["output"]],
        )

    def test_plain_output_updates_both_local_thinking_modes(self) -> None:
        model = _model(output=_rates([(0, None)], [False, True]))

        result = _parse(_page([(None, [("输出", "6.8")])]), model)

        self.assertEqual(
            [1.0, 1.0],
            [rate["per_million"] for rate in result.data["output"]],
        )
        self.assertEqual([], result.warnings)

    def test_implicit_cache_maps_to_cache_read(self) -> None:
        model = _model(cache=[{"cache_read": _rates([(0, None)])}])

        result = _parse(
            _page([(None, [("输入（缓存命中）", "3.4")])]), model
        )

        self.assertEqual(
            0.5, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertEqual([], result.warnings)

    def test_input_and_explicit_cache_creation_map_to_cache_writes(self) -> None:
        model = _model(
            cache=[
                {
                    "cache_writes": [
                        {
                            "context_min": 0,
                            "context_max": None,
                            "per_million": 99,
                            "5m_per_million": 99,
                        }
                    ]
                }
            ]
        )

        result = _parse(
            _page(
                [
                    (
                        None,
                        [("输入", "6.8"), ("显式缓存创建", "13.6")],
                    )
                ]
            ),
            model,
        )

        cache_write = result.data["cache"][0]["cache_writes"][0]
        self.assertEqual(1.0, cache_write["per_million"])
        self.assertEqual(2.0, cache_write["5m_per_million"])

    def test_missing_explicit_cache_creation_preserves_only_5m_price(self) -> None:
        model = _model(
            cache=[
                {
                    "cache_writes": [
                        {
                            "context_min": 0,
                            "context_max": None,
                            "per_million": 99,
                            "5m_per_million": 99,
                        }
                    ]
                }
            ]
        )

        result = _parse(_page([(None, [("输入", "6.8")])]), model)

        cache_write = result.data["cache"][0]["cache_writes"][0]
        self.assertEqual(1.0, cache_write["per_million"])
        self.assertEqual(99, cache_write["5m_per_million"])
        self.assertTrue(
            any("5m_per_million" in warning for warning in result.warnings)
        )

    def test_batch_file_thinking_prices_map_without_batch_chat(self) -> None:
        model = _model(
            batch={
                "input": _rates([(0, None)]),
                "output": _rates([(0, None)], [False, True]),
            }
        )
        html = _page(
            [
                (
                    None,
                    [
                        ("输入（Batch File）", "6.8"),
                        ("输入（思考模式 Batch File）", "6.8"),
                        ("输出（Batch File）", "13.6"),
                        ("思考模式输出（Batch File）", "27.2"),
                        ("输出（Batch Chat)", "680"),
                    ],
                )
            ]
        )

        result = _parse(html, model)

        self.assertEqual(1.0, result.data["batch"]["input"][0]["per_million"])
        self.assertEqual(
            [2.0, 4.0],
            [rate["per_million"] for rate in result.data["batch"]["output"]],
        )

    def test_missing_implicit_cache_preserves_only_cache_read_and_warns(self) -> None:
        model = _model(
            input=_rates([(0, None)]),
            cache=[{"cache_read": _rates([(0, None)])}],
        )

        result = _parse(_page([(None, [("输入", "6.8")])]), model)

        self.assertEqual(1.0, result.data["input"][0]["per_million"])
        self.assertEqual(
            99, result.data["cache"][0]["cache_read"][0]["per_million"]
        )
        self.assertTrue(any("cache_read" in warning for warning in result.warnings))

    def test_missing_context_tier_preserves_only_that_tier(self) -> None:
        tiers = [(0, 256000), (256000, 1000000)]
        model = _model(input=_rates(tiers))

        result = _parse(
            _page([("输入&lt;=256k", [("输入", "6.8")])]), model
        )

        self.assertEqual(
            [1.0, 99],
            [rate["per_million"] for rate in result.data["input"]],
        )
        self.assertTrue(any("256000-1000000" in warning for warning in result.warnings))

    def test_cny_conversion_uses_fixed_6_8_rate_and_half_up_rounding(self) -> None:
        model = _model(input=_rates([(0, None)]))

        result = _parse(_page([(None, [("输入", "1")])]), model)

        self.assertEqual(0.15, result.data["input"][0]["per_million"])

    def test_batch_cache_is_preserved_and_parse_does_not_mutate_original(self) -> None:
        model = _model(
            batch={"cache": [{"cache_read": _rates([(0, None)])}]}
        )
        untouched = copy.deepcopy(model)

        result = _parse(_page([(None, [("输入", "6.8")])]), model)

        self.assertEqual(untouched, model)
        self.assertEqual(untouched, result.data)
        self.assertIn(
            "official batch.cache combination price not found; retained existing "
            "values",
            result.warnings,
        )

    def test_unrepresentable_input_thinking_difference_is_preserved(self) -> None:
        model = _model(input=_rates([(0, None)]))
        html = _page(
            [(None, [("输入", "6.8"), ("输入（思考）", "13.6")])]
        )

        result = _parse(html, model)

        self.assertEqual(99, result.data["input"][0]["per_million"])
        self.assertTrue(any("Thinking prices conflict" in w for w in result.warnings))

    def test_parser_stops_before_other_regions_and_snapshot_versions(self) -> None:
        model = _model(input=_rates([(0, None)]))
        suffix = (
            _table([("输入", "68")])
            + "<h2>快照版本</h2><h3>qwen-plus-old</h3>"
            + "<h4>模型价格</h4><h4>华北2（北京）</h4>"
            + _table([("输入", "680")])
        )

        result = _parse(_page([(None, [("输入", "6.8")])], suffix), model)

        self.assertEqual(1.0, result.data["input"][0]["per_million"])

    def test_one_fetch_failure_does_not_block_another_model(self) -> None:
        class _Headers:
            def get(self, name: str, default: str = "") -> str:
                return default

            def get_content_charset(self) -> str:
                return "utf-8"

        class _Response:
            headers = _Headers()

            def __init__(self, html: str) -> None:
                self._body = html.encode()

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return self._body

        provider = AlibabaBailianProvider()
        provider.MODEL_URLS = {
            "qwen3.6-plus": "https://example.invalid/first",
            "qwen3.7-max": "https://example.invalid/second",
        }
        first = _model(
            model_api_id="qwen3.6-plus", input=_rates([(0, None)])
        )
        second = _model(
            model_api_id="qwen3.7-max", input=_rates([(0, None)])
        )
        successful_page = _page([(None, [("输入", "6.8")])])

        with patch(
            "src.providers.alibaba_bailian.urlopen",
            side_effect=[TimeoutError("timed out"), _Response(successful_page)],
        ):
            first_result, second_result = provider.run("unused", [first, second])

        self.assertEqual(99, first_result.data["input"][0]["per_million"])
        self.assertTrue(any("timed out" in w for w in first_result.warnings))
        self.assertEqual(1.0, second_result.data["input"][0]["per_million"])
        self.assertEqual([], second_result.warnings)


if __name__ == "__main__":
    unittest.main()
