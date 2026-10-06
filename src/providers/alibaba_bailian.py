from __future__ import annotations

import copy
import gzip
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult
from src.utils.currency import cny_to_usd
from src.utils.http import urlopen_with_retry


MODEL_URLS = {
    "qwen3.6-plus": "https://help.aliyun.com/zh/model-studio/qwen3-6-plus",
    "qwen3.7-max": "https://help.aliyun.com/zh/model-studio/qwen3-7-max",
    "qwen3.7-plus": "https://help.aliyun.com/zh/model-studio/qwen3-7-plus",
    "qwen-plus": "https://help.aliyun.com/zh/model-studio/qwen-plus",
    "qwen-vl-plus": "https://help.aliyun.com/zh/model-studio/qwen-vl-plus",
}


_Tier = tuple[int, int | None]


@dataclass(frozen=True)
class _FetchedPage:
    html: str | None
    error: str | None


@dataclass(frozen=True)
class _PricingRecord:
    context_min: int
    context_max: int | None
    billing_item: str
    price_cny: Decimal


class _PricingPageParser(HTMLParser):
    """Extract only the first Beijing pricing section from a model page."""

    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.found_price_heading = False
        self.found_beijing_heading = False
        self.finished = False
        self.tables: list[tuple[_Tier, list[list[str]]]] = []

        self._capture_tag: str | None = None
        self._capture_parts: list[str] = []
        self._current_tier: _Tier = (0, None)
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if self.finished:
            return
        if tag in self._HEADING_TAGS | {"p"} and self._capture_tag is None:
            self._capture_tag = tag
            self._capture_parts = []

        if tag == "table" and self.found_beijing_heading:
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_data(self, data: str) -> None:
        if self.finished:
            return
        if self._capture_tag is not None:
            self._capture_parts.append(data)
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.finished:
            return
        if tag in {"td", "th"} and self._cell is not None:
            if self._row is not None:
                self._row.append(_text("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row and self._table is not None:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append((self._current_tier, self._table))
            self._table = None

        if tag != self._capture_tag:
            return
        value = _text("".join(self._capture_parts))
        compact = _compact(value)
        captured_tag = self._capture_tag
        self._capture_tag = None
        self._capture_parts = []

        if not self.found_price_heading:
            if captured_tag in self._HEADING_TAGS and compact == "模型价格":
                self.found_price_heading = True
            return

        if not self.found_beijing_heading:
            if captured_tag in self._HEADING_TAGS and compact == "华北2(北京)":
                self.found_beijing_heading = True
            return

        # Context tiers are paragraphs immediately before their tables.  Once
        # another heading begins, the Beijing section is over; this also keeps
        # later snapshot-version pricing out of the current alias model.
        if captured_tag in self._HEADING_TAGS:
            self.finished = True
        elif captured_tag == "p" and self._table is None:
            tier = _parse_context(value)
            if tier is not None:
                self._current_tier = tier


def _text(value: str) -> str:
    return " ".join(value.split())


def _compact(value: str) -> str:
    return (
        "".join(value.split())
        .replace("（", "(")
        .replace("）", ")")
        .replace("≤", "<=")
        .casefold()
    )


def _token_count(number: str, unit: str) -> int | None:
    try:
        value = Decimal(number) * (1000 if unit.casefold() == "k" else 1000000)
    except InvalidOperation:
        return None
    if value != value.to_integral_value() or value < 0:
        return None
    return int(value)


def _parse_context(value: str) -> _Tier | None:
    compact = _compact(value)
    upper = re.fullmatch(
        r"输入(?:<=|=<)([0-9]+(?:\.[0-9]+)?)([km])", compact
    )
    if upper:
        maximum = _token_count(upper.group(1), upper.group(2))
        return (0, maximum) if maximum is not None else None

    bounded = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)([km])<输入(?:<=|=<)"
        r"([0-9]+(?:\.[0-9]+)?)([km])",
        compact,
    )
    if bounded:
        minimum = _token_count(bounded.group(1), bounded.group(2))
        maximum = _token_count(bounded.group(3), bounded.group(4))
        if minimum is not None and maximum is not None and minimum < maximum:
            return minimum, maximum
    return None


class AlibabaBailianProvider(BaseProvider):
    name = "Alibaba Bailian"
    MODEL_URLS = MODEL_URLS

    _USD_QUANTUM = Decimal("0.01")
    _INPUT = "输入"
    _THINKING_INPUT = "输入(思考)"
    _OUTPUT = "输出"
    _THINKING_OUTPUT = "输出(思考)"
    _CACHE_READ = "输入(缓存命中)"
    _THINKING_CACHE_READ = "输入(思考模式缓存命中)"
    _EXPLICIT_CACHE_CREATE = "显式缓存创建"
    _THINKING_EXPLICIT_CACHE_CREATE = "显式缓存创建(思考)"
    _BATCH_INPUT = "输入(batchfile)"
    _THINKING_BATCH_INPUT = "输入(思考模式batchfile)"
    _BATCH_OUTPUT = "输出(batchfile)"
    _THINKING_BATCH_OUTPUT = "思考模式输出(batchfile)"

    def fetch(self, source_url: str) -> dict[str, _FetchedPage]:
        # The registry URL identifies the provider source, while each supported
        # model deliberately uses its own explicit official Model Info URL.
        del source_url
        pages: dict[str, _FetchedPage] = {}
        for model_id, url in self.MODEL_URLS.items():
            try:
                request = Request(
                    url,
                    headers={
                        "Accept": "text/html, text/plain;q=0.9",
                        "Accept-Encoding": "identity",
                        "User-Agent": "llmcycle-pricing-maintainer/1.0",
                    },
                )
                with urlopen_with_retry(
                    request, timeout=30, opener=urlopen
                ) as response:
                    body = response.read()
                    if (
                        response.headers.get("Content-Encoding", "").lower()
                        == "gzip"
                        or body.startswith(b"\x1f\x8b")
                    ):
                        body = gzip.decompress(body)
                    encoding = response.headers.get_content_charset() or "utf-8"
                    html = body.decode(encoding)
                pages[model_id] = _FetchedPage(html=html, error=None)
            except Exception as exc:
                pages[model_id] = _FetchedPage(
                    html=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return pages

    def parse(
        self, raw_data: Any, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        pages = raw_data if isinstance(raw_data, dict) else {}
        results: list[ModelResult] = []
        for original in models:
            candidate = copy.deepcopy(original)
            warnings: list[str] = []
            model_id = original.get("model_api_id")
            region = original.get("model_region")

            if not isinstance(model_id, str) or not isinstance(region, str):
                warnings.append("model_api_id or model_region is missing or invalid")
            elif model_id not in self.MODEL_URLS:
                warnings.append(
                    "official Model Info URL not configured; retained existing values"
                )
            else:
                page = pages.get(model_id)
                if isinstance(page, str):
                    page = _FetchedPage(html=page, error=None)
                if not isinstance(page, _FetchedPage):
                    warnings.append(
                        "official page result not found; retained existing values"
                    )
                elif page.error is not None:
                    warnings.append(
                        f"official page fetch failed: {page.error}; "
                        "retained existing values"
                    )
                elif page.html is None:
                    warnings.append(
                        "official page content not found; retained existing values"
                    )
                else:
                    try:
                        records = self._parse_records(page.html)
                        if not records:
                            warnings.append(
                                "official Beijing pricing table not found; "
                                "retained existing values"
                            )
                        else:
                            self._update_candidate(candidate, records, warnings)
                    except Exception as exc:
                        warnings.append(
                            f"official page parsing failed: {type(exc).__name__}: "
                            f"{exc}; retained existing values"
                        )

            results.append(
                ModelResult(
                    provider=self.name,
                    model_api_id=str(model_id or "<missing>"),
                    model_region=str(region or "<missing>"),
                    data=candidate,
                    warnings=list(dict.fromkeys(warnings)),
                )
            )
        return results

    @classmethod
    def _parse_records(cls, html: str) -> list[_PricingRecord]:
        parser = _PricingPageParser()
        parser.feed(html)
        if not parser.found_price_heading or not parser.found_beijing_heading:
            return []

        records: list[_PricingRecord] = []
        for tier, table in parser.tables:
            header_index = next(
                (
                    index
                    for index, row in enumerate(table)
                    if "计费项" in {_compact(cell) for cell in row}
                    and "价格(元)" in {_compact(cell) for cell in row}
                    and "单位" in {_compact(cell) for cell in row}
                ),
                None,
            )
            if header_index is None:
                continue
            headers = [_compact(cell) for cell in table[header_index]]
            billing_index = headers.index("计费项")
            price_index = headers.index("价格(元)")
            unit_index = headers.index("单位")
            for row in table[header_index + 1 :]:
                if max(billing_index, price_index, unit_index) >= len(row):
                    continue
                unit = _compact(row[unit_index])
                if unit not in {"每百万token", "每百万tokens"}:
                    continue
                price = cls._decimal_price(row[price_index])
                billing_item = _compact(row[billing_index])
                if price is None or not billing_item:
                    continue
                records.append(
                    _PricingRecord(
                        context_min=tier[0],
                        context_max=tier[1],
                        billing_item=billing_item,
                        price_cny=price,
                    )
                )
        return records

    @staticmethod
    def _decimal_price(value: str) -> Decimal | None:
        try:
            price = Decimal(value.replace(",", "").strip())
        except InvalidOperation:
            return None
        return price if price.is_finite() and price >= 0 else None

    @classmethod
    def _price_map(
        cls, records: list[_PricingRecord]
    ) -> dict[_Tier, dict[str, set[Decimal]]]:
        prices: dict[_Tier, dict[str, set[Decimal]]] = {}
        for record in records:
            tier = (record.context_min, record.context_max)
            prices.setdefault(tier, {}).setdefault(
                record.billing_item, set()
            ).add(record.price_cny)
        return prices

    @classmethod
    def _update_candidate(
        cls,
        candidate: dict[str, Any],
        records: list[_PricingRecord],
        warnings: list[str],
    ) -> None:
        prices = cls._price_map(records)
        if "input" in candidate:
            cls._update_untyped_rates(
                candidate["input"],
                prices,
                cls._INPUT,
                cls._THINKING_INPUT,
                "input",
                "per_million",
                warnings,
            )
        if "output" in candidate:
            cls._update_typed_rates(
                candidate["output"],
                prices,
                cls._OUTPUT,
                cls._THINKING_OUTPUT,
                "output",
                warnings,
            )
        if "cache" in candidate:
            cls._update_cache(candidate["cache"], prices, warnings)
        if "batch" in candidate:
            cls._update_batch(candidate["batch"], prices, warnings)

    @classmethod
    def _update_cache(
        cls,
        groups: Any,
        prices: dict[_Tier, dict[str, set[Decimal]]],
        warnings: list[str],
    ) -> None:
        if not isinstance(groups, list):
            warnings.append(
                "cache has an unexpected structure; retained existing value"
            )
            return
        for index, group in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"[{index}]"
            if not isinstance(group, dict):
                warnings.append(
                    f"cache{suffix} has an unexpected structure; "
                    "retained existing value"
                )
                continue
            if "cache_read" in group:
                cls._update_untyped_rates(
                    group["cache_read"],
                    prices,
                    cls._CACHE_READ,
                    cls._THINKING_CACHE_READ,
                    f"cache{suffix}.cache_read",
                    "per_million",
                    warnings,
                )
            if "cache_writes" in group:
                cls._update_cache_writes(
                    group["cache_writes"],
                    prices,
                    f"cache{suffix}.cache_writes",
                    warnings,
                )

    @classmethod
    def _update_cache_writes(
        cls,
        rates: Any,
        prices: dict[_Tier, dict[str, set[Decimal]]],
        path: str,
        warnings: list[str],
    ) -> None:
        if not isinstance(rates, list):
            warnings.append(
                f"{path} has an unexpected structure; retained existing value"
            )
            return
        for rate in rates:
            tier = cls._rate_tier(rate, path, warnings)
            if tier is None:
                continue
            if "per_million" in rate:
                price = cls._untyped_price(
                    prices,
                    tier,
                    cls._INPUT,
                    cls._THINKING_INPUT,
                    cls._rate_path(path, tier, "per_million"),
                    warnings,
                )
                if price is not None:
                    rate["per_million"] = cls._to_usd_float(price)
            if "5m_per_million" in rate:
                price = cls._untyped_price(
                    prices,
                    tier,
                    cls._EXPLICIT_CACHE_CREATE,
                    cls._THINKING_EXPLICIT_CACHE_CREATE,
                    cls._rate_path(path, tier, "5m_per_million"),
                    warnings,
                )
                if price is not None:
                    rate["5m_per_million"] = cls._to_usd_float(price)

    @classmethod
    def _update_batch(
        cls,
        batch: Any,
        prices: dict[_Tier, dict[str, set[Decimal]]],
        warnings: list[str],
    ) -> None:
        if not isinstance(batch, dict):
            warnings.append(
                "batch has an unexpected structure; retained existing value"
            )
            return
        if "input" in batch:
            cls._update_untyped_rates(
                batch["input"],
                prices,
                cls._BATCH_INPUT,
                cls._THINKING_BATCH_INPUT,
                "batch.input",
                "per_million",
                warnings,
            )
        if "output" in batch:
            cls._update_typed_rates(
                batch["output"],
                prices,
                cls._BATCH_OUTPUT,
                cls._THINKING_BATCH_OUTPUT,
                "batch.output",
                warnings,
            )
        if "cache" in batch:
            warnings.append(
                "official batch.cache combination price not found; "
                "retained existing values"
            )

    @classmethod
    def _update_untyped_rates(
        cls,
        rates: Any,
        prices: dict[_Tier, dict[str, set[Decimal]]],
        ordinary_label: str,
        thinking_label: str,
        path: str,
        value_key: str,
        warnings: list[str],
    ) -> None:
        if not isinstance(rates, list):
            warnings.append(
                f"{path} has an unexpected structure; retained existing value"
            )
            return
        for rate in rates:
            tier = cls._rate_tier(rate, path, warnings)
            if tier is None:
                continue
            rate_path = cls._rate_path(path, tier)
            if value_key not in rate:
                warnings.append(
                    f"{rate_path} has an unexpected structure; retained existing value"
                )
                continue
            price = cls._untyped_price(
                prices,
                tier,
                ordinary_label,
                thinking_label,
                rate_path,
                warnings,
            )
            if price is not None:
                rate[value_key] = cls._to_usd_float(price)

    @classmethod
    def _update_typed_rates(
        cls,
        rates: Any,
        prices: dict[_Tier, dict[str, set[Decimal]]],
        ordinary_label: str,
        thinking_label: str,
        path: str,
        warnings: list[str],
    ) -> None:
        if not isinstance(rates, list):
            warnings.append(
                f"{path} has an unexpected structure; retained existing value"
            )
            return
        for rate in rates:
            tier = cls._rate_tier(rate, path, warnings)
            if tier is None:
                continue
            thinking = rate.get("thinking", False)
            rate_path = cls._rate_path(path, tier, thinking=thinking)
            if "per_million" not in rate or not isinstance(thinking, bool):
                warnings.append(
                    f"{rate_path} has an unexpected structure; retained existing value"
                )
                continue
            price = cls._typed_price(
                prices,
                tier,
                ordinary_label,
                thinking_label,
                thinking,
                rate_path,
                warnings,
            )
            if price is not None:
                rate["per_million"] = cls._to_usd_float(price)

    @classmethod
    def _untyped_price(
        cls,
        prices: dict[_Tier, dict[str, set[Decimal]]],
        tier: _Tier,
        ordinary_label: str,
        thinking_label: str,
        path: str,
        warnings: list[str],
    ) -> Decimal | None:
        ordinary = prices.get(tier, {}).get(_compact(ordinary_label), set())
        thinking = prices.get(tier, {}).get(_compact(thinking_label), set())
        if len(ordinary) != 1:
            cls._price_warning(path, ordinary, warnings)
            return None
        ordinary_price = next(iter(ordinary))
        if not thinking:
            return ordinary_price
        if len(thinking) != 1 or next(iter(thinking)) != ordinary_price:
            warnings.append(
                f"official standard and Thinking prices conflict for {path}; "
                "retained existing value"
            )
            return None
        return ordinary_price

    @classmethod
    def _typed_price(
        cls,
        prices: dict[_Tier, dict[str, set[Decimal]]],
        tier: _Tier,
        ordinary_label: str,
        thinking_label: str,
        thinking: bool,
        path: str,
        warnings: list[str],
    ) -> Decimal | None:
        tier_prices = prices.get(tier, {})
        ordinary = tier_prices.get(_compact(ordinary_label), set())
        thinking_prices = tier_prices.get(_compact(thinking_label), set())
        selected = thinking_prices if thinking and thinking_prices else ordinary
        if len(selected) != 1:
            cls._price_warning(path, selected, warnings)
            return None
        return next(iter(selected))

    @staticmethod
    def _price_warning(
        path: str, values: set[Decimal], warnings: list[str]
    ) -> None:
        if not values:
            warnings.append(
                f"official price not found for {path}; retained existing value"
            )
        else:
            warnings.append(
                f"multiple official prices found for {path}; retained existing value"
            )

    @staticmethod
    def _rate_tier(
        rate: Any, path: str, warnings: list[str]
    ) -> _Tier | None:
        if not isinstance(rate, dict):
            warnings.append(
                f"{path} has an unexpected structure; retained existing value"
            )
            return None
        minimum = rate.get("context_min")
        maximum = rate.get("context_max")
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or (
                maximum is not None
                and (not isinstance(maximum, int) or isinstance(maximum, bool))
            )
        ):
            warnings.append(
                f"{path} has an invalid context tier; retained existing value"
            )
            return None
        return minimum, maximum

    @staticmethod
    def _rate_path(
        path: str,
        tier: _Tier,
        field: str | None = None,
        thinking: bool | None = None,
    ) -> str:
        maximum = tier[1] if tier[1] is not None else "∞"
        suffix = f", thinking={str(thinking).lower()}" if thinking is not None else ""
        field_suffix = f".{field}" if field is not None else ""
        return f"{path}[{tier[0]}-{maximum}{suffix}]{field_suffix}"

    @classmethod
    def _to_usd_float(cls, value: Decimal) -> float:
        return float(
            cny_to_usd(value).quantize(cls._USD_QUANTUM, rounding=ROUND_HALF_UP)
        )
