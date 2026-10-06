from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult
from src.utils.http import urlopen_with_retry


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row and self._table is not None:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


@dataclass(frozen=True)
class _PricingRecord:
    source_name: str
    input: Decimal | None
    output: Decimal | None
    cache_read: Decimal | None


class DeepSeekProvider(BaseProvider):
    name = "DeepSeek"

    # The pricing page explicitly states that these two retired model IDs are
    # served by DeepSeek-V4.1-Flash and billed at the deepseek-flash price.
    MODEL_ALIASES: dict[str, tuple[str, ...]] = {
        "deepseek-v4-pro": ("deepseek-v4-pro",),
        "deepseek-v4-flash": ("deepseek-flash",),
        "deepseek-v4-flash-vision-exp": ("deepseek-flash",),
    }

    def fetch(self, source_url: str) -> str:
        request = Request(
            source_url,
            headers={
                "Accept": "text/html, text/plain;q=0.9",
                "User-Agent": "llmcycle-pricing-maintainer/1.0",
            },
        )
        with urlopen_with_retry(request, timeout=30, opener=urlopen) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding)

    def parse(
        self, raw_data: str, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        records = self._parse_records(raw_data)
        if not records:
            raise ValueError("DeepSeek pricing table is empty")

        results: list[ModelResult] = []
        for original in models:
            candidate = copy.deepcopy(original)
            warnings: list[str] = []
            model_id = original.get("model_api_id")
            region = original.get("model_region")
            if not isinstance(model_id, str) or not isinstance(region, str):
                warnings.append("model_api_id or model_region is missing or invalid")
                results.append(
                    ModelResult(
                        provider=self.name,
                        model_api_id=str(model_id or "<missing>"),
                        model_region=str(region or "<missing>"),
                        data=candidate,
                        warnings=warnings,
                    )
                )
                continue

            names = (model_id, *self.MODEL_ALIASES.get(model_id, ()))
            matches = {
                record
                for name in names
                for record in records.get(self._normalize(name), set())
            }
            if not matches:
                warnings.append("official price not found")
            elif len(matches) > 1:
                warnings.append(
                    "multiple official price records matched; retained existing values"
                )
            else:
                record = next(iter(matches))
                self._update_rates(
                    candidate.get("input"), record.input, "input", warnings
                )
                self._update_rates(
                    candidate.get("output"), record.output, "output", warnings
                )
                if "cache" in candidate:
                    self._update_cache(candidate["cache"], record.cache_read, warnings)

            results.append(
                ModelResult(
                    provider=self.name,
                    model_api_id=model_id,
                    model_region=region,
                    data=candidate,
                    warnings=list(dict.fromkeys(warnings)),
                )
            )
        return results

    @classmethod
    def _parse_records(
        cls, raw_data: str
    ) -> dict[str, set[_PricingRecord]]:
        parser = _TableParser()
        parser.feed(raw_data)
        records: dict[str, set[_PricingRecord]] = {}

        for table in parser.tables:
            model_row_index = next(
                (
                    index
                    for index, row in enumerate(table)
                    if row and cls._normalize(row[0]) == "model"
                ),
                None,
            )
            if model_row_index is None:
                continue

            source_names: list[str] = []
            for cell in table[model_row_index][1:]:
                match = re.search(r"deepseek-[a-z0-9.-]+", cell, re.I)
                if match:
                    source_names.append(match.group(0))
            if not source_names:
                continue

            prices: list[dict[str, Decimal | None]] = [
                {"input": None, "output": None, "cache_read": None}
                for _ in source_names
            ]
            current_field: str | None = None
            for row in table[model_row_index + 1 :]:
                compact_row = " ".join(row).upper()
                if "1M INPUT TOKENS" in compact_row and "CACHE HIT" in compact_row:
                    current_field = "cache_read"
                elif "1M INPUT TOKENS" in compact_row and "CACHE MISS" in compact_row:
                    current_field = "input"
                elif "1M OUTPUT TOKENS" in compact_row:
                    current_field = "output"

                peak_index = next(
                    (
                        index
                        for index, cell in enumerate(row)
                        if cls._normalize(cell) == "peak"
                    ),
                    None,
                )
                if peak_index is None or current_field is None:
                    continue
                raw_prices = row[peak_index + 1 :]
                if len(raw_prices) != len(source_names):
                    continue
                for index, value in enumerate(raw_prices):
                    prices[index][current_field] = cls._parse_price(value)

            for source_name, values in zip(source_names, prices):
                if all(value is None for value in values.values()):
                    continue
                record = _PricingRecord(source_name=source_name, **values)
                records.setdefault(cls._normalize(source_name), set()).add(record)

        return records

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(value.split()).casefold()

    @staticmethod
    def _parse_price(value: str) -> Decimal | None:
        match = re.fullmatch(r"\$\s*([0-9]+(?:\.[0-9]+)?)", value.strip())
        if match is None:
            return None
        try:
            return Decimal(match.group(1))
        except InvalidOperation:
            return None

    @staticmethod
    def _update_rates(
        rates: Any,
        price: Decimal | None,
        path: str,
        warnings: list[str],
    ) -> None:
        if rates is None:
            return
        if price is None:
            warnings.append(
                f"official peak price not found for {path}; retained existing values"
            )
            return
        if not isinstance(rates, list):
            warnings.append(f"{path} has an unexpected structure; retained existing value")
            return
        for rate in rates:
            if not isinstance(rate, dict) or "per_million" not in rate:
                warnings.append(
                    f"{path} has an unexpected structure; retained existing value"
                )
                continue
            rate["per_million"] = float(price)

    @classmethod
    def _update_cache(
        cls, groups: Any, cache_read: Decimal | None, warnings: list[str]
    ) -> None:
        if not isinstance(groups, list):
            warnings.append("cache has an unexpected structure; retained existing value")
            return
        for index, group in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"[{index}]"
            if not isinstance(group, dict):
                warnings.append(
                    f"cache{suffix} has an unexpected structure; retained existing value"
                )
                continue
            cls._update_rates(
                group.get("cache_read"),
                cache_read,
                f"cache{suffix}.cache_read",
                warnings,
            )
            if "cache_writes" in group:
                warnings.append(
                    f"official peak price not found for cache{suffix}.cache_writes; "
                    "retained existing values"
                )
