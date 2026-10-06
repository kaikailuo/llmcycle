from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult
from src.utils.currency import cny_to_usd
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
            self._cell.append("\n")

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


@dataclass
class _PricingRecord:
    source_name: str
    input: dict[tuple[int, int | None], Decimal] = field(default_factory=dict)
    output: dict[tuple[int, int | None], Decimal] = field(default_factory=dict)


class BaiduProvider(BaseProvider):
    name = "Baidu ERNIE"

    MODEL_ALIASES: dict[str, tuple[str, ...]] = {
        "ernie-5.1": ("ERNIE-5.1",),
    }

    _USD_QUANTUM = Decimal("0.01")

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
            raise ValueError("Baidu pricing table is empty")

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
                id(records[name.casefold()]): records[name.casefold()]
                for name in names
                if name.casefold() in records
            }
            if not matches:
                warnings.append("official price not found")
            elif len(matches) > 1:
                warnings.append(
                    "multiple official price records matched; retained existing values"
                )
            else:
                record = next(iter(matches.values()))
                self._update_rates(
                    candidate.get("input"), record.input, "input", warnings
                )
                self._update_rates(
                    candidate.get("output"), record.output, "output", warnings
                )

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
    def _parse_records(cls, raw_data: str) -> dict[str, _PricingRecord]:
        parser = _TableParser()
        parser.feed(raw_data)
        records: dict[str, _PricingRecord] = {}

        for table in parser.tables:
            current: _PricingRecord | None = None
            unit = ""
            for row in table:
                if len(row) >= 7:
                    source_names = re.findall(r"ERNIE-[A-Za-z0-9.-]+", row[1], re.I)
                    if not source_names:
                        current = None
                        continue
                    current = _PricingRecord(source_name=source_names[0])
                    for source_name in source_names:
                        records[source_name.casefold()] = current
                    label, raw_price, unit = row[3], row[4], row[6]
                elif current is not None and len(row) >= 2:
                    label, raw_price = row[0], row[1]
                else:
                    continue

                parsed = cls._parse_label(label)
                price = cls._parse_price(raw_price, unit)
                if parsed is None or price is None:
                    continue
                field_name, tier = parsed
                getattr(current, field_name)[tier] = price

        return records

    @staticmethod
    def _parse_label(value: str) -> tuple[str, tuple[int, int | None]] | None:
        compact = value.replace("（", "(").replace("）", ")").replace(" ", "")
        field_name = "input" if compact.startswith("输入") else "output" if compact.startswith("输出") else None
        if field_name is None:
            return None

        two_bounds = re.search(
            r"([0-9]+(?:\.[0-9]+)?)k<输入(?:<=|=<)([0-9]+(?:\.[0-9]+)?)k",
            compact,
            re.I,
        )
        if two_bounds:
            return field_name, (
                int(Decimal(two_bounds.group(1)) * 1000),
                int(Decimal(two_bounds.group(2)) * 1000),
            )
        upper = re.search(
            r"输入(?:<=|=<)([0-9]+(?:\.[0-9]+)?)k", compact, re.I
        )
        if upper:
            return field_name, (0, int(Decimal(upper.group(1)) * 1000))
        return None

    @staticmethod
    def _parse_price(value: str, unit: str) -> Decimal | None:
        if "元/千" not in unit.replace(" ", ""):
            return None
        try:
            # The source is CNY per thousand tokens; the project stores prices
            # per million tokens.
            return Decimal(value.replace(",", "")) * 1000
        except InvalidOperation:
            return None

    @classmethod
    def _update_rates(
        cls,
        rates: Any,
        prices: dict[tuple[int, int | None], Decimal],
        path: str,
        warnings: list[str],
    ) -> None:
        if rates is None:
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
            tier = (rate.get("context_min"), rate.get("context_max"))
            price = prices.get(tier)
            if price is None:
                maximum = tier[1] if tier[1] is not None else "∞"
                warnings.append(
                    f"official price not found for {path}[{tier[0]}-{maximum}]; "
                    "retained existing value"
                )
                continue
            usd = cny_to_usd(price).quantize(
                cls._USD_QUANTUM, rounding=ROUND_HALF_UP
            )
            rate["per_million"] = float(usd)
