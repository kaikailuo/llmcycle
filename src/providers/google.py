from __future__ import annotations

import copy
import html
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "tr":
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
            if self._row:
                self.rows.append(self._row)
            self._row = None


@dataclass(frozen=True)
class _TieredPrice:
    short: Decimal | None
    long: Decimal | None = None
    split: int | None = None


@dataclass(frozen=True)
class _ModePrices:
    input: _TieredPrice | None = None
    output: _TieredPrice | None = None
    cache_read: _TieredPrice | None = None


class GoogleProvider(BaseProvider):
    name = "Google"

    def fetch(self, source_url: str) -> str:
        request = Request(
            source_url,
            headers={
                "Accept": "text/html, text/plain;q=0.9",
                "User-Agent": "llmcycle-pricing-maintainer/1.0",
            },
        )
        with urlopen(request, timeout=30) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding)

    def parse(
        self, raw_data: str, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
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

            section = self._model_section(raw_data, model_id)
            if section is None:
                warnings.append("official price not found")
            else:
                standard = self._parse_mode(section, "Standard")
                if standard is None:
                    warnings.append("official price not found")
                else:
                    self._update_group(candidate, standard, "", warnings)
                if "batch" in candidate:
                    batch = self._parse_mode(section, "Batch")
                    if batch is None:
                        warnings.append(
                            "official batch price not found; retained existing values"
                        )
                    elif isinstance(candidate["batch"], dict):
                        self._update_group(
                            candidate["batch"], batch, "batch.", warnings
                        )
                    else:
                        warnings.append(
                            "batch has an unexpected structure; retained existing value"
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

    @staticmethod
    def _model_section(raw_data: str, model_id: str) -> str | None:
        heading = re.search(
            rf"<h2\b[^>]*\bid=[\"']{re.escape(model_id)}[\"'][^>]*>",
            raw_data,
            re.I,
        )
        if heading is None:
            return None
        next_heading = re.search(r"<h2\b", raw_data[heading.end() :], re.I)
        end = (
            heading.end() + next_heading.start()
            if next_heading is not None
            else len(raw_data)
        )
        return raw_data[heading.start() : end]

    @classmethod
    def _parse_mode(cls, section: str, mode: str) -> _ModePrices | None:
        heading = re.search(
            rf"<h3\b[^>]*>\s*{re.escape(mode)}\s*</h3>", section, re.I
        )
        if heading is None:
            return None
        remainder = section[heading.end() :]
        next_heading = re.search(r"<h3\b", remainder, re.I)
        mode_section = remainder[: next_heading.start()] if next_heading else remainder
        table_match = re.search(r"<table\b[^>]*>.*?</table>", mode_section, re.I | re.S)
        if table_match is None:
            return None
        parser = _TableParser()
        parser.feed(table_match.group(0))
        values = {
            row[0].strip().lower(): row[-1]
            for row in parser.rows
            if len(row) >= 2 and row[0].strip()
        }
        return _ModePrices(
            input=cls._parse_tiered_price(values.get("input price")),
            output=cls._parse_tiered_price(
                values.get("output price (including thinking tokens)")
                or values.get("output price")
            ),
            cache_read=cls._parse_tiered_price(
                values.get("context caching price")
            ),
        )

    @staticmethod
    def _parse_tiered_price(value: str | None) -> _TieredPrice | None:
        if value is None:
            return None
        value = html.unescape(value)
        price_matches = list(
            re.finditer(r"\$([0-9]+(?:\.[0-9]+)?)", value.replace(",", ""))
        )
        if not price_matches:
            return None
        try:
            prices = [Decimal(match.group(1)) for match in price_matches]
        except InvalidOperation as exc:
            raise ValueError(f"invalid price value: {value}") from exc

        tier_match = re.search(r"(?:<=|≤)\s*([0-9]+(?:\.[0-9]+)?)\s*[kK]", value)
        if tier_match is None:
            return _TieredPrice(short=prices[0])
        if len(prices) < 2:
            return None
        split = int(Decimal(tier_match.group(1)) * 1000)
        return _TieredPrice(short=prices[0], long=prices[1], split=split)

    def _update_group(
        self,
        target: dict[str, Any],
        prices: _ModePrices,
        prefix: str,
        warnings: list[str],
    ) -> None:
        self._update_rates(target.get("input"), prices.input, f"{prefix}input", warnings)
        self._update_rates(
            target.get("output"), prices.output, f"{prefix}output", warnings
        )
        if "cache" not in target:
            return
        groups = target["cache"]
        if not isinstance(groups, list):
            warnings.append(f"{prefix}cache has an unexpected structure; retained existing value")
            return
        for index, group in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"[{index}]"
            if not isinstance(group, dict):
                warnings.append(
                    f"{prefix}cache{suffix} has an unexpected structure; retained existing value"
                )
                continue
            self._update_rates(
                group.get("cache_read"),
                prices.cache_read,
                f"{prefix}cache{suffix}.cache_read",
                warnings,
            )

    @staticmethod
    def _update_rates(
        rates: Any,
        price: _TieredPrice | None,
        path: str,
        warnings: list[str],
    ) -> None:
        if rates is None:
            return
        if price is None or price.short is None:
            warnings.append(f"official price not found for {path}; retained existing values")
            return
        if not isinstance(rates, list):
            warnings.append(f"{path} has an unexpected structure; retained existing value")
            return

        for rate in rates:
            if not isinstance(rate, dict) or "per_million" not in rate:
                warnings.append(f"{path} has an unexpected structure; retained existing value")
                continue
            if price.split is None:
                value = price.short
            elif rate.get("context_min") == 0 and rate.get("context_max") == price.split:
                value = price.short
            elif (
                rate.get("context_min") == price.split
                and rate.get("context_max") is None
            ):
                value = price.long
            else:
                minimum = rate.get("context_min")
                maximum = rate.get("context_max")
                label = f"[{minimum}-{maximum if maximum is not None else '∞'}]"
                warnings.append(
                    f"cannot map official context tier to {path}{label}; retained existing value"
                )
                continue
            if value is None:
                warnings.append(
                    f"official price not found for {path}; retained existing value"
                )
                continue
            rate["per_million"] = float(value)
