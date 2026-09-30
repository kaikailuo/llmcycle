from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult


@dataclass(frozen=True)
class _PricingRecord:
    source_name: str
    input: Decimal | None
    output: Decimal | None
    cache_read: Decimal | None
    cache_writes: Decimal | None


class MiniMaxProvider(BaseProvider):
    name = "MiniMax"

    MODEL_ALIASES: dict[str, tuple[str, ...]] = {}

    def fetch(self, source_url: str) -> str:
        landing_page = self._fetch_text(source_url, "text/html, text/plain;q=0.9")
        if self._contains_llm_pricing(landing_page):
            return landing_page

        # The pricing landing page links to this same-origin full pay-as-you-go
        # document for current and legacy model prices. Prefer a link exposed in
        # the page, with the documented path as a same-origin fallback when the
        # landing page is client-rendered.
        link_match = re.search(
            r'href=["\']([^"\']*pricing-paygo(?:\.md)?)["\']',
            landing_page,
            re.I,
        )
        if link_match:
            pricing_url = urljoin(source_url, link_match.group(1))
        else:
            parts = urlsplit(source_url)
            pricing_url = urlunsplit(
                (parts.scheme, parts.netloc, "/docs/guides/pricing-paygo.md", "", "")
            )
        if not pricing_url.endswith(".md"):
            pricing_url = f"{pricing_url}.md"
        return self._fetch_text(pricing_url, "text/markdown, text/plain;q=0.9")

    @staticmethod
    def _fetch_text(url: str, accept: str) -> str:
        request = Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": "llmcycle-pricing-maintainer/1.0",
            },
        )
        with urlopen(request, timeout=30) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding)

    @staticmethod
    def _contains_llm_pricing(raw_data: str) -> bool:
        return (
            "MiniMax-M2.7" in raw_data
            and "MiniMax-M2.5" in raw_data
            and ("Prompt caching" in raw_data or "Cache read" in raw_data)
        )

    def parse(
        self, raw_data: str, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        current_records, legacy_records = self._parse_sections(raw_data)
        records = {
            **current_records,
            **legacy_records,
        }
        if not records:
            raise ValueError("MiniMax LLM pricing tables are empty")

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
                records[self._normalize(name)]
                for name in names
                if self._normalize(name) in records
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
                    self._update_cache(candidate["cache"], record, warnings)

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
    def _parse_sections(
        cls, raw_data: str
    ) -> tuple[dict[str, _PricingRecord], dict[str, _PricingRecord]]:
        llm_match = re.search(r"^##\s+LLM\s*$", raw_data, re.I | re.M)
        if llm_match:
            remainder = raw_data[llm_match.end() :]
            next_section = re.search(r"^##\s+", remainder, re.M)
            llm_section = (
                remainder[: next_section.start()] if next_section else remainder
            )
        else:
            llm_section = raw_data

        legacy_match = re.search(
            r'<(?:Accordion|details)\b[^>]*(?:title=["\']Legacy Models["\']|[^>]*legacy-models)[^>]*>(.*?)</(?:Accordion|details)>',
            llm_section,
            re.I | re.S,
        )
        legacy_section = legacy_match.group(1) if legacy_match else ""
        current_section = (
            llm_section[: legacy_match.start()] if legacy_match else llm_section
        )
        return cls._parse_tables(current_section), cls._parse_tables(legacy_section)

    @classmethod
    def _parse_tables(cls, section: str) -> dict[str, _PricingRecord]:
        records: dict[str, _PricingRecord] = {}
        lines = section.splitlines()
        for index, line in enumerate(lines):
            if not line.lstrip().startswith("|"):
                continue
            headers = [cls._clean_cell(cell) for cell in cls._split_row(line)]
            normalized_headers = [header.casefold() for header in headers]
            if not {"model", "input", "output"}.issubset(normalized_headers):
                continue
            if index + 1 >= len(lines) or not cls._is_separator(lines[index + 1]):
                continue

            positions = {name: normalized_headers.index(name) for name in ("model", "input", "output")}
            cache_read_index = next(
                (
                    i
                    for i, header in enumerate(normalized_headers)
                    if "caching read" in header or "cache read" in header
                ),
                None,
            )
            cache_write_index = next(
                (
                    i
                    for i, header in enumerate(normalized_headers)
                    if "caching write" in header or "cache write" in header
                ),
                None,
            )
            for row_line in lines[index + 2 :]:
                if not row_line.lstrip().startswith("|"):
                    break
                cells = cls._split_row(row_line)
                if len(cells) != len(headers):
                    continue
                source_name = cls._model_name(cells[positions["model"]])
                if not source_name:
                    continue
                record = _PricingRecord(
                    source_name=source_name,
                    input=cls._parse_price(cells[positions["input"]]),
                    output=cls._parse_price(cells[positions["output"]]),
                    cache_read=(
                        cls._parse_price(cells[cache_read_index])
                        if cache_read_index is not None
                        else None
                    ),
                    cache_writes=(
                        cls._parse_price(cells[cache_write_index])
                        if cache_write_index is not None
                        else None
                    ),
                )
                records[cls._normalize(source_name)] = record
        return records

    @staticmethod
    def _split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    @classmethod
    def _clean_cell(cls, value: str) -> str:
        value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
        value = re.sub(r"<[^>]+>", " ", value)
        return " ".join(unescape(value).split())

    @staticmethod
    def _is_separator(line: str) -> bool:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return bool(cells) and all(re.fullmatch(r":?-+:?", cell) for cell in cells)

    @classmethod
    def _model_name(cls, value: str) -> str:
        first_line = re.split(r"<br\s*/?>", value, maxsplit=1, flags=re.I)[0]
        return cls._clean_cell(first_line).strip("* ")

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(value.split()).casefold()

    @staticmethod
    def _parse_price(value: str) -> Decimal | None:
        value = re.sub(r"~~.*?~~", "", value)
        value = value.replace(r"\$", "$").replace(",", "")
        match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", value)
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
                f"official price not found for {path}; retained existing values"
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
        cls, groups: Any, record: _PricingRecord, warnings: list[str]
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
                record.cache_read,
                f"cache{suffix}.cache_read",
                warnings,
            )
            cls._update_rates(
                group.get("cache_writes"),
                record.cache_writes,
                f"cache{suffix}.cache_writes",
                warnings,
            )
