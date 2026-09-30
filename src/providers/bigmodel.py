from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult
from src.utils.currency import cny_to_usd


_Tier = tuple[int, int | None]


@dataclass
class _PricingRecord:
    source_name: str
    input: dict[_Tier, set[Decimal]] = field(default_factory=dict)
    output: dict[_Tier, set[Decimal]] = field(default_factory=dict)
    cache_read: dict[_Tier, set[Decimal]] = field(default_factory=dict)


class BigModelProvider(BaseProvider):
    name = "BigModel"

    MODEL_ALIASES: dict[str, tuple[str, ...]] = {
        "glm-4.6v": ("GLM-4.6V",),
        "glm-5.1": ("GLM-5.1",),
        "glm-5.2": ("GLM-5.2",),
    }

    _USD_QUANTUM = Decimal("0.01")

    def fetch(self, source_url: str) -> str:
        request = Request(
            source_url,
            headers={
                "Accept": "text/markdown, text/plain;q=0.9",
                "User-Agent": "llmcycle-pricing-maintainer/1.0",
            },
        )
        with urlopen(request, timeout=30) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding)

    def parse(
        self, raw_data: str, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        records = self._parse_records(raw_data)
        if not records:
            raise ValueError("BigModel pricing table is empty")

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
                id(records[self._normalize(name)]): records[self._normalize(name)]
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
                record = next(iter(matches.values()))
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
    def _parse_records(cls, raw_data: str) -> dict[str, _PricingRecord]:
        records: dict[str, _PricingRecord] = {}
        lines = raw_data.splitlines()
        for index, line in enumerate(lines):
            if not line.lstrip().startswith("|"):
                continue
            headers = [cls._compact(cell) for cell in cls._split_row(line)]
            name_index = cls._column(headers, "模型名称")
            context_index = cls._column(headers, "上下文")
            input_index = cls._column(headers, "输入单价")
            output_index = cls._column(headers, "输出单价")
            cache_index = cls._column(headers, "缓存命中")
            if None in {
                name_index,
                context_index,
                input_index,
                output_index,
                cache_index,
            }:
                continue
            if index + 1 >= len(lines) or not cls._is_separator(lines[index + 1]):
                continue

            for row_line in lines[index + 2 :]:
                if not row_line.lstrip().startswith("|"):
                    break
                cells = cls._split_row(row_line)
                if len(cells) != len(headers):
                    continue
                source_name = cls._clean_cell(cells[name_index])
                tier = cls._parse_context(cells[context_index])
                if not source_name or tier is None:
                    continue
                record = records.setdefault(
                    cls._normalize(source_name), _PricingRecord(source_name)
                )
                for field_name, column_index in (
                    ("input", input_index),
                    ("output", output_index),
                    ("cache_read", cache_index),
                ):
                    price = cls._parse_price(cells[column_index])
                    if price is not None:
                        getattr(record, field_name).setdefault(tier, set()).add(price)

        return records

    @staticmethod
    def _split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    @staticmethod
    def _clean_cell(value: str) -> str:
        value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
        value = re.sub(r"<[^>]+>", " ", value)
        return " ".join(value.split())

    @classmethod
    def _compact(cls, value: str) -> str:
        return "".join(cls._clean_cell(value).split()).casefold()

    @classmethod
    def _normalize(cls, value: str) -> str:
        return cls._compact(value)

    @staticmethod
    def _column(headers: list[str], prefix: str) -> int | None:
        normalized = prefix.casefold()
        return next(
            (index for index, header in enumerate(headers) if normalized in header),
            None,
        )

    @classmethod
    def _is_separator(cls, line: str) -> bool:
        cells = cls._split_row(line)
        return bool(cells) and all(re.fullmatch(r":?-+:?", cell) for cell in cells)

    @staticmethod
    def _token_count(value: str, unit: str) -> int:
        multiplier = {"": 1, "k": 1000, "m": 1_000_000}[unit.casefold()]
        return int(Decimal(value) * multiplier)

    @classmethod
    def _parse_context(cls, value: str) -> _Tier | None:
        cleaned = cls._clean_cell(value).replace("\\", "")
        range_match = re.search(
            r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*([KM]?)\s*,\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*([KM]?)\s*\)",
            cleaned,
            re.I,
        )
        if range_match:
            return (
                cls._token_count(range_match.group(1), range_match.group(2)),
                cls._token_count(range_match.group(3), range_match.group(4)),
            )
        lower_match = re.search(
            r"(?:≥|>=)\s*([0-9]+(?:\.[0-9]+)?)\s*([KM]?)", cleaned, re.I
        )
        if lower_match:
            return (
                cls._token_count(lower_match.group(1), lower_match.group(2)),
                None,
            )
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?\s*[KM]", cleaned, re.I):
            return (0, None)
        return None

    @staticmethod
    def _parse_price(value: str) -> Decimal | None:
        cleaned = BigModelProvider._clean_cell(value).replace(",", "")
        if cleaned in {"免费", "限时免费"}:
            return Decimal(0)
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    @classmethod
    def _to_usd_float(cls, value: Decimal) -> float:
        return float(
            cny_to_usd(value).quantize(cls._USD_QUANTUM, rounding=ROUND_HALF_UP)
        )

    @classmethod
    def _update_rates(
        cls,
        rates: Any,
        prices: dict[_Tier, set[Decimal]],
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
            values = prices.get(tier, set())
            maximum = tier[1] if tier[1] is not None else "∞"
            if not values:
                warnings.append(
                    f"official price not found for {path}[{tier[0]}-{maximum}]; "
                    "retained existing value"
                )
                continue
            if len(values) > 1:
                warnings.append(
                    f"multiple official prices found for {path}[{tier[0]}-{maximum}]; "
                    "retained existing value"
                )
                continue
            rate["per_million"] = cls._to_usd_float(next(iter(values)))

    @classmethod
    def _update_cache(
        cls,
        groups: Any,
        prices: dict[_Tier, set[Decimal]],
        warnings: list[str],
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
                prices,
                f"cache{suffix}.cache_read",
                warnings,
            )
            if "cache_writes" in group:
                warnings.append(
                    f"official price not found for cache{suffix}.cache_writes; "
                    "retained existing values"
                )
