from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult


@dataclass(frozen=True)
class _ContextPrices:
    input: Decimal | None
    cache_read: Decimal | None
    cache_writes: Decimal | None
    output: Decimal | None


@dataclass(frozen=True)
class _PricingRecord:
    source_name: str
    short: _ContextPrices
    long: _ContextPrices


class OpenAIProvider(BaseProvider):
    name = "OpenAI"

    # Only explicit, reviewed differences between model_api_id and the exact
    # name used by the official pricing table belong here.
    MODEL_ALIASES: dict[str, tuple[str, ...]] = {
        "gpt-5.4": ("gpt-5.4 (<272K context length)",),
        "gpt-5.5": ("gpt-5.5 (<272K context length)",),
    }

    _MODES = ("standard", "batch")
    def fetch(self, source_url: str) -> str:
        # OpenAI documents the Markdown representation as the same page with
        # `.md` appended.  The registry remains the owner of the source URL.
        markdown_url = source_url if source_url.endswith(".md") else f"{source_url}.md"
        request = Request(
            markdown_url,
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
        tables = {mode: self._parse_pricing_table(raw_data, mode) for mode in self._MODES}
        context_split = self._parse_context_split(raw_data)
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

            standard = self._match_record(model_id, tables["standard"], warnings)
            if standard is None:
                warnings.append("official price not found")
            else:
                self._update_price_group(
                    candidate,
                    standard,
                    context_split,
                    prefix="",
                    warnings=warnings,
                )

            if "batch" in candidate:
                batch = self._match_record(model_id, tables["batch"], warnings, mode="batch")
                if batch is None:
                    warnings.append("official batch price not found; retained existing values")
                elif isinstance(candidate["batch"], dict):
                    self._update_price_group(
                        candidate["batch"],
                        batch,
                        context_split,
                        prefix="batch.",
                        warnings=warnings,
                    )
                else:
                    warnings.append("batch has an unexpected structure; retained existing value")

            results.append(
                ModelResult(
                    provider=self.name,
                    model_api_id=model_id,
                    model_region=region,
                    data=candidate,
                    warnings=self._deduplicate(warnings),
                )
            )

        return results

    def _parse_pricing_table(
        self, raw_data: str, mode: str
    ) -> dict[str, list[_PricingRecord]]:
        heading = re.compile(rf"^###\s+{re.escape(mode)} pricing data\s*$", re.I | re.M)
        match = heading.search(raw_data)
        if match is None:
            raise ValueError(f"{mode} pricing table not found in official source")

        records: dict[str, list[_PricingRecord]] = {}
        table_started = False
        for line in raw_data[match.end() :].splitlines():
            stripped = line.strip()
            if not stripped:
                if table_started:
                    break
                continue
            if not stripped.startswith("|"):
                if table_started:
                    break
                continue
            table_started = True
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not cells or cells[0].lower() == "model" or self._is_separator_row(cells):
                continue
            if len(cells) != 9:
                raise ValueError(
                    f"unexpected {mode} pricing row with {len(cells)} columns: {stripped}"
                )
            record = _PricingRecord(
                source_name=cells[0],
                short=_ContextPrices(*[self._parse_price(value) for value in cells[1:5]]),
                long=_ContextPrices(*[self._parse_price(value) for value in cells[5:9]]),
            )
            records.setdefault(record.source_name, []).append(record)

        if not records:
            raise ValueError(f"{mode} pricing table is empty")
        return records

    @staticmethod
    def _is_separator_row(cells: list[str]) -> bool:
        return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)

    @staticmethod
    def _parse_price(value: str) -> Decimal | None:
        value = value.strip()
        if value in {"-", "—", ""}:
            return None
        match = re.fullmatch(r"\$([0-9]+(?:\.[0-9]+)?)", value.replace(",", ""))
        if match is None:
            raise ValueError(f"unexpected price value: {value}")
        try:
            return Decimal(match.group(1))
        except InvalidOperation as exc:
            raise ValueError(f"invalid price value: {value}") from exc

    @staticmethod
    def _parse_context_split(raw_data: str) -> int | None:
        match = re.search(
            r"Short context:\s*(?:≤|<=)\s*([0-9]+(?:\.[0-9]+)?)K\s+input tokens",
            raw_data,
            re.I,
        )
        if match is None:
            return None
        return int(Decimal(match.group(1)) * 1000)

    def _match_record(
        self,
        model_id: str,
        records: dict[str, list[_PricingRecord]],
        warnings: list[str],
        mode: str = "standard",
    ) -> _PricingRecord | None:
        names = (model_id, *self.MODEL_ALIASES.get(model_id, ()))
        matches = [record for name in names for record in records.get(name, [])]
        if len(matches) > 1:
            warnings.append(
                f"multiple official {mode} price records matched; retained existing values"
            )
            return None
        return matches[0] if matches else None

    def _update_price_group(
        self,
        target: dict[str, Any],
        record: _PricingRecord,
        context_split: int | None,
        prefix: str,
        warnings: list[str],
    ) -> None:
        self._update_rate_list(
            target.get("input"), record, "input", context_split, f"{prefix}input", warnings
        )
        self._update_rate_list(
            target.get("output"), record, "output", context_split, f"{prefix}output", warnings
        )

        if "cache" not in target:
            return
        cache_groups = target["cache"]
        if not isinstance(cache_groups, list):
            warnings.append(f"{prefix}cache has an unexpected structure; retained existing value")
            return
        for index, cache_group in enumerate(cache_groups):
            suffix = "" if len(cache_groups) == 1 else f"[{index}]"
            if not isinstance(cache_group, dict):
                warnings.append(
                    f"{prefix}cache{suffix} has an unexpected structure; retained existing value"
                )
                continue
            self._update_rate_list(
                cache_group.get("cache_read"),
                record,
                "cache_read",
                context_split,
                f"{prefix}cache{suffix}.cache_read",
                warnings,
            )
            self._update_rate_list(
                cache_group.get("cache_writes"),
                record,
                "cache_writes",
                context_split,
                f"{prefix}cache{suffix}.cache_writes",
                warnings,
            )

    def _update_rate_list(
        self,
        rates: Any,
        record: _PricingRecord,
        price_field: str,
        context_split: int | None,
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
                warnings.append(f"{path} has an unexpected structure; retained existing value")
                continue
            range_label = self._range_label(rate)
            short_value = getattr(record.short, price_field)
            long_value = getattr(record.long, price_field)
            if (
                len(rates) == 1
                and rate.get("context_min") == 0
                and rate.get("context_max") is None
            ):
                if (
                    short_value is not None
                    and long_value is not None
                    and long_value != short_value
                ):
                    warnings.append(
                        f"official short/long prices cannot map to {path}{range_label}; "
                        "retained existing value"
                    )
                    continue
                value = short_value
            else:
                context_kind = self._context_kind(rate, context_split)
                if context_kind is None:
                    warnings.append(
                        f"cannot map official price to {path}{range_label}; retained existing value"
                    )
                    continue
                value = short_value if context_kind == "short" else long_value
            if value is None:
                warnings.append(
                    f"official price unavailable for {path}{range_label}; retained existing value"
                )
                continue
            rate["per_million"] = float(value)

    @staticmethod
    def _context_kind(
        rate: dict[str, Any], context_split: int | None
    ) -> str | None:
        minimum = rate.get("context_min")
        maximum = rate.get("context_max")
        if context_split is None:
            return None
        if minimum == 0 and maximum == context_split:
            return "short"
        if minimum == context_split and maximum is None:
            return "long"
        return None

    @staticmethod
    def _range_label(rate: dict[str, Any]) -> str:
        minimum = rate.get("context_min")
        maximum = rate.get("context_max")
        return f"[{minimum}-{maximum if maximum is not None else '∞'}]"

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))
