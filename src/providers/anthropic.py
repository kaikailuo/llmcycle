from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult
from src.utils.http import urlopen_with_retry


@dataclass(frozen=True)
class _PricingRecord:
    source_name: str
    input: Decimal | None = None
    output: Decimal | None = None
    cache_read: Decimal | None = None
    cache_write_5m: Decimal | None = None
    cache_write_1h: Decimal | None = None
    effective_from: date | None = None
    effective_through: date | None = None


class AnthropicProvider(BaseProvider):
    name = "Anthropic"

    MODEL_ALIASES: dict[str, tuple[str, ...]] = {
        "claude-haiku-4-5": ("Claude Haiku 4.5",),
        "claude-opus-4-6": ("Claude Opus 4.6",),
        "claude-opus-4-7": ("Claude Opus 4.7",),
        "claude-sonnet-5": ("Claude Sonnet 5",),
        "claude-sonnet-5-5": ("Claude Sonnet 5.5",),
    }

    _MONTH_DATE = (
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},\s+\d{4}"
    )
    _DATE_TEXT = rf"(?:{_MONTH_DATE}|\d{{4}}-\d{{2}}-\d{{2}})"

    def fetch(self, source_url: str) -> str:
        markdown_url = source_url if source_url.endswith(".md") else f"{source_url}.md"
        request = Request(
            markdown_url,
            headers={
                "Accept": "text/markdown, text/plain;q=0.9",
                "User-Agent": "llmcycle-pricing-maintainer/1.0",
            },
        )
        with urlopen_with_retry(request, timeout=30, opener=urlopen) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding)

    def parse(
        self, raw_data: str, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        standard_records = self._parse_table(raw_data, "Model pricing")
        batch_records = self._parse_table(raw_data, "Batch processing")
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

            standard = self._match_record(model_id, standard_records, warnings)
            if standard is None:
                if not warnings:
                    warnings.append("official price not found")
            else:
                self._update_group(candidate, standard, "", warnings)
                if "batch" in candidate:
                    batch = self._match_record(
                        model_id, batch_records, warnings, mode="batch"
                    )
                    if batch is None:
                        if not any("batch" in warning for warning in warnings):
                            warnings.append(
                                "official batch price not found; retained existing values"
                            )
                    elif isinstance(candidate["batch"], dict):
                        batch = self._with_batch_cache(standard, batch, raw_data)
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

    def _parse_table(
        self, raw_data: str, heading: str
    ) -> dict[str, list[_PricingRecord]]:
        heading_match = re.search(
            rf"^##+\s+{re.escape(heading)}\s*$", raw_data, re.I | re.M
        )
        if heading_match is None:
            raise ValueError(f"{heading.lower()} table not found in official source")

        table_lines: list[str] = []
        started = False
        for line in raw_data[heading_match.end() :].splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                started = True
                table_lines.append(stripped)
            elif started:
                break

        if len(table_lines) < 3:
            raise ValueError(f"{heading.lower()} table is empty")

        headers = [self._clean_text(cell) for cell in self._split_row(table_lines[0])]
        records: dict[str, list[_PricingRecord]] = {}
        for line in table_lines[2:]:
            cells = self._split_row(line)
            if len(cells) != len(headers):
                continue
            values = dict(zip(headers, cells))
            raw_name = values.get("model") or values.get("name")
            if raw_name is None:
                raise ValueError(f"model column not found in {heading.lower()} table")
            source_name, starts, through = self._parse_effective_window(raw_name)
            starts = starts or self._date_for(
                values, "effective from", "effective date", "starting"
            )
            through = through or self._date_for(
                values, "effective through", "through", "end date"
            )
            record = _PricingRecord(
                source_name=source_name,
                input=self._price_for(
                    values,
                    "base input tokens",
                    "base input",
                    "batch input",
                    "input",
                ),
                output=self._price_for(values, "output tokens", "batch output", "output"),
                cache_read=self._price_for(
                    values, "cache hits and refreshes", "cache hit", "cache read"
                ),
                cache_write_5m=self._price_for(
                    values, "5m cache writes", "cache write 5m", "5m cache write"
                ),
                cache_write_1h=self._price_for(
                    values, "1h cache writes", "cache write 1h", "1h cache write"
                ),
                effective_from=starts,
                effective_through=through,
            )
            records.setdefault(source_name, []).append(record)

        if not records:
            raise ValueError(f"{heading.lower()} table is empty")
        return records

    @staticmethod
    def _split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    @staticmethod
    def _clean_text(value: str) -> str:
        value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
        value = re.sub(r"<[^>]+>", "", value)
        return re.sub(r"\s+", " ", value).strip().lower()

    @classmethod
    def _parse_effective_window(
        cls, raw_name: str
    ) -> tuple[str, date | None, date | None]:
        cleaned = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", raw_name)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        starts = cls._extract_date(
            cleaned, rf"(?:starting|effective(?:\s+on)?)\s+({cls._DATE_TEXT})"
        )
        through = cls._extract_date(cleaned, rf"through\s+({cls._DATE_TEXT})")
        cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip(), starts, through

    @staticmethod
    def _extract_date(value: str, pattern: str) -> date | None:
        match = re.search(pattern, value, re.I)
        if match is None:
            return None
        raw_date = match.group(1)
        date_format = (
            "%Y-%m-%d"
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date)
            else "%B %d, %Y"
        )
        return datetime.strptime(raw_date, date_format).date()

    @classmethod
    def _date_for(cls, values: dict[str, str], *headers: str) -> date | None:
        for header in headers:
            if header not in values:
                continue
            match = re.search(rf"({cls._DATE_TEXT})", values[header], re.I)
            if match is not None:
                return cls._extract_date(match.group(1), rf"({cls._DATE_TEXT})")
        return None

    @classmethod
    def _price_for(cls, values: dict[str, str], *headers: str) -> Decimal | None:
        for header in headers:
            if header in values:
                return cls._parse_price(values[header])
        return None

    @staticmethod
    def _parse_price(value: str) -> Decimal | None:
        if value.strip() in {"", "-", "—"}:
            return None
        match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", value.replace(",", ""))
        if match is None:
            return None
        return Decimal(match.group(1))

    def _match_record(
        self,
        model_id: str,
        records: dict[str, list[_PricingRecord]],
        warnings: list[str],
        mode: str = "standard",
    ) -> _PricingRecord | None:
        names = (model_id, *self.MODEL_ALIASES.get(model_id, ()))
        matches = [record for name in names for record in records.get(name, [])]
        if not matches:
            return None
        selected = self._select_active_record(matches)
        if selected is None:
            warnings.append(
                f"multiple official {mode} price records matched; retained existing values"
            )
        return selected

    @staticmethod
    def _select_active_record(
        records: list[_PricingRecord], on_date: date | None = None
    ) -> _PricingRecord | None:
        on_date = on_date or date.today()
        active = [
            record
            for record in records
            if (record.effective_from is None or record.effective_from <= on_date)
            and (record.effective_through is None or on_date <= record.effective_through)
        ]
        if not active:
            return None
        latest_start = max(record.effective_from or date.min for record in active)
        latest = [
            record
            for record in active
            if (record.effective_from or date.min) == latest_start
        ]
        return latest[0] if len(latest) == 1 else None

    @staticmethod
    def _with_batch_cache(
        standard: _PricingRecord, batch: _PricingRecord, raw_data: str
    ) -> _PricingRecord:
        if batch.cache_read is not None or batch.cache_write_5m is not None:
            return batch
        if standard.input is None or standard.input == 0 or batch.input is None:
            return batch
        if re.search(
            r"multipliers stack with other pricing modifiers, including the Batch API discount",
            raw_data,
            re.I,
        ) is None:
            return batch
        factor = batch.input / standard.input
        return _PricingRecord(
            source_name=batch.source_name,
            input=batch.input,
            output=batch.output,
            cache_read=(
                standard.cache_read * factor
                if standard.cache_read is not None
                else None
            ),
            cache_write_5m=(
                standard.cache_write_5m * factor
                if standard.cache_write_5m is not None
                else None
            ),
            cache_write_1h=(
                standard.cache_write_1h * factor
                if standard.cache_write_1h is not None
                else None
            ),
            effective_from=batch.effective_from,
            effective_through=batch.effective_through,
        )

    def _update_group(
        self,
        target: dict[str, Any],
        record: _PricingRecord,
        prefix: str,
        warnings: list[str],
    ) -> None:
        self._update_rates(target, "input", record.input, prefix, warnings)
        self._update_rates(target, "output", record.output, prefix, warnings)
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
                group,
                "cache_read",
                record.cache_read,
                f"{prefix}cache{suffix}.",
                warnings,
            )
            self._update_cache_writes(
                group.get("cache_writes"),
                record,
                f"{prefix}cache{suffix}.cache_writes",
                warnings,
            )

    @staticmethod
    def _update_rates(
        target: dict[str, Any],
        key: str,
        value: Decimal | None,
        prefix: str,
        warnings: list[str],
    ) -> None:
        if key not in target:
            return
        path = f"{prefix}{key}"
        rates = target[key]
        if value is None:
            warnings.append(f"official price not found for {path}; retained existing values")
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
            rate["per_million"] = float(value)

    @staticmethod
    def _update_cache_writes(
        rates: Any,
        record: _PricingRecord,
        path: str,
        warnings: list[str],
    ) -> None:
        if rates is None:
            return
        if not isinstance(rates, list):
            warnings.append(f"{path} has an unexpected structure; retained existing value")
            return
        for rate in rates:
            if not isinstance(rate, dict):
                warnings.append(f"{path} has an unexpected structure; retained existing value")
                continue
            for key, value in (
                ("5m_per_million", record.cache_write_5m),
                ("1h_per_million", record.cache_write_1h),
            ):
                if key not in rate:
                    continue
                if value is None:
                    warnings.append(
                        f"official price not found for {path}.{key}; retained existing value"
                    )
                else:
                    rate[key] = float(value)
