from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


@dataclass(frozen=True)
class _PricingRecord:
    source_name: str
    input: Decimal | None
    output: Decimal | None


class TypesafeProvider(BaseProvider):
    name = "Typesafe AI"

    MODEL_ALIASES: dict[str, tuple[str, ...]] = {
        "jev-latest": ("Jev",),
    }

    def fetch(self, source_url: str) -> str:
        landing_page = self._fetch_text(source_url)
        if self._parse_output_price(self._plain_text(landing_page)) is not None:
            return landing_page

        # The homepage publishes the input price directly and links to the
        # official Jev launch article that states output tokens are free.
        link_match = re.search(
            r'href=["\']([^"\']*introducing-system-one-models-and-jev[^"\']*)["\']',
            landing_page,
            re.I,
        )
        if link_match is None:
            return landing_page
        article_url = urljoin(source_url, link_match.group(1))
        if urlsplit(article_url).netloc != urlsplit(source_url).netloc:
            return landing_page
        return f"{landing_page}\n{self._fetch_text(article_url)}"

    @staticmethod
    def _fetch_text(url: str) -> str:
        request = Request(
            url,
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
        text = self._plain_text(raw_data)
        record = self._parse_record(text)
        if record is None:
            raise ValueError("Typesafe Jev pricing is empty")

        records = {self._normalize(record.source_name): record}
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
                matched = next(iter(matches))
                self._update_rates(
                    candidate.get("input"), matched.input, "input", warnings
                )
                self._update_rates(
                    candidate.get("output"), matched.output, "output", warnings
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
    def _parse_record(cls, text: str) -> _PricingRecord | None:
        if not re.search(r"\bJev\b", text, re.I):
            return None
        input_price = cls._parse_input_price(text)
        output_price = cls._parse_output_price(text)
        if input_price is None and output_price is None:
            return None
        return _PricingRecord("Jev", input_price, output_price)

    @staticmethod
    def _plain_text(raw_data: str) -> str:
        parser = _TextParser()
        parser.feed(raw_data)
        return parser.text

    @staticmethod
    def _parse_input_price(text: str) -> Decimal | None:
        matches = re.findall(
            r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:per|/)\s*"
            r"(?:one\s+)?billion\s+input\s+tokens?",
            text,
            re.I,
        )
        try:
            values = {Decimal(value) / Decimal(1000) for value in matches}
        except InvalidOperation:
            return None
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def _parse_output_price(text: str) -> Decimal | None:
        if re.search(r"output\s+tokens?\s*:?\s*free\b", text, re.I):
            return Decimal(0)
        return None

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(value.split()).casefold()

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
