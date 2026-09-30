from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult


class _ModelCardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: dict[str, dict[str, str]] = {}
        self._div_depth = 0
        self._card_depth: int | None = None
        self._pricing_depth: int | None = None
        self._in_heading = False
        self._in_price_span = False
        self._heading_parts: list[str] = []
        self._span_parts: list[str] = []
        self._price_parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "div":
            self._div_depth += 1
            if self._card_depth is None and "home-card" in classes:
                self._card_depth = self._div_depth
                self._heading_parts = []
                self._price_parts = []
            if self._card_depth is not None and "home-card-pricing" in classes:
                self._pricing_depth = self._div_depth
        elif tag == "h3" and self._card_depth is not None:
            self._in_heading = True
        elif tag == "span" and self._pricing_depth is not None:
            self._in_price_span = True
            self._span_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading_parts.append(data)
        if self._in_price_span:
            self._span_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h3" and self._in_heading:
            self._in_heading = False
        elif tag == "span" and self._in_price_span:
            self._in_price_span = False
            text = " ".join("".join(self._span_parts).split())
            if text:
                self._price_parts.append(text)
        elif tag == "div":
            if self._pricing_depth == self._div_depth:
                self._pricing_depth = None
            if self._card_depth == self._div_depth:
                name = " ".join("".join(self._heading_parts).split())
                prices = {
                    self._price_parts[index]: self._price_parts[index + 1]
                    for index in range(0, len(self._price_parts) - 1, 2)
                }
                if name:
                    self.cards[name] = prices
                self._card_depth = None
            self._div_depth -= 1


class MoonshotProvider(BaseProvider):
    name = "Moonshot AI"

    MODEL_ALIASES: dict[str, tuple[str, ...]] = {
        "kimi-k2.6": ("K2.6",),
    }

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
        parser = _ModelCardParser()
        parser.feed(raw_data)
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
            matches = [parser.cards[name] for name in names if name in parser.cards]
            if len(matches) != 1:
                warnings.append(
                    "official price not found"
                    if not matches
                    else "multiple official price records matched; retained existing values"
                )
            else:
                prices = matches[0]
                self._update_rates(candidate, "input", prices.get("Input"), warnings)
                self._update_rates(candidate, "output", prices.get("Output"), warnings)
                if "cache" in candidate:
                    self._update_cache(candidate["cache"], prices.get("Cache Hit"), warnings)
                if "batch" in candidate:
                    warnings.append(
                        "official batch price not found; retained existing values"
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
    def _update_rates(
        cls,
        target: dict[str, Any],
        key: str,
        raw_value: str | None,
        warnings: list[str],
    ) -> None:
        if key not in target:
            return
        value = cls._parse_price(raw_value)
        if value is None:
            warnings.append(f"official {key} price not found; retained existing values")
            return
        rates = target[key]
        if not isinstance(rates, list):
            warnings.append(f"{key} has an unexpected structure; retained existing value")
            return
        for rate in rates:
            if not isinstance(rate, dict) or "per_million" not in rate:
                warnings.append(
                    f"{key} has an unexpected structure; retained existing value"
                )
                continue
            rate["per_million"] = float(value)

    @classmethod
    def _update_cache(
        cls, groups: Any, raw_value: str | None, warnings: list[str]
    ) -> None:
        value = cls._parse_price(raw_value)
        if value is None:
            warnings.append(
                "official cache hit price not found; retained existing values"
            )
            return
        if not isinstance(groups, list):
            warnings.append("cache has an unexpected structure; retained existing value")
            return
        for group in groups:
            if not isinstance(group, dict) or "cache_read" not in group:
                continue
            rates = group["cache_read"]
            if not isinstance(rates, list):
                warnings.append(
                    "cache.cache_read has an unexpected structure; retained existing value"
                )
                continue
            for rate in rates:
                if not isinstance(rate, dict) or "per_million" not in rate:
                    warnings.append(
                        "cache.cache_read has an unexpected structure; retained existing value"
                    )
                    continue
                rate["per_million"] = float(value)

    @staticmethod
    def _parse_price(value: str | None) -> Decimal | None:
        if value is None:
            return None
        cleaned = value.strip().removeprefix("$").split()[0].replace(",", "")
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
