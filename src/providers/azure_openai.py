from __future__ import annotations

import copy
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult


class AzureOpenAIProvider(BaseProvider):
    name = "Azure OpenAI"

    _PRODUCT_NAMES = ("Azure OpenAI", "Azure OpenAI GPT5")
    _MODEL_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?", re.I)
    _INPUT_MARKERS = {"inp", "inpt", "input"}
    _OUTPUT_MARKERS = {"opt", "out", "outp", "outpt", "output"}
    _CACHE_MARKERS = {"cached", "cchd", "cd"}
    _GLOBAL_MARKERS = {"global", "glbl", "gl"}
    _ALLOWED_DESCRIPTORS = (
        _INPUT_MARKERS
        | _OUTPUT_MARKERS
        | _CACHE_MARKERS
        | _GLOBAL_MARKERS
        | {"batch", "chat", "token", "tokens", "tkn"}
    )

    def fetch(self, source_url: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for product_name in self._PRODUCT_NAMES:
            query = urlencode(
                {
                    "api-version": "2023-01-01-preview",
                    "$filter": (
                        f"productName eq '{product_name}' "
                        "and priceType eq 'Consumption'"
                    ),
                }
            )
            next_url = f"{source_url}?{query}"
            visited: set[str] = set()
            while next_url:
                if next_url in visited:
                    raise ValueError("Azure Retail Prices API pagination loop detected")
                visited.add(next_url)
                request = Request(
                    next_url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "llmcycle-pricing-maintainer/1.0",
                    },
                )
                with urlopen(request, timeout=60) as response:
                    page = json.load(response)
                if not isinstance(page, dict) or not isinstance(
                    page.get("Items"), list
                ):
                    raise ValueError("Azure Retail Prices API returned invalid JSON")
                items.extend(item for item in page["Items"] if isinstance(item, dict))
                raw_next = page.get("NextPageLink")
                if raw_next in (None, ""):
                    next_url = ""
                elif isinstance(raw_next, str):
                    next_url = raw_next
                else:
                    raise ValueError(
                        "Azure Retail Prices API returned an invalid NextPageLink"
                    )
        return items

    def parse(
        self, raw_data: list[dict[str, Any]], models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        if not isinstance(raw_data, list) or not raw_data:
            raise ValueError("Azure OpenAI retail pricing data is empty")

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
            if region != "Global":
                warnings.append(
                    "only Global Azure OpenAI pricing is supported; retained existing values"
                )
                results.append(
                    ModelResult(
                        provider=self.name,
                        model_api_id=model_id,
                        model_region=region,
                        data=candidate,
                        warnings=warnings,
                    )
                )
                continue

            prices, matched_meter_count = self._collect_prices(model_id, raw_data)
            if matched_meter_count == 0:
                warnings.append(
                    "official Global price not found; retained existing values"
                )
            else:
                updated = self._update_candidate(candidate, prices, warnings)
                if updated:
                    candidate["currency"] = "USD"

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
    def _collect_prices(
        cls, model_id: str, items: list[dict[str, Any]]
    ) -> tuple[dict[str, set[Decimal]], int]:
        prices: dict[str, set[Decimal]] = {}
        matched_meter_count = 0
        model_tokens = cls._tokens(model_id)
        for item in items:
            parsed = cls._parse_meter(item, model_tokens)
            if parsed is None:
                continue
            target, price = parsed
            prices.setdefault(target, set()).add(price)
            matched_meter_count += 1
        return prices, matched_meter_count

    @classmethod
    def _parse_meter(
        cls, item: dict[str, Any], model_tokens: list[str]
    ) -> tuple[str, Decimal] | None:
        product_name = item.get("productName")
        if product_name is not None and product_name not in cls._PRODUCT_NAMES:
            return None
        if item.get("type") != "Consumption":
            return None
        if item.get("isPrimaryMeterRegion") is not True:
            return None
        if item.get("currencyCode") != "USD":
            return None

        sku_name = item.get("skuName")
        if not isinstance(sku_name, str):
            return None
        sku_tokens = cls._tokens(sku_name)
        if sku_tokens[: len(model_tokens)] != model_tokens:
            return None
        descriptors = sku_tokens[len(model_tokens) :]
        if not descriptors or not any(
            token in cls._GLOBAL_MARKERS for token in descriptors
        ):
            return None
        if any(
            token not in cls._ALLOWED_DESCRIPTORS and not token.isdigit()
            for token in descriptors
        ):
            return None

        has_input = any(token in cls._INPUT_MARKERS for token in descriptors)
        has_output = any(token in cls._OUTPUT_MARKERS for token in descriptors)
        if has_input == has_output:
            return None
        is_batch = "batch" in descriptors
        is_cached = any(token in cls._CACHE_MARKERS for token in descriptors)
        if is_cached and not has_input:
            return None

        if is_batch and is_cached:
            target = "batch.cache_read"
        elif is_batch:
            target = "batch.input" if has_input else "batch.output"
        elif is_cached:
            target = "cache_read"
        else:
            target = "input" if has_input else "output"

        price = cls._price_per_million(
            item.get("retailPrice"), item.get("unitOfMeasure")
        )
        if price is None:
            return None
        return target, price

    @classmethod
    def _tokens(cls, value: str) -> list[str]:
        return [token.casefold() for token in cls._MODEL_TOKEN_RE.findall(value)]

    @staticmethod
    def _price_per_million(value: Any, unit: Any) -> Decimal | None:
        try:
            price = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        if not price.is_finite() or price < 0 or not isinstance(unit, str):
            return None
        normalized = "".join(unit.casefold().split())
        if normalized in {"1k", "1ktoken", "1ktokens", "1000token", "1000tokens"}:
            return price * 1000
        if normalized in {
            "1m",
            "1mtoken",
            "1mtokens",
            "1000000token",
            "1000000tokens",
        }:
            return price
        return None

    @classmethod
    def _update_candidate(
        cls,
        candidate: dict[str, Any],
        prices: dict[str, set[Decimal]],
        warnings: list[str],
    ) -> bool:
        updated = False
        updated |= cls._update_rates(
            candidate.get("input"), prices.get("input", set()), "input", warnings
        )
        updated |= cls._update_rates(
            candidate.get("output"),
            prices.get("output", set()),
            "output",
            warnings,
        )
        if "cache" in candidate:
            updated |= cls._update_cache(
                candidate["cache"], prices.get("cache_read", set()), "cache", warnings
            )

        if "batch" in candidate:
            batch = candidate["batch"]
            if not isinstance(batch, dict):
                warnings.append("batch has an unexpected structure; retained existing value")
            else:
                updated |= cls._update_rates(
                    batch.get("input"),
                    prices.get("batch.input", set()),
                    "batch.input",
                    warnings,
                )
                updated |= cls._update_rates(
                    batch.get("output"),
                    prices.get("batch.output", set()),
                    "batch.output",
                    warnings,
                )
                if "cache" in batch:
                    updated |= cls._update_cache(
                        batch["cache"],
                        prices.get("batch.cache_read", set()),
                        "batch.cache",
                        warnings,
                    )
        return updated

    @staticmethod
    def _one_price(
        prices: set[Decimal], path: str, warnings: list[str]
    ) -> Decimal | None:
        if not prices:
            warnings.append(
                f"official price not found for {path}; retained existing values"
            )
            return None
        if len(prices) > 1:
            warnings.append(
                f"multiple official prices found for {path}; retained existing values"
            )
            return None
        return next(iter(prices))

    @classmethod
    def _update_rates(
        cls,
        rates: Any,
        prices: set[Decimal],
        path: str,
        warnings: list[str],
    ) -> bool:
        if rates is None:
            return False
        price = cls._one_price(prices, path, warnings)
        if price is None:
            return False
        if not isinstance(rates, list):
            warnings.append(f"{path} has an unexpected structure; retained existing value")
            return False
        updated = False
        for rate in rates:
            if not isinstance(rate, dict) or "per_million" not in rate:
                warnings.append(
                    f"{path} has an unexpected structure; retained existing value"
                )
                continue
            rate["per_million"] = float(price)
            updated = True
        return updated

    @classmethod
    def _update_cache(
        cls,
        groups: Any,
        prices: set[Decimal],
        path: str,
        warnings: list[str],
    ) -> bool:
        if not isinstance(groups, list):
            warnings.append(f"{path} has an unexpected structure; retained existing value")
            return False
        updated = False
        for index, group in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"[{index}]"
            if not isinstance(group, dict):
                warnings.append(
                    f"{path}{suffix} has an unexpected structure; retained existing value"
                )
                continue
            updated |= cls._update_rates(
                group.get("cache_read"),
                prices,
                f"{path}{suffix}.cache_read",
                warnings,
            )
            if "cache_writes" in group:
                warnings.append(
                    f"official price not found for {path}{suffix}.cache_writes; "
                    "retained existing values"
                )
        return updated
