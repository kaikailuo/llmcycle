from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult
from src.utils.http import urlopen_with_retry


_UNSCOPED_CONTEXT = "unscoped"
_SHORT_CONTEXT = "short"
_LONG_CONTEXT = "long"
_CONTEXT_BOUNDARY = 272000


@dataclass(frozen=True)
class _SkuDimensions:
    context: str
    billing: str
    deployment: str
    region: str
    is_batch: bool

    @property
    def target(self) -> str:
        prefix = "batch." if self.is_batch else ""
        return f"{prefix}{self.billing}"


PriceMap = dict[tuple[str, str], set[Decimal]]


class AzureOpenAIProvider(BaseProvider):
    name = "Azure OpenAI"

    _PRODUCT_NAMES = ("Azure OpenAI", "Azure OpenAI GPT5", "Azure OpenAI GPT6")
    _MODEL_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?", re.I)
    _INPUT_MARKERS = {"inp", "inpt", "input"}
    _OUTPUT_MARKERS = {"opt", "out", "outp", "outpt", "output"}
    _CACHE_MARKERS = {"cache", "cached", "cchd", "cd"}
    _CACHE_WRITE_MARKERS = {"wr", "write", "writes"}
    _CONTEXT_MARKERS = {
        "shortco": _SHORT_CONTEXT,
        "longco": _LONG_CONTEXT,
    }
    _DEPLOYMENT_MARKERS = {
        "std": "standard",
        "pp": "priority",
        "flex": "flex",
        "fl": "flex",
    }
    _REGION_MARKERS = {
        "global": "global",
        "glbl": "global",
        "gl": "global",
        "dz": "data_zone",
        "dzone": "data_zone",
        "datazone": "data_zone",
        "regional": "regional",
        "regnl": "regional",
    }
    _WORKLOAD_MARKERS = {"batch", "chat"}
    _UNIT_MARKERS = {"token", "tokens", "tkn"}

    # Azure SKU identities are deliberately explicit. In particular, the generic
    # gpt-4o API id is the 2024-11-20 (1120) Azure snapshot, not every SKU whose
    # name starts with "gpt 4o".
    MODEL_SKU_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
        "gpt-4.1-mini": (("gpt", "4.1", "mini"),),
        "gpt-4o": (("gpt", "4", "o", "1120"),),
        "gpt-4o-mini": (("gpt", "4", "o", "mini", "0718"),),
        "gpt-5-mini": (("gpt", "5", "mini"),),
        "gpt-5.1": (("gpt", "5.1"),),
        "gpt-5.2": (("gpt", "5.2"),),
        "gpt-5.4": (("5.4",),),
        "gpt-5.4-mini": (("5.4", "mini"),),
        "gpt-5.5": (("5.5",),),
        "gpt-5.6-luna": (("5.6", "luna"),),
        "gpt-5.6-sol": (("5.6", "sol"),),
        "gpt-5.6-terra": (("5.6", "terra"),),
        "gpt-6-luna": (("6", "luna"),),
        "gpt-6-sol": (("6", "sol"),),
    }

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
                with urlopen_with_retry(
                    request, timeout=60, opener=urlopen
                ) as response:
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
    ) -> tuple[PriceMap, int]:
        prices: PriceMap = {}
        matched_meter_count = 0
        for item in items:
            parsed = cls._parse_meter(item, model_id)
            if parsed is None:
                continue
            dimensions, price = parsed
            key = (dimensions.target, dimensions.context)
            # Repeated regions and effective dates collapse only when the price
            # is identical. Different values remain an explicit conflict for
            # _one_price instead of silently selecting a date or region.
            prices.setdefault(key, set()).add(price)
            matched_meter_count += 1
        return prices, matched_meter_count

    @classmethod
    def _parse_meter(
        cls, item: dict[str, Any], model_id: str
    ) -> tuple[_SkuDimensions, Decimal] | None:
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
        aliases = cls.MODEL_SKU_ALIASES.get(model_id, ())
        matched_alias = next(
            (
                alias
                for alias in sorted(aliases, key=len, reverse=True)
                if tuple(sku_tokens[: len(alias)]) == alias
            ),
            None,
        )
        if matched_alias is None:
            return None
        descriptors = sku_tokens[len(matched_alias) :]
        dimensions = cls._parse_dimensions(descriptors)
        if dimensions is None:
            return None
        if dimensions.deployment != "standard" or dimensions.region != "global":
            return None

        price = cls._price_per_million(
            item.get("retailPrice"), item.get("unitOfMeasure")
        )
        if price is None:
            return None
        return dimensions, price

    @classmethod
    def _parse_dimensions(cls, descriptors: list[str]) -> _SkuDimensions | None:
        tokens = cls._normalize_descriptor_tokens(descriptors)
        if not tokens:
            return None

        known_tokens = (
            cls._INPUT_MARKERS
            | cls._OUTPUT_MARKERS
            | cls._CACHE_MARKERS
            | cls._CACHE_WRITE_MARKERS
            | set(cls._CONTEXT_MARKERS)
            | set(cls._DEPLOYMENT_MARKERS)
            | set(cls._REGION_MARKERS)
            | cls._WORKLOAD_MARKERS
            | cls._UNIT_MARKERS
        )
        if any(token not in known_tokens for token in tokens):
            return None

        context_values = {
            cls._CONTEXT_MARKERS[token]
            for token in tokens
            if token in cls._CONTEXT_MARKERS
        }
        deployment_values = {
            cls._DEPLOYMENT_MARKERS[token]
            for token in tokens
            if token in cls._DEPLOYMENT_MARKERS
        }
        region_values = {
            cls._REGION_MARKERS[token]
            for token in tokens
            if token in cls._REGION_MARKERS
        }
        if (
            len(context_values) > 1
            or len(deployment_values) > 1
            or len(region_values) != 1
        ):
            return None

        has_input = any(token in cls._INPUT_MARKERS for token in tokens)
        has_output = any(token in cls._OUTPUT_MARKERS for token in tokens)
        has_cache = any(token in cls._CACHE_MARKERS for token in tokens)
        has_cache_write = any(token in cls._CACHE_WRITE_MARKERS for token in tokens)

        if has_cache_write:
            if not has_cache or has_input or has_output:
                return None
            billing = "cache_write"
        elif has_cache:
            if not has_input or has_output:
                return None
            billing = "cache_read"
        elif has_input != has_output:
            billing = "input" if has_input else "output"
        else:
            return None

        return _SkuDimensions(
            context=next(iter(context_values), _UNSCOPED_CONTEXT),
            billing=billing,
            # Older Azure meters omit the deployment marker for ordinary
            # Standard pricing. Named alternatives (PP/Flex) are still parsed
            # explicitly and filtered by _parse_meter.
            deployment=next(iter(deployment_values), "standard"),
            region=next(iter(region_values)),
            is_batch="batch" in tokens,
        )

    @staticmethod
    def _normalize_descriptor_tokens(tokens: list[str]) -> list[str]:
        normalized: list[str] = []
        index = 0
        while index < len(tokens):
            if tokens[index : index + 2] == ["data", "zone"]:
                normalized.append("datazone")
                index += 2
            elif tokens[index] == "batchoutp":
                normalized.extend(("batch", "outp"))
                index += 1
            else:
                normalized.append(tokens[index])
                index += 1
        return normalized

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
        prices: PriceMap,
        warnings: list[str],
    ) -> bool:
        updated = False
        updated |= cls._update_rates(
            candidate.get("input"), prices, "input", warnings
        )
        updated |= cls._update_rates(
            candidate.get("output"),
            prices,
            "output",
            warnings,
        )
        if "cache" in candidate:
            updated |= cls._update_cache(
                candidate["cache"], prices, "cache", warnings
            )

        if "batch" in candidate:
            batch = candidate["batch"]
            if not isinstance(batch, dict):
                warnings.append(
                    "batch has an unexpected structure; retained existing value"
                )
            else:
                updated |= cls._update_rates(
                    batch.get("input"),
                    prices,
                    "batch.input",
                    warnings,
                )
                updated |= cls._update_rates(
                    batch.get("output"),
                    prices,
                    "batch.output",
                    warnings,
                )
                if "cache" in batch:
                    updated |= cls._update_cache(
                        batch["cache"],
                        prices,
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
        prices: PriceMap,
        path: str,
        warnings: list[str],
        target: str | None = None,
    ) -> bool:
        if rates is None:
            return False
        if not isinstance(rates, list):
            warnings.append(
                f"{path} has an unexpected structure; retained existing value"
            )
            return False
        updated = False
        for index, rate in enumerate(rates):
            rate_path = path if len(rates) == 1 else f"{path}[{index}]"
            if not isinstance(rate, dict) or "per_million" not in rate:
                warnings.append(
                    f"{rate_path} has an unexpected structure; retained existing value"
                )
                continue
            context = cls._rate_context(rate)
            if context is None:
                warnings.append(
                    f"{rate_path} has an unsupported context tier; "
                    "retained existing value"
                )
                continue
            price_target = target or path
            candidates = prices.get((price_target, context))
            if candidates is None and context == _SHORT_CONTEXT:
                # Some GPT-5 meters omit ShortCo; their unqualified meter is
                # the 0-272K price while LongCo remains explicit.
                candidates = prices.get((price_target, _UNSCOPED_CONTEXT))
            price = cls._one_price(candidates or set(), rate_path, warnings)
            if price is None:
                continue
            rate["per_million"] = float(price)
            updated = True
        return updated

    @staticmethod
    def _rate_context(rate: dict[str, Any]) -> str | None:
        context_min = rate.get("context_min")
        context_max = rate.get("context_max")
        if context_min == 0 and context_max is None:
            return _UNSCOPED_CONTEXT
        if context_min == 0 and context_max == _CONTEXT_BOUNDARY:
            return _SHORT_CONTEXT
        if context_min == _CONTEXT_BOUNDARY and context_max is None:
            return _LONG_CONTEXT
        return None

    @classmethod
    def _update_cache(
        cls,
        groups: Any,
        prices: PriceMap,
        path: str,
        warnings: list[str],
    ) -> bool:
        if not isinstance(groups, list):
            warnings.append(
                f"{path} has an unexpected structure; retained existing value"
            )
            return False
        updated = False
        target_prefix = "batch." if path.startswith("batch.") else ""
        for index, group in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"[{index}]"
            if not isinstance(group, dict):
                warnings.append(
                    f"{path}{suffix} has an unexpected structure; "
                    "retained existing value"
                )
                continue
            updated |= cls._update_rates(
                group.get("cache_read"),
                prices,
                f"{path}{suffix}.cache_read",
                warnings,
                target=f"{target_prefix}cache_read",
            )
            if "cache_writes" in group:
                updated |= cls._update_rates(
                    group.get("cache_writes"),
                    prices,
                    f"{path}{suffix}.cache_writes",
                    warnings,
                    target=f"{target_prefix}cache_write",
                )
        return updated
