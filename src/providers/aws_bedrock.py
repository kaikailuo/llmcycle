from __future__ import annotations

import copy
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult


class AWSBedrockProvider(BaseProvider):
    """Read Claude token prices from the public AWS Price List offer."""

    name = "AWS Bedrock"

    _BEDROCK_EDITION_SUFFIX = " (Amazon Bedrock Edition)"
    _EXCLUDED_USAGE_MARKERS = (
        "reserved",
        "tpm",
        "flex",
        "priority",
        "latencyoptimized",
        "latency_optimized",
        "long_ctx",
    )
    _USAGE_TYPES: dict[str, tuple[str, bool]] = {
        # Legacy AWS Price List names.
        "inputtokencount": ("input", False),
        "outputtokencount": ("output", False),
        "cachereadinputtokencount": ("cache_read", False),
        "cachewriteinputtokencount": ("cache_write_5m", False),
        "cachewrite1hinputtokencount": ("cache_write_1h", False),
        "inputtokencount_batch": ("batch.input", False),
        "outputtokencount_batch": ("batch.output", False),
        "inputtokencount_global": ("input", True),
        "outputtokencount_global": ("output", True),
        "cachereadinputtokencount_global": ("cache_read", True),
        "cachewriteinputtokencount_global": ("cache_write_5m", True),
        "cachewrite1hinputtokencount_global": ("cache_write_1h", True),
        "inputtokencount_global_batch": ("batch.input", True),
        "outputtokencount_global_batch": ("batch.output", True),
        # Current snake_case AWS Price List names.
        "input_tokens_standard": ("input", False),
        "output_tokens_standard": ("output", False),
        "cache_read_tokens_standard": ("cache_read", False),
        "cache_write_tokens_standard": ("cache_write_5m", False),
        "cache_write_tokens_1h_standard": ("cache_write_1h", False),
        "input_tokens_batch": ("batch.input", False),
        "output_tokens_batch": ("batch.output", False),
        "input_tokens_global_standard": ("input", True),
        "output_tokens_global_standard": ("output", True),
        "cache_read_tokens_global_standard": ("cache_read", True),
        "cache_write_tokens_global_standard": ("cache_write_5m", True),
        "cache_write_tokens_1h_global_standard": ("cache_write_1h", True),
        "input_tokens_global_batch": ("batch.input", True),
        "output_tokens_global_batch": ("batch.output", True),
    }

    def fetch(self, source_url: str) -> dict[str, Any]:
        request = Request(
            source_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "llmcycle-pricing-maintainer/1.0",
            },
        )
        with urlopen(request, timeout=60) as response:
            data = json.load(response)
        if not isinstance(data, dict):
            raise ValueError("AWS Price List returned invalid JSON")
        return data

    def parse(
        self, raw_data: dict[str, Any], models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        products, terms = self._offer_sections(raw_data)
        results: list[ModelResult] = []

        for original in models:
            candidate = copy.deepcopy(original)
            warnings: list[str] = []
            model_id = original.get("model_api_id")
            model_region = original.get("model_region")
            model_name = original.get("name")
            if not all(
                isinstance(value, str) and value
                for value in (model_id, model_region, model_name)
            ):
                warnings.append(
                    "model_api_id, model_region, or name is missing or invalid; "
                    "retained existing values"
                )
            else:
                prices, unsupported_units = self._collect_prices(
                    products, terms, model_name, model_region
                )
                if self._update_candidate(
                    candidate, prices, unsupported_units, warnings
                ):
                    candidate["currency"] = "USD"

            results.append(
                ModelResult(
                    provider=self.name,
                    model_api_id=str(model_id or "<missing>"),
                    model_region=str(model_region or "<missing>"),
                    data=candidate,
                    warnings=list(dict.fromkeys(warnings)),
                )
            )

        return results

    @staticmethod
    def _offer_sections(
        raw_data: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(raw_data, dict):
            raise ValueError("AWS Price List data must be an object")
        products = raw_data.get("products")
        all_terms = raw_data.get("terms")
        terms = all_terms.get("OnDemand") if isinstance(all_terms, dict) else None
        if not isinstance(products, dict) or not isinstance(terms, dict):
            raise ValueError("AWS Price List products or OnDemand terms are missing")
        return products, terms

    @classmethod
    def _collect_prices(
        cls,
        products: dict[str, Any],
        terms: dict[str, Any],
        model_name: str,
        model_region: str,
    ) -> tuple[dict[str, set[Decimal]], set[str]]:
        prices: dict[str, set[Decimal]] = {}
        unsupported_units: set[str] = set()

        for sku, product in products.items():
            if not isinstance(product, dict):
                continue
            attributes = product.get("attributes")
            if not isinstance(attributes, dict):
                continue
            source_name = attributes.get("servicename")
            if not isinstance(source_name, str):
                continue
            if cls._normalize_model_name(source_name) != model_name:
                continue

            classified = cls._classify_usage_type(attributes.get("usagetype"))
            if classified is None:
                continue
            target, is_global = classified
            if model_region == "Global":
                if not is_global:
                    continue
            elif is_global or attributes.get("regionCode") != model_region:
                continue

            for dimension in cls._price_dimensions(terms.get(str(sku))):
                if not cls._is_per_million_tokens(dimension.get("unit")):
                    unsupported_units.add(target)
                    continue
                price_per_unit = dimension.get("pricePerUnit")
                usd = price_per_unit.get("USD") if isinstance(price_per_unit, dict) else None
                price = cls._decimal_price(usd)
                if price is not None:
                    prices.setdefault(target, set()).add(price)

        return prices, unsupported_units

    @classmethod
    def _normalize_model_name(cls, value: str) -> str:
        value = value.strip()
        if value.endswith(cls._BEDROCK_EDITION_SUFFIX):
            return value[: -len(cls._BEDROCK_EDITION_SUFFIX)].strip()
        return value

    @classmethod
    def _classify_usage_type(cls, value: Any) -> tuple[str, bool] | None:
        if not isinstance(value, str):
            return None
        usage_type = value.split(":", 1)[-1]
        usage_type = re.sub(r"-Units$", "", usage_type, flags=re.I)
        usage_type = re.sub(
            r"^[A-Z0-9]+_(?=(?:Input|Output|Cache|input_|output_|cache_))",
            "",
            usage_type,
        )
        normalized = usage_type.casefold()
        if any(marker in normalized for marker in cls._EXCLUDED_USAGE_MARKERS):
            return None
        return cls._USAGE_TYPES.get(normalized)

    @staticmethod
    def _price_dimensions(raw_terms: Any) -> list[dict[str, Any]]:
        dimensions: list[dict[str, Any]] = []
        if not isinstance(raw_terms, dict):
            return dimensions
        for term in raw_terms.values():
            if not isinstance(term, dict):
                continue
            price_dimensions = term.get("priceDimensions")
            if not isinstance(price_dimensions, dict):
                continue
            dimensions.extend(
                dimension
                for dimension in price_dimensions.values()
                if isinstance(dimension, dict)
            )
        return dimensions

    @staticmethod
    def _is_per_million_tokens(unit: Any) -> bool:
        if not isinstance(unit, str):
            return False
        normalized = "".join(unit.casefold().split())
        return normalized in {
            "1mtoken",
            "1mtokens",
            "1000000token",
            "1000000tokens",
        }

    @staticmethod
    def _decimal_price(value: Any) -> Decimal | None:
        try:
            price = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        if not price.is_finite() or price < 0:
            return None
        return price

    @classmethod
    def _update_candidate(
        cls,
        candidate: dict[str, Any],
        prices: dict[str, set[Decimal]],
        unsupported_units: set[str],
        warnings: list[str],
    ) -> bool:
        updated = False
        if "input" in candidate:
            updated |= cls._update_rates(
                candidate["input"], prices, unsupported_units, "input", "input", warnings
            )
        if "output" in candidate:
            updated |= cls._update_rates(
                candidate["output"], prices, unsupported_units, "output", "output", warnings
            )
        if "cache" in candidate:
            updated |= cls._update_cache(
                candidate["cache"], prices, unsupported_units, warnings
            )
        if "batch" in candidate:
            batch = candidate["batch"]
            if not isinstance(batch, dict):
                warnings.append("batch has an unexpected structure; retained existing value")
            else:
                if "input" in batch:
                    updated |= cls._update_rates(
                        batch["input"],
                        prices,
                        unsupported_units,
                        "batch.input",
                        "batch.input",
                        warnings,
                    )
                if "output" in batch:
                    updated |= cls._update_rates(
                        batch["output"],
                        prices,
                        unsupported_units,
                        "batch.output",
                        "batch.output",
                        warnings,
                    )
        return updated

    @classmethod
    def _one_price(
        cls,
        prices: dict[str, set[Decimal]],
        unsupported_units: set[str],
        target: str,
        path: str,
        warnings: list[str],
    ) -> Decimal | None:
        matches = prices.get(target, set())
        if not matches:
            if target in unsupported_units:
                warnings.append(
                    f"official unit is not per 1M tokens for {path}; "
                    "retained existing value"
                )
            else:
                warnings.append(
                    f"official price not found for {path}; retained existing value"
                )
            return None
        if len(matches) > 1:
            rendered = ", ".join(str(price) for price in sorted(matches))
            warnings.append(
                f"multiple official prices matched for {path} ({rendered}); "
                "retained existing value"
            )
            return None
        return next(iter(matches))

    @classmethod
    def _update_rates(
        cls,
        rates: Any,
        prices: dict[str, set[Decimal]],
        unsupported_units: set[str],
        target: str,
        path: str,
        warnings: list[str],
    ) -> bool:
        price = cls._one_price(prices, unsupported_units, target, path, warnings)
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
        prices: dict[str, set[Decimal]],
        unsupported_units: set[str],
        warnings: list[str],
    ) -> bool:
        if not isinstance(groups, list):
            warnings.append("cache has an unexpected structure; retained existing value")
            return False
        updated = False
        for index, group in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"[{index}]"
            if not isinstance(group, dict):
                warnings.append(
                    f"cache{suffix} has an unexpected structure; retained existing value"
                )
                continue
            if "cache_read" in group:
                updated |= cls._update_rates(
                    group["cache_read"],
                    prices,
                    unsupported_units,
                    "cache_read",
                    f"cache{suffix}.cache_read",
                    warnings,
                )
            if "cache_writes" in group:
                writes = group["cache_writes"]
                updated |= cls._update_cache_writes(
                    writes,
                    prices,
                    unsupported_units,
                    "cache_write_5m",
                    "5m_per_million",
                    f"cache{suffix}.cache_writes.5m_per_million",
                    warnings,
                )
                updated |= cls._update_cache_writes(
                    writes,
                    prices,
                    unsupported_units,
                    "cache_write_1h",
                    "1h_per_million",
                    f"cache{suffix}.cache_writes.1h_per_million",
                    warnings,
                )
        return updated

    @classmethod
    def _update_cache_writes(
        cls,
        writes: Any,
        prices: dict[str, set[Decimal]],
        unsupported_units: set[str],
        target: str,
        field: str,
        path: str,
        warnings: list[str],
    ) -> bool:
        if not isinstance(writes, list):
            warnings.append(f"{path} has an unexpected structure; retained existing value")
            return False
        applicable = [write for write in writes if isinstance(write, dict) and field in write]
        if not applicable:
            return False
        price = cls._one_price(prices, unsupported_units, target, path, warnings)
        if price is None:
            return False
        for write in applicable:
            write[field] = float(price)
        return True
