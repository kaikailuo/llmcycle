from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from src.base.provider import BaseProvider
from src.base.result import ModelResult
from src.utils.currency import cny_to_usd


@dataclass
class _TierPrices:
    max_prompt_tokens: int | None
    max_completion_tokens: int | None
    prices: dict[str, set[Decimal]] = field(default_factory=dict)


class VolcengineProvider(BaseProvider):
    name = "Volcengine"

    FOUNDATION_MODEL_MAP = {
        "ep-20251029183622-7nd6b": "doubao-1.5-pro-32k",
    }

    _ACTION = "GetModelActivation"
    _VERSION = "2024-01-01"
    _REGION = "cn-beijing"
    _SERVICE = "ark"
    _CONTENT_TYPE = "application/json"
    _SIGNED_HEADERS = "content-type;host;x-content-sha256;x-date"
    _USD_QUANTUM = Decimal("0.01")
    _TYPE_TARGETS = {
        "inferenceprompt": "input",
        "prompttoken": "input",
        "inferencecompletion": "output",
        "completiontoken": "output",
    }

    def run(
        self, source_url: str, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        access_key = os.environ.get("VOLCENGINE_ACCESS_KEY_ID")
        secret_key = os.environ.get("VOLCENGINE_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            return self._unchanged_results(
                models,
                "VOLCENGINE_ACCESS_KEY_ID and VOLCENGINE_SECRET_ACCESS_KEY are not set; "
                "skipped Volcengine price update",
            )

        responses: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for original in models:
            model_id = original.get("model_api_id")
            if not isinstance(model_id, str):
                continue
            foundation_model = self.FOUNDATION_MODEL_MAP.get(model_id, model_id)
            try:
                responses[model_id] = self.fetch(source_url, foundation_model)
            except Exception as exc:
                errors[model_id] = (
                    f"GetModelActivation failed for {foundation_model}: "
                    f"{type(exc).__name__}: {exc}; retained existing values"
                )
        return self.parse({"responses": responses, "errors": errors}, models)

    def fetch(
        self, source_url: str, foundation_model_name: str | None = None
    ) -> dict[str, Any]:
        if not foundation_model_name:
            raise ValueError(
                "Volcengine GetModelActivation requires a foundation model name"
            )
        access_key = os.environ.get("VOLCENGINE_ACCESS_KEY_ID")
        secret_key = os.environ.get("VOLCENGINE_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            raise ValueError(
                "VOLCENGINE_ACCESS_KEY_ID and VOLCENGINE_SECRET_ACCESS_KEY are required"
            )

        body = json.dumps(
            {
                "FoundationModelName": foundation_model_name,
                "WithPrice": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = self._signed_request(
            source_url, body, access_key, secret_key, datetime.now(timezone.utc)
        )
        with urlopen(request, timeout=60) as response:
            data = json.load(response)
        if not isinstance(data, dict):
            raise ValueError("GetModelActivation returned invalid JSON")
        return data

    def parse(
        self, raw_data: Any, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        responses, errors = self._response_maps(raw_data, models)
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

            if model_id in errors:
                warnings.append(errors[model_id])
            else:
                response = responses.get(model_id)
                foundation_model = self.FOUNDATION_MODEL_MAP.get(model_id, model_id)
                item = self._activation_item(response)
                if item is None:
                    warnings.append(
                        f"GetModelActivation returned no pricing item for "
                        f"{foundation_model}; retained existing values"
                    )
                elif item.get("FoundationModelName") != foundation_model:
                    warnings.append(
                        "GetModelActivation returned a different FoundationModelName; "
                        "retained existing values"
                    )
                else:
                    updated = self._update_from_item(candidate, item, warnings)
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
    def _unchanged_results(
        cls, models: list[dict[str, Any]], warning: str
    ) -> list[ModelResult]:
        results: list[ModelResult] = []
        for original in models:
            model_id = original.get("model_api_id")
            region = original.get("model_region")
            results.append(
                ModelResult(
                    provider=cls.name,
                    model_api_id=str(model_id or "<missing>"),
                    model_region=str(region or "<missing>"),
                    data=copy.deepcopy(original),
                    warnings=[warning],
                )
            )
        return results

    @classmethod
    def _response_maps(
        cls, raw_data: Any, models: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        if not isinstance(raw_data, dict):
            return {}, {}
        if "Result" in raw_data and len(models) == 1:
            model_id = models[0].get("model_api_id")
            return ({str(model_id): raw_data} if model_id is not None else {}), {}

        raw_responses = raw_data.get("responses", raw_data)
        raw_errors = raw_data.get("errors", {})
        responses = {
            str(key): value
            for key, value in raw_responses.items()
            if isinstance(value, dict)
        } if isinstance(raw_responses, dict) else {}
        errors = {
            str(key): str(value)
            for key, value in raw_errors.items()
        } if isinstance(raw_errors, dict) else {}
        for model in models:
            model_id = model.get("model_api_id")
            if not isinstance(model_id, str) or model_id in responses:
                continue
            foundation_model = cls.FOUNDATION_MODEL_MAP.get(model_id, model_id)
            value = responses.get(foundation_model)
            if isinstance(value, dict):
                responses[model_id] = value
        return responses, errors

    @staticmethod
    def _activation_item(response: Any) -> dict[str, Any] | None:
        if not isinstance(response, dict):
            return None
        result = response.get("Result")
        if not isinstance(result, dict):
            return None
        item = result.get("Item")
        return item if isinstance(item, dict) else None

    @classmethod
    def _update_from_item(
        cls,
        candidate: dict[str, Any],
        item: dict[str, Any],
        warnings: list[str],
    ) -> bool:
        flat_prices = cls._parse_charge_items(item.get("ChargeItems"))
        tier_prices = cls._parse_multi_charge_items(item.get("MultiChargeItems"))
        updated = False

        updated |= cls._update_field(
            candidate.get("input"), "input", flat_prices, tier_prices, warnings
        )
        updated |= cls._update_field(
            candidate.get("output"), "output", flat_prices, tier_prices, warnings
        )

        if "cache" in candidate:
            updated |= cls._update_cache(
                candidate["cache"], "cache_read", flat_prices, tier_prices, "cache", warnings
            )
        if "batch" in candidate:
            batch = candidate["batch"]
            if not isinstance(batch, dict):
                warnings.append("batch has an unexpected structure; retained existing value")
            else:
                updated |= cls._update_field(
                    batch.get("input"),
                    "batch.input",
                    flat_prices,
                    tier_prices,
                    warnings,
                )
                updated |= cls._update_field(
                    batch.get("output"),
                    "batch.output",
                    flat_prices,
                    tier_prices,
                    warnings,
                )
                if "cache" in batch:
                    updated |= cls._update_cache(
                        batch["cache"],
                        "batch.cache_read",
                        flat_prices,
                        tier_prices,
                        "batch.cache",
                        warnings,
                    )
        return updated

    @classmethod
    def _parse_charge_items(cls, value: Any) -> dict[str, set[Decimal]]:
        prices: dict[str, set[Decimal]] = {}
        if not isinstance(value, list):
            return prices
        for charge_item in value:
            parsed = cls._parse_charge_item(charge_item)
            if parsed is None:
                continue
            target, price = parsed
            prices.setdefault(target, set()).add(price)
        return prices

    @classmethod
    def _parse_multi_charge_items(cls, value: Any) -> list[_TierPrices]:
        if not isinstance(value, list):
            return []
        tiers: dict[tuple[int | None, int | None], _TierPrices] = {}
        for raw_tier in value:
            if not isinstance(raw_tier, dict):
                continue
            max_prompt = cls._nonnegative_int(raw_tier.get("MaxPromptTokens"))
            max_completion = cls._nonnegative_int(
                raw_tier.get("MaxCompletionTokens")
            )
            if max_prompt is None and max_completion is None:
                continue
            key = (max_prompt, max_completion)
            tier = tiers.setdefault(key, _TierPrices(max_prompt, max_completion))
            parsed = cls._parse_charge_items(raw_tier.get("ChargeItems"))
            for target, prices in parsed.items():
                tier.prices.setdefault(target, set()).update(prices)
        return sorted(
            tiers.values(),
            key=lambda tier: (
                tier.max_prompt_tokens is None,
                tier.max_prompt_tokens or 0,
                tier.max_completion_tokens is None,
                tier.max_completion_tokens or 0,
            ),
        )

    @classmethod
    def _parse_charge_item(cls, value: Any) -> tuple[str, Decimal] | None:
        if not isinstance(value, dict):
            return None
        type_name = value.get("Type")
        if not isinstance(type_name, str):
            return None
        target = cls._TYPE_TARGETS.get(
            "".join(character for character in type_name.casefold() if character.isalnum())
        )
        if target is None:
            return None
        # Price is the current billable unit price. OriginalPrice is the
        # pre-discount reference price and must not replace it.
        price = cls._cny_per_million(value.get("Price"), value.get("UnitCode"))
        if price is None:
            return None
        return target, price

    @staticmethod
    def _nonnegative_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    def _cny_per_million(value: Any, unit: Any) -> Decimal | None:
        try:
            price = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        if not price.is_finite() or price < 0 or not isinstance(unit, str):
            return None
        normalized = "".join(unit.casefold().split())
        if normalized in {
            "cny/1ktoken",
            "cny/1ktokens",
            "cny/ktoken",
            "cny/ktokens",
            "千token",
            "千tokens",
        }:
            return price * 1000
        if normalized in {
            "cny/1mtoken",
            "cny/1mtokens",
            "cny/mtoken",
            "cny/mtokens",
            "百万token",
            "百万tokens",
        }:
            return price
        return None

    @classmethod
    def _update_field(
        cls,
        rates: Any,
        target: str,
        flat_prices: dict[str, set[Decimal]],
        tiers: list[_TierPrices],
        warnings: list[str],
    ) -> bool:
        if rates is None:
            return False
        applicable_tiers = [tier for tier in tiers if target in tier.prices]
        if applicable_tiers:
            return cls._update_tiered_rates(rates, target, applicable_tiers, warnings)
        return cls._update_flat_rates(
            rates, flat_prices.get(target, set()), target, warnings
        )

    @classmethod
    def _update_tiered_rates(
        cls,
        rates: Any,
        target: str,
        tiers: list[_TierPrices],
        warnings: list[str],
    ) -> bool:
        if not isinstance(rates, list):
            warnings.append(f"{target} has an unexpected structure; retained existing value")
            return False
        if len(tiers) == 1:
            return cls._update_flat_rates(
                rates, tiers[0].prices[target], target, warnings
            )

        bounds: dict[tuple[int, int | None], set[Decimal]] = {}
        lower = 0
        for index, tier in enumerate(tiers):
            upper = tier.max_prompt_tokens
            if upper is None or upper <= lower:
                warnings.append(
                    f"official context tiers cannot be mapped for {target}; "
                    "retained existing values"
                )
                return False
            candidate_upper = None if index == len(tiers) - 1 else upper
            bounds.setdefault((lower, candidate_upper), set()).update(
                tier.prices[target]
            )
            lower = upper

        updated = False
        for rate in rates:
            if not isinstance(rate, dict) or "per_million" not in rate:
                warnings.append(
                    f"{target} has an unexpected structure; retained existing value"
                )
                continue
            key = (rate.get("context_min"), rate.get("context_max"))
            prices = bounds.get(key, set())
            price = cls._one_price(prices, cls._tier_path(target, key), warnings)
            if price is None:
                continue
            rate["per_million"] = cls._to_usd_float(price)
            updated = True
        return updated

    @classmethod
    def _update_flat_rates(
        cls,
        rates: Any,
        prices: set[Decimal],
        path: str,
        warnings: list[str],
    ) -> bool:
        if not isinstance(rates, list):
            warnings.append(f"{path} has an unexpected structure; retained existing value")
            return False
        price = cls._one_price(prices, path, warnings)
        if price is None:
            return False
        updated = False
        for rate in rates:
            if not isinstance(rate, dict) or "per_million" not in rate:
                warnings.append(
                    f"{path} has an unexpected structure; retained existing value"
                )
                continue
            rate["per_million"] = cls._to_usd_float(price)
            updated = True
        return updated

    @classmethod
    def _update_cache(
        cls,
        groups: Any,
        target: str,
        flat_prices: dict[str, set[Decimal]],
        tiers: list[_TierPrices],
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
            updated |= cls._update_field(
                group.get("cache_read"), target, flat_prices, tiers, warnings
            )
            if "cache_writes" in group:
                warnings.append(
                    f"official price not found for {path}{suffix}.cache_writes; "
                    "retained existing values"
                )
        return updated

    @staticmethod
    def _tier_path(target: str, key: tuple[Any, Any]) -> str:
        maximum = key[1] if key[1] is not None else "∞"
        return f"{target}[{key[0]}-{maximum}]"

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
    def _to_usd_float(cls, cny_per_million: Decimal) -> float:
        return float(
            cny_to_usd(cny_per_million).quantize(
                cls._USD_QUANTUM, rounding=ROUND_HALF_UP
            )
        )

    @classmethod
    def _signed_request(
        cls,
        source_url: str,
        body: bytes,
        access_key: str,
        secret_key: str,
        now: datetime,
    ) -> Request:
        parsed = urlsplit(source_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Volcengine endpoint must be an HTTPS URL")
        path = parsed.path or "/"
        query_items = list(parse_qsl(parsed.query, keep_blank_values=True))
        query_items.extend((("Action", cls._ACTION), ("Version", cls._VERSION)))
        canonical_query = urlencode(
            sorted(query_items), quote_via=quote, safe="-_.~"
        )
        request_url = urlunsplit(
            (parsed.scheme, parsed.netloc, path, canonical_query, "")
        )

        x_date = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short_date = x_date[:8]
        payload_hash = hashlib.sha256(body).hexdigest()
        canonical_headers = (
            f"content-type:{cls._CONTENT_TYPE}\n"
            f"host:{parsed.netloc}\n"
            f"x-content-sha256:{payload_hash}\n"
            f"x-date:{x_date}\n"
        )
        canonical_request = "\n".join(
            (
                "POST",
                path,
                canonical_query,
                canonical_headers,
                cls._SIGNED_HEADERS,
                payload_hash,
            )
        )
        credential_scope = (
            f"{short_date}/{cls._REGION}/{cls._SERVICE}/request"
        )
        string_to_sign = "\n".join(
            (
                "HMAC-SHA256",
                x_date,
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            )
        )
        signing_key = cls._signing_key(
            secret_key, short_date, cls._REGION, cls._SERVICE
        )
        signature = hmac.new(
            signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        authorization = (
            f"HMAC-SHA256 Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={cls._SIGNED_HEADERS}, Signature={signature}"
        )
        return Request(
            request_url,
            data=body,
            method="POST",
            headers={
                "Authorization": authorization,
                "Content-Type": cls._CONTENT_TYPE,
                "Host": parsed.netloc,
                "X-Content-Sha256": payload_hash,
                "X-Date": x_date,
                "User-Agent": "llmcycle-pricing-maintainer/1.0",
            },
        )

    @staticmethod
    def _signing_key(
        secret_key: str, date: str, region: str, service: str
    ) -> bytes:
        key_date = hmac.new(
            secret_key.encode("utf-8"), date.encode("utf-8"), hashlib.sha256
        ).digest()
        key_region = hmac.new(
            key_date, region.encode("utf-8"), hashlib.sha256
        ).digest()
        key_service = hmac.new(
            key_region, service.encode("utf-8"), hashlib.sha256
        ).digest()
        return hmac.new(key_service, b"request", hashlib.sha256).digest()
