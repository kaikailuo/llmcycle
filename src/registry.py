from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from src.base.provider import BaseProvider
from src.providers.openai import OpenAIProvider


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    provider: Type[BaseProvider]
    url: str


PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="OpenAI",
        provider=OpenAIProvider,
        url="https://developers.openai.com/api/docs/pricing",
    )
}


def get_provider(key: str) -> ProviderConfig:
    try:
        return PROVIDERS[key.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {key!r}; available: {available}") from exc
