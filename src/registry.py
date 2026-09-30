from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from src.base.provider import BaseProvider
from src.providers.anthropic import AnthropicProvider
from src.providers.google import GoogleProvider
from src.providers.moonshot import MoonshotProvider
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
    ),
    "anthropic": ProviderConfig(
        name="Anthropic",
        provider=AnthropicProvider,
        url="https://platform.claude.com/docs/en/about-claude/pricing",
    ),
    "moonshot": ProviderConfig(
        name="Moonshot AI",
        provider=MoonshotProvider,
        url="https://platform.kimi.ai/zh-hans",
    ),
    "google": ProviderConfig(
        name="Google",
        provider=GoogleProvider,
        url="https://ai.google.dev/gemini-api/docs/pricing",
    ),
}


def get_provider(key: str) -> ProviderConfig:
    try:
        return PROVIDERS[key.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {key!r}; available: {available}") from exc
