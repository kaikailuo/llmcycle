from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from src.base.provider import BaseProvider
from src.providers.anthropic import AnthropicProvider
from src.providers.azure_openai import AzureOpenAIProvider
from src.providers.baidu import BaiduProvider
from src.providers.bigmodel import BigModelProvider
from src.providers.deepseek import DeepSeekProvider
from src.providers.google import GoogleProvider
from src.providers.minimax import MiniMaxProvider
from src.providers.moonshot import MoonshotProvider
from src.providers.openai import OpenAIProvider
from src.providers.tencent import TencentProvider
from src.providers.typesafe import TypesafeProvider
from src.providers.volcengine import VolcengineProvider


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
    "baidu": ProviderConfig(
        name="Baidu ERNIE",
        provider=BaiduProvider,
        url="https://cloud.baidu.com/doc/qianfan/s/wmh4sv6ya",
    ),
    "tencent": ProviderConfig(
        name="Tencent Hunyuan",
        provider=TencentProvider,
        url="https://cloud.tencent.com/document/product/1823/130055",
    ),
    "minimax": ProviderConfig(
        name="MiniMax",
        provider=MiniMaxProvider,
        url="https://platform.minimax.io/subscribe/token-plan?tab=api-enterprise",
    ),
    "deepseek": ProviderConfig(
        name="DeepSeek",
        provider=DeepSeekProvider,
        url="https://api-docs.deepseek.com/quick_start/pricing/",
    ),
    "typesafe": ProviderConfig(
        name="Typesafe AI",
        provider=TypesafeProvider,
        url="https://docs.typesafe.ai/models.md",
    ),
    "bigmodel": ProviderConfig(
        name="BigModel",
        provider=BigModelProvider,
        url="https://docs.bigmodel.cn/cn/guide/start/pricing.md",
    ),
    "azure_openai": ProviderConfig(
        name="Azure OpenAI",
        provider=AzureOpenAIProvider,
        url="https://prices.azure.com/api/retail/prices",
    ),
    "volcengine": ProviderConfig(
        name="Volcengine",
        provider=VolcengineProvider,
        url="https://ark.cn-beijing.volcengineapi.com/",
    ),
}


def get_provider(key: str) -> ProviderConfig:
    try:
        return PROVIDERS[key.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {key!r}; available: {available}") from exc
