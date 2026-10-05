from __future__ import annotations

import unittest

from src.providers.alibaba_bailian import AlibabaBailianProvider
from src.providers.azure_openai import AzureOpenAIProvider
from src.providers.aws_bedrock import AWSBedrockProvider
from src.providers.bigmodel import BigModelProvider
from src.providers.deepseek import DeepSeekProvider
from src.providers.typesafe import TypesafeProvider
from src.providers.volcengine import VolcengineProvider
from src.registry import get_provider


class RegistryTests(unittest.TestCase):
    def test_new_provider_configs(self) -> None:
        expected = {
            "deepseek": (
                "DeepSeek",
                DeepSeekProvider,
                "https://api-docs.deepseek.com/quick_start/pricing/",
            ),
            "typesafe": (
                "Typesafe AI",
                TypesafeProvider,
                "https://docs.typesafe.ai/models.md",
            ),
            "bigmodel": (
                "BigModel",
                BigModelProvider,
                "https://docs.bigmodel.cn/cn/guide/start/pricing.md",
            ),
            "azure_openai": (
                "Azure OpenAI",
                AzureOpenAIProvider,
                "https://prices.azure.com/api/retail/prices",
            ),
            "aws_bedrock": (
                "AWS Bedrock",
                AWSBedrockProvider,
                "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
                "AmazonBedrockFoundationModels/current/index.json",
            ),
            "volcengine": (
                "Volcengine",
                VolcengineProvider,
                "https://ark.cn-beijing.volcengineapi.com/",
            ),
            "alibaba_bailian": (
                "Alibaba Bailian",
                AlibabaBailianProvider,
                "https://help.aliyun.com/zh/model-studio/",
            ),
        }

        for key, values in expected.items():
            config = get_provider(key)
            self.assertEqual(values, (config.name, config.provider, config.url))


if __name__ == "__main__":
    unittest.main()
