from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .result import ModelResult


class BaseProvider(ABC):
    """Common provider contract without provider-specific assumptions."""

    name: str

    def run(
        self, source_url: str, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        raw_data = self.fetch(source_url)
        return self.parse(raw_data, models)

    @abstractmethod
    def fetch(self, source_url: str) -> str:
        """Fetch raw data from a provider's official source."""

    @abstractmethod
    def parse(
        self, raw_data: str, models: list[dict[str, Any]]
    ) -> list[ModelResult]:
        """Convert official source data into candidate model objects."""
