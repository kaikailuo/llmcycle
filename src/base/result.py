from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResult:
    """A candidate replacement for one existing model object.

    ``data`` deliberately retains the shape from models.json.  Provider code
    only changes fields it can obtain reliably from the official source.
    """

    provider: str
    model_api_id: str
    model_region: str
    data: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
