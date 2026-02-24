"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Settings:
    """Container for hierarchical project settings."""

    raw: Dict[str, Any]

    @property
    def experiment(self) -> Dict[str, Any]:
        return self.raw["experiment"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def inference(self) -> Dict[str, Any]:
        return self.raw["inference"]


def load_settings(path: str | Path) -> Settings:
    """Load YAML settings file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    return Settings(raw=content)
