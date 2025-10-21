"""Configuration loading and management utilities."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration manager for the RAG pipeline."""

    def __init__(self, config_dict: Dict[str, Any], base_path: Path):
        """
        Initialize configuration.

        Args:
            config_dict: Dictionary containing configuration values
            base_path: Base path for resolving relative paths
        """
        self._config = config_dict
        self._base_path = base_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'paths.corpus_root')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_path(self, key: str, create: bool = False) -> Path:
        """
        Get a path from configuration and resolve it relative to base path.

        Args:
            key: Configuration key for the path
            create: Whether to create the directory if it doesn't exist

        Returns:
            Resolved Path object
        """
        path_str = self.get(key)
        if path_str is None:
            raise ValueError(f"Path configuration '{key}' not found")

        path = Path(path_str)
        if not path.is_absolute():
            path = self._base_path / path

        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found")
        return value

    @property
    def base_path(self) -> Path:
        """Get the base path for this configuration."""
        return self._base_path


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, looks for config.yaml
                    in default locations.

    Returns:
        Config object

    Raises:
        FileNotFoundError: If configuration file not found
        yaml.YAMLError: If configuration file is invalid
    """
    if config_path is None:
        # Try to find config in default locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
            Path.cwd() / "config/config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(
                "Configuration file not found. Searched: " + 
                ", ".join(str(p) for p in possible_paths)
            )
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Base path is the parent of the config directory
    base_path = config_path.parent.parent

    return Config(config_dict, base_path)

