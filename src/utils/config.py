"""
Configuration loader for ExoScope.

Loads the default YAML config and allows overrides from
a user-specified config file or environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional


# Default config path relative to project root
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"


class Config:
    """Hierarchical configuration manager.

    Loads default_config.yaml, then merges any user overrides.
    Access values with dot-separated keys: config.get("fusion.weights.tabular")
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Optional path to a user config YAML that overrides defaults.
        """
        self._data: dict = {}

        # Load defaults
        if _DEFAULT_CONFIG.exists():
            with open(_DEFAULT_CONFIG, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}

        # Merge user overrides
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            self._deep_merge(self._data, user_config)

        # Apply environment variable overrides (EXOSCOPE_<SECTION>_<KEY>=value)
        self._apply_env_overrides()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot-separated key path.

        Args:
            key: Dot-separated path, e.g. "fusion.weights.tabular"
            default: Value to return if key not found.

        Returns:
            The config value, or default if not found.
        """
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_section(self, section: str) -> dict:
        """Get an entire config section as a dict.

        Args:
            section: Top-level section name, e.g. "vision", "tabular".

        Returns:
            The section dict, or empty dict if not found.
        """
        return self._data.get(section, {})

    @property
    def data(self) -> dict:
        """Return the full config dict."""
        return self._data

    def _deep_merge(self, base: dict, override: dict) -> None:
        """Recursively merge override into base dict (in-place)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides with prefix EXOSCOPE_."""
        prefix = "EXOSCOPE_"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Convert EXOSCOPE_FUSION_WEIGHTS_TABULAR -> fusion.weights.tabular
                config_key = env_key[len(prefix):].lower().replace("_", ".")
                keys = config_key.split(".")
                target = self._data
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                # Try to cast to appropriate type
                target[keys[-1]] = self._cast_value(env_value)

    @staticmethod
    def _cast_value(value: str) -> Any:
        """Attempt to cast a string env value to int/float/bool."""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def __repr__(self) -> str:
        return f"Config(sections={list(self._data.keys())})"


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create the global Config singleton.

    Args:
        config_path: Optional path to user config YAML (only used on first call).

    Returns:
        The global Config instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
