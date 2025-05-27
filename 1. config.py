## 1. config.py

import os
import json
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration loader and accessor for YAML or JSON files.

    Provides hierarchical access to configuration values using dot-separated keys.
    Validates presence of required top-level sections on initialization.
    """

    REQUIRED_TOP_LEVEL_KEYS = ["model", "training", "inference", "parsing"]

    def __init__(self, path: str) -> None:
        """Load and parse the configuration file.

        Args:
            path: Path to a YAML (.yaml, .yml) or JSON (.json) config file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported or parsing fails.
            KeyError: If required top-level sections are missing.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        _, ext = os.path.splitext(path.lower())
        try:
            with open(path, "r", encoding="utf-8") as f:
                if ext in (".yaml", ".yml"):
                    self._cfg: Dict[str, Any] = yaml.safe_load(f)
                elif ext == ".json":
                    self._cfg = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported configuration file type '{ext}'. "
                        "Expected .yaml, .yml, or .json"
                    )
        except Exception as e:
            raise ValueError(f"Failed to parse configuration file '{path}': {e}")

        if not isinstance(self._cfg, dict):
            raise ValueError(
                f"Configuration root must be a mapping/dictionary; got {type(self._cfg)}"
            )

        # Validate presence of required top-level sections
        missing = [k for k in self.REQUIRED_TOP_LEVEL_KEYS if k not in self._cfg]
        if missing:
            raise KeyError(
                f"Missing required top-level config section(s): {', '.join(missing)}"
            )

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieve a configuration value by dot-separated key.

        Args:
            key: Dot-separated path to the config entry (e.g., "training.optimizer.type").
            default: Value to return if the key is not found. If None and key is missing,
                     a KeyError is raised.

        Returns:
            The configuration value corresponding to `key`, or `default` if provided.

        Raises:
            KeyError: If the key is not found and `default` is None.
        """
        parts = key.split(".")
        curr: Any = self._cfg
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                if default is not None:
                    return default
                raise KeyError(f"Config key '{key}' not found (failed at '{part}').")
        return curr

    def as_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary.

        Returns:
            A shallow copy of the internal configuration dictionary.
        """
        return dict(self._cfg)

    def subset(self, prefix: str) -> Dict[str, Any]:
        """Retrieve an entire subsection of the config as a dict.

        Args:
            prefix: Dot-separated path to the subsection (e.g., "training.scheduler").

        Returns:
            The subsection dictionary.

        Raises:
            KeyError: If the subsection key is not found or not a dict.
        """
        sub = self.get(prefix)
        if not isinstance(sub, dict):
            raise KeyError(f"Config subsection '{prefix}' is not a mapping.")
        return dict(sub)
