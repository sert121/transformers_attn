# config.py

import os
import json
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration loader and accessor for YAML/JSON config files.

    Provides a dot-separated key lookup interface to access nested config values.
    """

    # Required top-level sections in the config file
    _REQUIRED_SECTIONS = [
        "training",
        "model",
        "optimizer",
        "learning_rate_scheduler",
        "data",
        "inference",
        "hardware",
    ]

    def __init__(self, path: str) -> None:
        """Initializes the Config object by loading and validating the config file.

        Args:
            path: Path to a YAML (.yaml/.yml) or JSON (.json) configuration file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file extension is unsupported.
            RuntimeError: If parsing the file fails.
            KeyError: If required top-level sections are missing.
        """
        self._path = path
        self._cfg_dict: Dict[str, Any] = {}

        if not os.path.isfile(self._path):
            raise FileNotFoundError(f"Config file not found at '{self._path}'")

        ext = os.path.splitext(self._path)[1].lower()
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                if ext in (".yaml", ".yml"):
                    self._cfg_dict = yaml.safe_load(f)
                elif ext == ".json":
                    self._cfg_dict = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported config file type '{ext}'. "
                        "Expected '.yaml', '.yml', or '.json'."
                    )
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Error parsing config file '{self._path}': {e}") from e

        if not isinstance(self._cfg_dict, dict):
            raise RuntimeError(
                f"Config file '{self._path}' did not produce a dict; got {type(self._cfg_dict)}"
            )

        self._validate_required_sections()

    def _validate_required_sections(self) -> None:
        """Validates that all required top-level sections are present in the config.

        Raises:
            KeyError: If any required section is missing.
        """
        missing = [
            section
            for section in self._REQUIRED_SECTIONS
            if section not in self._cfg_dict
        ]
        if missing:
            raise KeyError(
                f"Missing required config section(s): {', '.join(missing)}"
            )

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieves a configuration value using dot-separated keys.

        Args:
            key: Dot-separated key string, e.g., "training.total_steps".
            default: Value to return if the key is not found.

        Returns:
            The config value if found; otherwise, returns `default`.
        """
        parts = key.split(".")
        node: Any = self._cfg_dict

        for part in parts:
            if not isinstance(node, dict):
                return default
            if part in node:
                node = node[part]
            else:
                return default

        return node

    def all(self) -> Dict[str, Any]:
        """Returns the entire configuration dictionary.

        Returns:
            A dict representation of the loaded config.
        """
        return self._cfg_dict

    def __repr__(self) -> str:
        return f"<Config path='{self._path}' sections={list(self._cfg_dict.keys())}>"
