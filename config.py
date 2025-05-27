"""
config.py

Loads and validates configuration from a YAML or JSON file for the Transformer experiment.
Exposes configuration as attributes: config.model, config.training, config.optimizer,
config.lr_scheduler, config.evaluation, config.data, along with derived parameters.
"""

import os
import json
from typing import Any, Dict
from types import SimpleNamespace

import yaml


class Config:
    """
    Configuration loader and validator.

    Usage:
        cfg = Config('config.yaml')
        # Access parameters:
        d_model = cfg.model.d_model
        warmup = cfg.training.warmup_steps
        lr_scale = cfg.lr_scale
    """

    # Define the required schema: section -> {key: type}
    _SCHEMA: Dict[str, Dict[str, type]] = {
        "model": {
            "encoder_layers": int,
            "decoder_layers": int,
            "d_model": int,
            "d_ff": int,
            "num_heads": int,
            "dropout_rate": float,
            "positional_encoding": str,
            "share_embeddings": bool,
        },
        "training": {
            "batch_tokens_src": int,
            "batch_tokens_tgt": int,
            "max_steps": int,
            "warmup_steps": int,
            "label_smoothing": float,
        },
        "optimizer": {
            "type": str,
            "beta1": float,
            "beta2": float,
            "eps": float,
        },
        "lr_scheduler": {
            "type": str,
        },
        "evaluation": {
            "beam_size": int,
            "length_penalty": float,
            "checkpoint_average": int,
            "max_output_offset": int,
        },
        "data": {
            "spm_vocab_size": int,
        },
    }

    def __init__(self, path: str) -> None:
        """
        Initialize Config by loading and validating the given YAML/JSON file.

        :param path: Path to YAML (.yaml/.yml) or JSON (.json) configuration file.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file contents are invalid or missing required fields.
        """
        self._path = path
        self._config_dict = self._load_file(path)
        self._validate_and_populate(self._config_dict)
        self._compute_derived()

    def _load_file(self, path: str) -> Dict[str, Any]:
        """
        Load a configuration file in YAML or JSON format.

        :param path: File path.
        :return: Parsed configuration as a dictionary.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If parsing fails or root is not a dict.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, "r", encoding="utf-8") as f:
                if ext in [".yaml", ".yml"]:
                    cfg = yaml.safe_load(f)
                elif ext == ".json":
                    cfg = json.load(f)
                else:
                    # Try YAML by default
                    cfg = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse configuration file {path}: {e}")
        if not isinstance(cfg, dict):
            raise ValueError(f"Configuration root must be a mapping/object, got {type(cfg)}")
        return cfg

    def _validate_and_populate(self, cfg: Dict[str, Any]) -> None:
        """
        Validate the loaded config dict against the schema, and populate attributes.

        :param cfg: Raw configuration dictionary.
        :raises ValueError: If validation fails.
        """
        for section, fields in self._SCHEMA.items():
            if section not in cfg:
                raise ValueError(f"Missing required config section: '{section}'")
            section_val = cfg[section]
            if not isinstance(section_val, dict):
                raise ValueError(f"Config section '{section}' must be a mapping, got {type(section_val)}")
            # Validate each field
            for key, expected_type in fields.items():
                if key not in section_val:
                    raise ValueError(f"Missing required config field: '{section}.{key}'")
                val = section_val[key]
                # Allow int where float is expected
                if expected_type is float and isinstance(val, int):
                    val = float(val)
                    section_val[key] = val
                if not isinstance(val, expected_type):
                    raise ValueError(
                        f"Config field '{section}.{key}' must be of type {expected_type.__name__}, "
                        f"got {type(val).__name__}"
                    )
            # Populate as SimpleNamespace for attribute access
            setattr(self, section, SimpleNamespace(**section_val))

    def _compute_derived(self) -> None:
        """
        Compute derived configuration values for convenience.
        """
        # Inverse square-root of d_model for learning rate scaling
        d_model = self.model.d_model
        self.lr_scale: float = d_model ** -0.5
        # Total token count per batch (src + tgt)
        self.total_batch_tokens: int = self.training.batch_tokens_src + self.training.batch_tokens_tgt

    def dump(self) -> None:
        """
        Print the loaded configuration in a friendly YAML format.
        """
        try:
            dump_dict = self._config_dict.copy()
            # Insert derived values
            dump_dict["derived"] = {
                "lr_scale": self.lr_scale,
                "total_batch_tokens": self.total_batch_tokens,
            }
            print(yaml.dump(dump_dict, default_flow_style=False, sort_keys=False))
        except Exception:
            # Fallback to simple repr
            print(self._config_dict)

    def __repr__(self) -> str:
        return f"<Config path={self._path!r}>"
