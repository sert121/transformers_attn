# config.py

import os
from typing import Any, Dict

import yaml


class Config:
    """
    Configuration utility for loading and accessing settings from a YAML file
    and command-line arguments.

    Public methods:
      - get_training_params() -> Dict[str, Any]
      - get_model_params() -> Dict[str, Any]
      - get_data_paths() -> Dict[str, Any]
    """

    REQUIRED_SECTIONS = {
        "training",
        "optimizer",
        "scheduler",
        "model",
        "regularization",
        "data",
        "evaluation",
    }

    def __init__(self, args: Dict[str, Any]) -> None:
        """
        Initializes the Config object by loading the YAML configuration
        and validating required sections.

        Args:
            args: A dictionary of CLI arguments. Must contain 'config' key
                  pointing to the path of the YAML configuration file.
        Raises:
            ValueError: If 'config' is missing or required sections are absent.
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the config file is invalid YAML.
        """
        if "config" not in args or not isinstance(args["config"], str):
            raise ValueError("Missing required 'config' argument (path to config.yaml).")
        config_path = args["config"]
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as f:
            try:
                self._cfg: Dict[str, Any] = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML config: {e}")

        self._args: Dict[str, Any] = args
        self._validate()

    def _validate(self) -> None:
        """
        Validates that all required top-level sections are present in the
        loaded YAML configuration.

        Raises:
            ValueError: If any required section is missing.
        """
        missing = self.REQUIRED_SECTIONS - set(self._cfg.keys())
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

    def get_training_params(self) -> Dict[str, Any]:
        """
        Retrieves training-related parameters from the configuration.

        Returns:
            A dict containing:
              - train_steps: int
              - batch_source_tokens: int
              - batch_target_tokens: int
              - beam_size: int
              - length_penalty: float
        """
        t = self._cfg["training"]
        return {
            "train_steps": int(t.get("train_steps", 0)),
            "batch_source_tokens": int(t.get("batch_source_tokens", 0)),
            "batch_target_tokens": int(t.get("batch_target_tokens", 0)),
            "beam_size": int(t.get("beam_size", 1)),
            "length_penalty": float(t.get("length_penalty", 0.0)),
        }

    def get_model_params(self) -> Dict[str, Any]:
        """
        Retrieves model hyperparameters from the configuration.

        Returns:
            A dict containing:
              - encoder_layers: int
              - decoder_layers: int
              - d_model: int
              - d_ff: int
              - n_heads: int
              - d_k: int
              - d_v: int
              - dropout_rate: float
        """
        m = self._cfg["model"]
        return {
            "encoder_layers": int(m.get("encoder_layers", 0)),
            "decoder_layers": int(m.get("decoder_layers", 0)),
            "d_model": int(m.get("d_model", 0)),
            "d_ff": int(m.get("d_ff", 0)),
            "n_heads": int(m.get("n_heads", 0)),
            "d_k": int(m.get("d_k", 0)),
            "d_v": int(m.get("d_v", 0)),
            "dropout_rate": float(m.get("dropout_rate", 0.0)),
        }

    def get_data_paths(self) -> Dict[str, Any]:
        """
        Retrieves paths and data settings from CLI args and YAML config.

        Expects CLI args to provide:
          - train_src, train_tgt, val_src, val_tgt, test_src, test_tgt
          - output_dir

        Returns:
            A dict containing:
              - train_src, train_tgt, val_src, val_tgt, test_src, test_tgt: str
              - output_dir: str
              - bpe_model_path: str
              - vocab_path: str
              - task: str
              - tokenizer: str
              - shared_vocab: bool
              - vocab_size: int
              - bpe_merges: int
        """
        a = self._args
        d = self._cfg["data"]

        # CLI-provided paths; default to empty string if absent
        train_src = str(a.get("train_src", ""))
        train_tgt = str(a.get("train_tgt", ""))
        val_src = str(a.get("val_src", ""))
        val_tgt = str(a.get("val_tgt", ""))
        test_src = str(a.get("test_src", ""))
        test_tgt = str(a.get("test_tgt", ""))

        # Output directory for artifacts
        output_dir = str(a.get("output_dir", os.getcwd()))
        os.makedirs(output_dir, exist_ok=True)

        # Derived artifact paths
        bpe_model_path = os.path.join(output_dir, "bpe.model")
        vocab_path = os.path.join(output_dir, "vocab.txt")

        return {
            "train_src": train_src,
            "train_tgt": train_tgt,
            "val_src": val_src,
            "val_tgt": val_tgt,
            "test_src": test_src,
            "test_tgt": test_tgt,
            "output_dir": output_dir,
            "bpe_model_path": bpe_model_path,
            "vocab_path": vocab_path,
            "task": str(d.get("task", "")),
            "tokenizer": str(d.get("tokenizer", "")),
            "shared_vocab": bool(d.get("shared_vocab", False)),
            "vocab_size": int(d.get("vocab_size", 0)),
            "bpe_merges": int(d.get("bpe_merges", 0)),
        }
