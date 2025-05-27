## utils.py

import yaml
import math
from typing import Any, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Config:
    """
    Loads and provides access to configuration parameters from a YAML file.
    Supports nested lookup via dot-separated keys.
    """

    def __init__(self, path: str) -> None:
        """
        Reads the YAML configuration file at the given path.

        Args:
            path: Path to the YAML config file.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file cannot be parsed.
        """
        try:
            with open(path, 'r') as f:
                self._cfg = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found: {path}") from e
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML config at {path}: {e}") from e

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the configuration using a dot-separated key.

        Args:
            key: Dot-separated key string, e.g. "model.d_model".
            default: Optional default to return if key is missing.

        Returns:
            The value corresponding to the key, or default if provided.

        Raises:
            KeyError: If key is not found and default is None.
        """
        parts = key.split('.')
        node = self._cfg
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is not None:
                    return default
                raise KeyError(f"Missing configuration key: '{key}'")
        return node


def make_src_mask(src: Tensor, pad_id: int = 0) -> Tensor:
    """
    Create encoder padding mask.

    Args:
        src: Tensor of shape (batch_size, src_len) containing token IDs.
        pad_id: ID of the padding token.

    Returns:
        A boolean mask of shape (batch_size, 1, 1, src_len), where True
        indicates positions that are *not* padding.
    """
    # src != pad_id gives True for real tokens
    non_pad = src.ne(pad_id)  # (B, S)
    # reshape to (B, 1, 1, S) for broadcasting in attention
    return non_pad.unsqueeze(1).unsqueeze(2)  # bool tensor


def make_tgt_mask(tgt: Tensor, pad_id: int = 0) -> Tensor:
    """
    Create decoder self-attention mask to hide padding and future tokens.

    Args:
        tgt: Tensor of shape (batch_size, tgt_len) containing token IDs.
        pad_id: ID of the padding token.

    Returns:
        A boolean mask of shape (batch_size, 1, tgt_len, tgt_len), where True
        indicates allowed attend positions.
    """
    batch_size, tgt_len = tgt.size()
    # Padding mask: True where not pad
    non_pad = tgt.ne(pad_id)  # (B, T)
    pad_mask = non_pad.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

    # Subsequent mask: lower triangle (including diagonal)
    # shape (T, T)
    subsequent = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device))
    # reshape to (1, 1, T, T)
    subsequent_mask = subsequent.unsqueeze(0).unsqueeze(0)

    # Combine masks: both padding and future masking
    return pad_mask & subsequent_mask  # (B, 1, T, T)


def positional_encoding(max_len: int, d_model: int) -> Tensor:
    """
    Generate sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length.
        d_model: Dimensionality of the model.

    Returns:
        A FloatTensor of shape (1, max_len, d_model) containing positional encodings.
    """
    # Position indices (max_len, 1)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # Dimension indices (1, d_model)
    dim = torch.arange(0, d_model, dtype=torch.float).unsqueeze(0)
    # Compute the angle rates
    angle_rates = 1.0 / (10000 ** ((2 * (dim // 2)) / d_model))  # (1, d_model)
    # Compute the angles
    angle_rads = pos * angle_rates  # (max_len, d_model)

    # Initialize positional encoding matrix
    pe = torch.zeros((max_len, d_model), dtype=torch.float)
    # Apply sin to even indices, cos to odd
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    # Add batch dimension
    pe = pe.unsqueeze(0)  # (1, max_len, d_model)
    return pe


def label_smoothing(one_hot: Tensor, eps: float) -> Tensor:
    """
    Apply label smoothing to one-hot targets.

    Args:
        one_hot: Tensor of shape (..., num_classes) with one-hot encoding.
        eps: Smoothing factor epsilon.

    Returns:
        A Tensor of same shape as one_hot with smoothed label distribution.
    """
    if one_hot.dim() < 1:
        raise ValueError("one_hot tensor must have at least 1 dimension")
    num_classes = one_hot.size(-1)
    if num_classes < 2:
        raise ValueError("Number of classes must be >= 2 for label smoothing")
    # On and off values
    on_value = 1.0 - eps
    off_value = eps / (num_classes - 1)
    # Smooth: one_hot * on_value + (1 - one_hot) * off_value
    return one_hot * on_value + (1.0 - one_hot) * off_value


def get_lr_scheduler(
    optimizer: Optimizer,
    d_model: int,
    warmup_steps: int
) -> LambdaLR:
    """
    Get Noam (inverse sqrt) learning rate scheduler.

    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

    Args:
        optimizer: Optimizer with initial lr=1.0.
        d_model: Model dimensionality.
        warmup_steps: Number of warmup steps.

    Returns:
        A torch.optim.lr_scheduler.LambdaLR scheduler.
    """
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be greater than zero")

    def lr_lambda(step: int) -> float:
        # step is 0-based in PyTorch; avoid division by zero
        if step <= 0:
            return 0.0
        arg1 = step ** -0.5
        arg2 = step * (warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(arg1, arg2)

    return LambdaLR(optimizer, lr_lambda)
