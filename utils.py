## utils.py

import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from config import Config


def get_device(config: Config) -> torch.device:
    """Selects the appropriate device ('cuda' if GPUs available and requested, else 'cpu').

    Args:
        config: Config object with hyperparameters.

    Returns:
        A torch.device object.
    """
    gpu_count = config.get("hardware.gpus", 0)
    if isinstance(gpu_count, int) and gpu_count > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_padding_mask(seq: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Creates a padding mask for a batch of sequences.

    The mask indicates positions that are padding (equal to pad_token_id).

    Args:
        seq: Tensor of shape (batch_size, seq_len) containing token IDs.
        pad_token_id: The ID used for padding tokens.

    Returns:
        A mask tensor of shape (batch_size, 1, 1, seq_len) with dtype bool,
        where True indicates the position is a pad token.
    """
    # seq == pad_token_id --> True where padding
    mask = seq.eq(pad_token_id)  # (batch_size, seq_len)
    # Expand to (batch_size, 1, 1, seq_len) for broadcasting
    return mask.unsqueeze(1).unsqueeze(1)


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """Creates a look-ahead mask to mask future positions in a sequence.

    The mask has True in positions where j > i (future positions).

    Args:
        size: The length of the sequence.

    Returns:
        A mask tensor of shape (size, size) with dtype bool.
    """
    # torch.triu returns upper triangular part of matrix including diagonal if diagonal=0
    mask = torch.triu(torch.ones((size, size), dtype=torch.bool), diagonal=1)
    return mask  # True in positions to mask


def create_combined_mask(
    tgt_seq: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    """Combines padding and look-ahead masks for target sequences.

    Args:
        tgt_seq: Tensor of shape (batch_size, tgt_seq_len) containing target token IDs.
        pad_token_id: The ID used for padding tokens.

    Returns:
        A combined mask tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        with dtype bool, where True indicates positions that should be masked.
    """
    batch_size, seq_len = tgt_seq.size()
    # Padding mask: (batch_size, 1, 1, seq_len)
    padding_mask = create_padding_mask(tgt_seq, pad_token_id)
    # Look-ahead mask: (seq_len, seq_len) -> expand to (1, 1, seq_len, seq_len)
    look_ahead = create_look_ahead_mask(seq_len).to(tgt_seq.device)
    look_ahead = look_ahead.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,seq_len)
    # Combine masks: broadcast padding_mask to (batch_size,1,seq_len,seq_len)
    combined = padding_mask | look_ahead
    return combined


def initialize_weights(module: nn.Module) -> None:
    """Initializes weights of transformer modules.

    Uses Xavier uniform for Linear and Embedding layers,
    zeros for biases, and sets LayerNorm weights to 1 and biases to 0.

    Args:
        module: A torch.nn.Module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[Any],
    step: int,
) -> None:
    """Saves training checkpoint to disk.

    Args:
        path: File path to save checkpoint.
        model: The model to save.
        optimizer: The optimizer whose state to save.
        scheduler: The learning-rate scheduler whose state to save (optional).
        step: The current training step.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint: Dict[str, Any] = {
        "step": step,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        # Some schedulers may not have state_dict
        try:
            checkpoint["sched_state"] = scheduler.state_dict()
        except AttributeError:
            checkpoint["sched_state"] = None
    else:
        checkpoint["sched_state"] = None

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> int:
    """Loads training checkpoint from disk.

    Args:
        path: File path to load checkpoint.
        model: The model to load state into.
        optimizer: The optimizer to load state into (optional).
        scheduler: The scheduler to load state into (optional).
        device: Device to map the checkpoint tensors to (optional).

    Returns:
        The training step at which the checkpoint was saved.
    """
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optim_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state"])
    if scheduler is not None and checkpoint.get("sched_state", None) is not None:
        try:
            scheduler.load_state_dict(checkpoint["sched_state"])
        except Exception:
            # scheduler may not implement load_state_dict
            pass

    return int(checkpoint.get("step", 0))


class NoamScheduler:
    """Implements the Noam learning rate schedule from Vaswani et al. (2017).

    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
    """

    def __init__(
        self, optimizer: Optimizer, model_size: int, warmup_steps: int
    ) -> None:
        """
        Args:
            optimizer: Wrapped optimizer.
            model_size: The d_model value (embedding dimension).
            warmup_steps: Number of warmup steps.
        """
        self.optimizer: Optimizer = optimizer
        self.model_size: float = float(model_size)
        self.warmup_steps: float = float(warmup_steps)
        self._step: int = 0

    def step(self) -> float:
        """Updates the learning rate and steps the optimizer.

        Returns:
            The new learning rate.
        """
        self._step += 1
        # Compute scale factor
        arg1 = self._step ** -0.5
        arg2 = self._step * (self.warmup_steps ** -1.5)
        scale = (self.model_size ** -0.5) * min(arg1, arg2)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = scale

        return scale

    def state_dict(self) -> Dict[str, Any]:
        """Returns the scheduler state."""
        return {"step": self._step}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Loads the scheduler state."""
        if "step" in state:
            self._step = int(state["step"])
