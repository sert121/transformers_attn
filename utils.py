"""Utility functions for reproducible training, checkpointing, and data loading."""

import os
import random
import logging
from typing import Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Configure root logger if not already configured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"
)
_logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): Seed value to use for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _logger.info(f"Random seed set to {seed} for python, numpy, and torch.")


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool
) -> DataLoader:
    """
    Wrap a torch Dataset into a DataLoader with sensible defaults.

    Args:
        dataset (torch.utils.data.Dataset): Dataset returning dicts or tensors.
        batch_size (int): Number of examples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader instance.
    """
    num_workers_env = os.environ.get("NUM_WORKERS")
    try:
        num_workers = int(num_workers_env) if num_workers_env is not None else 4
    except ValueError:
        num_workers = 4
    pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None  # Assumes dataset returns already-tensorized, padded batches
    )
    _logger.debug(
        f"Created DataLoader(dataset={type(dataset).__name__}, "
        f"batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, pin_memory={pin_memory})"
    )
    return loader


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_dir: str
) -> str:
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        step (int): Training step number, used in filename.
        ckpt_dir (str): Directory in which to save checkpoints.

    Returns:
        str: Full path to the saved checkpoint file.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = f"checkpoint_step_{step}.pt"
    ckpt_path = os.path.join(ckpt_dir, filename)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step
    }
    try:
        torch.save(state, ckpt_path)
        _logger.info(f"Saved checkpoint at step {step} to {ckpt_path}.")
    except Exception as e:
        _logger.error(f"Failed to save checkpoint at {ckpt_path}: {e}")
        raise
    return ckpt_path


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Optional[int]:
    """
    Load model (and optionally optimizer) state from a checkpoint file.

    Args:
        ckpt_path (str): Path to checkpoint file (.pt).
        model (torch.nn.Module): Model into which to load the state.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore state.

    Returns:
        Optional[int]: Loaded training step, or None if not present.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint.get("step", None)
        _logger.info(f"Loaded checkpoint from {ckpt_path} at step {step}.")
        return step
    except Exception as e:
        _logger.error(f"Error loading checkpoint from {ckpt_path}: {e}")
        raise
