# trainer.py

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import Config
from utils import create_masks


class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.

    Args:
        vocab_size: Size of the vocabulary.
        smoothing: Smoothing factor ε_ls.
        pad_idx: Index of padding token to ignore.
    """

    def __init__(self, vocab_size: int, smoothing: float, pad_idx: int) -> None:
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0,1)"
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        # KLDivLoss expects log-probabilities and smoothed target distributions
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            logits: Tensor of shape (B, T, V) - raw model outputs.
            target: Tensor of shape (B, T)   - true token indices.

        Returns:
            Scalar loss (averaged over non-pad tokens).
        """
        batch_size, seq_len, vocab_size = logits.size()
        assert vocab_size == self.vocab_size

        # Flatten inputs
        n_tokens = batch_size * seq_len
        log_probs = F.log_softmax(logits.view(-1, vocab_size), dim=-1)  # (N, V)
        target_flat = target.view(-1)  # (N,)

        # Build smoothed targets
        with torch.no_grad():
            # Initialize with uniform smoothing for all non-pad classes
            true_dist = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))
            # Put confidence on the true labels
            true_dist.scatter_(1, target_flat.unsqueeze(1), self.confidence)
            # Zero out padding token distributions
            true_dist[:, self.pad_idx] = 0
            # Mask out positions where target is pad
            pad_mask = target_flat == self.pad_idx
            if pad_mask.any():
                true_dist[pad_mask, :] = 0.0

        # Compute KL divergence loss and normalize by number of non-pad tokens
        loss = self.kl_div(log_probs, true_dist)
        non_pad = (target_flat != self.pad_idx).sum().clamp_min(1)
        return loss / non_pad


class Trainer:
    """
    Trainer for the Transformer model.

    Public methods:
      - train() -> None
      - save_checkpoint(path: str) -> None
      - load_checkpoint(path: str) -> Optional[int]
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Args:
            model:        The TransformerModel instance.
            config:       Configuration object.
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation data.
        """
        # Core objects
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Extract training parameters
        train_params = self.config.get_training_params()
        self.train_steps = train_params["train_steps"]
        # Directories
        data_paths = self.config.get_data_paths()
        self.checkpoint_dir = os.path.join(data_paths["output_dir"], "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Optimizer: Adam with custom betas and epsilon
        opt_cfg = self.config._cfg["optimizer"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            betas=(float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.98))),
            eps=float(opt_cfg.get("epsilon", 1e-9)),
        )

        # Learning rate scheduler: Transformer schedule
        sched_cfg = self.config._cfg["scheduler"]
        d_model = float(sched_cfg.get("d_model", self.config.get_model_params()["d_model"]))
        warmup_steps = int(sched_cfg.get("warmup_steps", 4000))

        def lr_lambda(step: int) -> float:
            # step starts from 1
            step = max(step, 1)
            return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # Label smoothing criterion
        reg_cfg = self.config._cfg["regularization"]
        eps_ls = float(reg_cfg.get("label_smoothing", 0.1))
        # Model should expose vocab size and pad_idx attributes
        vocab_size = int(getattr(self.model, "vocab_size", None))
        pad_idx = int(getattr(self.model, "pad_idx", 0))
        self.criterion = LabelSmoothingLoss(vocab_size, eps_ls, pad_idx)

        # Logging and checkpoint intervals (defaults)
        self.log_interval = int(train_params.get("log_interval", 100))
        self.ckpt_interval = int(train_params.get("ckpt_interval", 1000))

    def train(self) -> None:
        """
        Runs the training loop for the configured number of steps.
        Logs loss and perplexity periodically and checkpoints the model.
        """
        self.model.train()
        train_iter = iter(self.train_loader)
        start_time = time.time()

        for step in range(1, self.train_steps + 1):
            # Fetch next batch (wrap-around)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Unpack and move to device
            src = batch["src"].to(self.device)
            tgt_input = batch["tgt_input"].to(self.device)
            tgt_output = batch["tgt_output"].to(self.device)

            # Masks
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx=self.model.pad_idx)
            src_mask = src_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)

            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask)  # (B, T, V)

            # Compute loss
            loss = self.criterion(logits, tgt_output)

            # Backward and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Logging
            if step % self.log_interval == 0 or step == 1:
                elapsed = time.time() - start_time
                lr = self.scheduler.get_last_lr()[0]
                ppl = torch.exp(loss).item()
                print(
                    f"[Step {step:>6}/{self.train_steps}] "
                    f"Loss: {loss.item():.4f}  PPL: {ppl:>.2f}  LR: {lr:.6e}  "
                    f"Time: {elapsed:.1f}s"
                )
                start_time = time.time()

            # Checkpointing
            if step % self.ckpt_interval == 0 or step == self.train_steps:
                ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_step{step}.pt")
                self.save_checkpoint(ckpt_path)

        print("Training complete.")

    def save_checkpoint(self, path: str) -> None:
        """
        Saves model, optimizer, scheduler states and current step.

        Args:
            path: File path to save the checkpoint.
        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(state, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Optional[int]:
        """
        Loads states from a checkpoint.

        Args:
            path: Path to the checkpoint file.
        Returns:
            The training step at which the checkpoint was saved, if available.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"Loaded checkpoint from {path}")
        # If step was stored, return it; else, return None
        return checkpoint.get("step", None)
