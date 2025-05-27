# trainer.py

import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
import utils
from model import TransformerModel


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss as described in 'Attention Is All You Need'."""

    def __init__(
        self,
        smoothing: float,
        vocab_size: int,
        ignore_index: int = 0,
    ) -> None:
        """
        Args:
            smoothing: Label smoothing factor (epsilon).
            vocab_size: Size of the vocabulary.
            ignore_index: Padding token ID to ignore in loss.
        """
        super(LabelSmoothingLoss, self).__init__()
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError("Smoothing value must be in [0, 1].")
        self.smoothing: float = smoothing
        self.confidence: float = 1.0 - smoothing
        self.vocab_size: int = vocab_size
        self.ignore_index: int = ignore_index

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the label-smoothed cross-entropy loss.

        Args:
            pred: Logits of shape (batch*seq_len, vocab_size).
            target: Ground truth indices of shape (batch*seq_len,).

        Returns:
            Scalar loss tensor.
        """
        # Compute log probabilities
        log_prob = F.log_softmax(pred, dim=-1)  # (N, V)
        # Create the smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            # Scatter the confidence scores at target positions
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # Zero out padding positions
            mask = target.eq(self.ignore_index)
            if mask.any():
                true_dist[mask, :] = 0.0

        # KL divergence between true distribution and log probabilities
        # Sum over classes, then average over non-ignored positions
        loss = (-true_dist * log_prob).sum(dim=1)
        non_pad_mask = target.ne(self.ignore_index)
        if non_pad_mask.any():
            loss = loss.masked_select(non_pad_mask).mean()
        else:
            loss = loss.mean()
        return loss


class Trainer:
    """Trainer for the Transformer model."""

    def __init__(
        self,
        config: Config,
        model: TransformerModel,
        dataloaders: Dict[str, DataLoader],
    ) -> None:
        """
        Args:
            config: Configuration object.
            model: The TransformerModel to train.
            dataloaders: Dict with 'train' (and optionally 'val', 'test') DataLoaders.
        """
        self.config = config
        self.model = model
        # DataLoaders
        if "train" not in dataloaders:
            raise ValueError("Trainer requires a 'train' DataLoader.")
        self.train_loader: DataLoader = dataloaders["train"]
        self.val_loader: Optional[DataLoader] = dataloaders.get("val", None)
        # Device
        self.device = utils.get_device(self.config)
        self.model.to(self.device)

        # Optimizer
        opt_type = config.get("optimizer.type", "Adam")
        betas = tuple(config.get("optimizer.betas", [0.9, 0.98]))
        eps = config.get("optimizer.eps", 1e-9)
        if opt_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), betas=betas, eps=eps
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        # Learning rate scheduler (Noam)
        warmup_steps = config.get("learning_rate_scheduler.warmup_steps", 4000)
        d_model = config.get("model.d_model", 512)
        self.scheduler = utils.NoamScheduler(
            optimizer=self.optimizer,
            model_size=d_model,
            warmup_steps=warmup_steps,
        )

        # Label smoothing loss
        label_smoothing = config.get("training.label_smoothing", 0.0)
        vocab_size = config.get("data.vocab.size", 0)
        pad_id = getattr(self.model, "pad_id", 0)
        self.criterion = LabelSmoothingLoss(
            smoothing=label_smoothing,
            vocab_size=vocab_size,
            ignore_index=pad_id,
        )

        # Training state
        self.step: int = 0
        self.total_steps: int = config.get("training.total_steps", 100000)
        # Logging & checkpointing intervals
        self.log_interval: int = config.get("training.log_interval", 100)
        # If checkpoint_interval not set, skip intermediate checkpoints
        self.checkpoint_interval: Optional[int] = config.get(
            "training.checkpoint_interval", None
        )
        # Directory to save checkpoints
        self.checkpoint_dir: str = config.get(
            "training.checkpoint_dir", "checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self) -> None:
        """Runs the training loop."""
        self.model.train()
        print(f"Starting training for {self.total_steps} steps...")
        while self.step < self.total_steps:
            for batch in self.train_loader:
                if self.step >= self.total_steps:
                    break

                # Unpack batch
                src = batch["src"].to(self.device)               # (B, src_len)
                tgt = batch["tgt"].to(self.device)               # (B, tgt_len)
                src_mask = batch["src_mask"].to(self.device)     # (B, 1, 1, src_len)
                tgt_mask_full = batch["tgt_mask"].to(self.device)  # (B, 1, tgt_len, tgt_len)

                # Prepare decoder input and target output
                # Input: all tokens except last; Output: all tokens except first
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:].contiguous()
                # Adjust mask to match input length
                tgt_mask = tgt_mask_full[:, :, :-1, :-1]

                # Forward pass
                logits = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                )  # (B, tgt_len-1, vocab_size)

                # Compute loss
                batch_size, seq_len, vocab_size = logits.size()
                pred = logits.view(-1, vocab_size)
                gold = tgt_output.view(-1)
                loss = self.criterion(pred, gold)

                # Backward and optimization step
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.step += 1

                # Logging
                if self.step % self.log_interval == 0 or self.step == 1:
                    ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"[Step {self.step}/{self.total_steps}] "
                        f"Loss: {loss.item():.4f} "
                        f"PPL: {ppl:.2f} "
                        f"LR: {lr:.2e}"
                    )

                # Intermediate checkpoint
                if (
                    self.checkpoint_interval is not None
                    and self.step % self.checkpoint_interval == 0
                ):
                    ckpt_path = os.path.join(
                        self.checkpoint_dir, f"ckpt_step{self.step}.pt"
                    )
                    self.save_checkpoint(ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")

            # end for batch
        # end while

        # Final checkpoint
        final_ckpt = os.path.join(self.checkpoint_dir, "ckpt_final.pt")
        self.save_checkpoint(final_ckpt)
        print(f"Training completed. Final checkpoint saved: {final_ckpt}")

    def save_checkpoint(self, path: str) -> None:
        """
        Saves model, optimizer, scheduler state, and current step.

        Args:
            path: File path to save checkpoint.
        """
        utils.save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Loads training state from checkpoint.

        Args:
            path: File path of checkpoint to load.
        """
        loaded_step = utils.load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.step = loaded_step
        print(f"Loaded checkpoint '{path}' at step {self.step}")
