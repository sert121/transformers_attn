"""
trainer.py

Implements the Trainer class to orchestrate training of the Transformer model
with label smoothing, inverse-sqrt learning rate schedule with warmup, and
checkpointing as per 'Attention Is All You Need'.
"""

import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from dataset_loader import DatasetLoader
from utils import Utils
from model import TransformerModel


class Trainer:
    """
    Trainer for the Transformer model.

    Responsibilities:
      - Run training loop for a fixed number of steps
      - Apply label smoothing and compute loss over non-padding tokens
      - Use Adam optimizer with inverse sqrt learning-rate schedule
      - Save checkpoints evenly spaced for later averaging
    """

    def __init__(
        self,
        model: TransformerModel,
        optimizer: Optimizer,
        loader: DatasetLoader,
        config: Config
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (TransformerModel): The model to train.
            optimizer (Optimizer): The optimizer (Adam).
            loader (DatasetLoader): Data loader for train/dev splits.
            config (Config): Configuration object.
        """
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.config = config

        # Device inferred from model parameters
        self.device = next(model.parameters()).device

        # Learning rate scheduler: inverse sqrt with warmup
        warmup_steps = self.config.training.warmup_steps
        # config.lr_scale = d_model^{-0.5}
        def lr_lambda(step: int) -> float:
            # step starts from 0 in LambdaLR; shift to 1-based
            curr_step = step + 1
            return self.config.lr_scale * min(
                curr_step ** -0.5,
                curr_step * (warmup_steps ** -1.5)
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # Checkpointing: split max_steps into `checkpoint_average` intervals
        avg_ckpts = self.config.evaluation.checkpoint_average
        max_steps = self.config.training.max_steps
        # Ensure at least one step interval
        self.ckpt_interval = max(1, max_steps // avg_ckpts)

        # Padding token ID needed for masking loss
        self.pad_id = loader.pad_id

        # Checkpoint directory (default './checkpoints' if not in config)
        ckpt_dir = getattr(self.config.training, 'checkpoint_dir', 'checkpoints')
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)


    def train(self) -> None:
        """
        Run the training loop for `config.training.max_steps` steps.
        """
        self.model.train()
        train_loader = self.loader.load_data('train')

        max_steps = self.config.training.max_steps
        label_smoothing = self.config.training.label_smoothing
        vocab_size = self.config.data.spm_vocab_size

        step = 0
        for batch in train_loader:
            step += 1
            if step > max_steps:
                break

            # Move inputs to device
            src = batch['src'].to(self.device)                # (B, src_len)
            tgt_input = batch['tgt_input'].to(self.device)    # (B, tgt_len)
            tgt_output = batch['tgt_output'].to(self.device)  # (B, tgt_len)

            # Forward pass (model builds its own masks internally)
            logits = self.model(src, tgt_input)               # (B, tgt_len, V)

            # Compute label-smoothed loss
            B, T, V = logits.size()
            log_probs = F.log_softmax(logits, dim=-1)         # (B, T, V)
            log_probs = log_probs.view(-1, V)                # (B*T, V)
            target = tgt_output.contiguous().view(-1)        # (B*T,)

            # Build smoothed target distributions
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.scatter_(1, target.unsqueeze(1), 1.0)
                true_dist = true_dist * (1.0 - label_smoothing) + (label_smoothing / V)

            # Negative log-likelihood per token
            nll = - (true_dist * log_probs).sum(dim=1)       # (B*T,)

            # Mask out padding tokens in loss
            pad_mask = (target != self.pad_id)
            nll = nll[pad_mask]                              # Only non-pad tokens
            n_tokens = pad_mask.sum().item()
            loss = nll.sum() / max(1, n_tokens)

            # Backward and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Checkpointing
            if step % self.ckpt_interval == 0:
                self.save_checkpoint(step)

            # Logging every 100 steps and first step
            if step == 1 or step % 100 == 0:
                curr_lr = self.optimizer.param_groups[0]['lr']
                print(f"[Step {step}/{max_steps}] Loss: {loss.item():.4f}, LR: {curr_lr:.6e}")

        # Final checkpoint if not aligned with interval
        if step % self.ckpt_interval != 0:
            self.save_checkpoint(step)
        print("Training complete.")


    def save_checkpoint(self, step: int) -> None:
        """
        Save a checkpoint at the given step, including model, optimizer,
        scheduler states and current step number.

        Args:
            step (int): The training step number.
        """
        ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_{step}.pt")
        state: Dict[str, object] = {
            'step': step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        torch.save(state, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
