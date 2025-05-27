"""
trainer.py

Implements the training loop for the Transformer model as specified in
'Attention Is All You Need'. Uses token-based batching, Adam optimizer with
warmup-inverse-sqrt learning rate schedule, label smoothing, and periodic
checkpointing.
"""

import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import utils


class Trainer:
    """
    Trainer for the Transformer model.

    Public methods:
        __init__(model, datasets, config)
        train() -> str  # returns path to the last saved checkpoint
    """

    def __init__(
        self,
        model: torch.nn.Module,
        datasets: Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset],
        config: object
    ):
        """
        Args:
            model (torch.nn.Module): TransformerModel instance.
            datasets (tuple): (train_dataset, dev_dataset)
            config: Config object with .get(section: str) -> dict interface.
        """
        self.config = config
        self.model = model

        # Unpack datasets
        self.train_dataset, self.dev_dataset = datasets

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            # Wrap for multi-GPU
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # Load training hyperparameters
        training_cfg = config.get("training")
        self.max_steps = int(training_cfg.get("max_steps", 100000))
        self.batch_tokens = int(training_cfg.get("batch_tokens", 25000))
        self.save_steps = int(training_cfg.get("save_steps", 10000))
        # eval_steps may be unused here; evaluation happens outside Trainer
        self.eval_steps = int(training_cfg.get("eval_steps", 10000))

        # Optimizer settings
        optim_cfg = config.get("optimizer")
        betas = tuple(optim_cfg.get("betas", [0.9, 0.98]))
        eps = float(optim_cfg.get("epsilon", 1e-9))

        # Scheduler settings
        sched_cfg = config.get("scheduler")
        self.warmup_steps = int(sched_cfg.get("warmup_steps", 4000))

        # Model dimension (for LR schedule) and label smoothing
        model_cfg = config.get("model")
        self.d_model = float(model_cfg.get("d_model", 512))
        reg_cfg = config.get("regularization")
        self.eps_ls = float(reg_cfg.get("label_smoothing", 0.1))

        # Checkpoint directory
        logging_cfg = config.get("logging")
        self.checkpoint_dir = logging_cfg.get("checkpoint_dir", "checkpoints/")

        # DataLoaders
        self.train_loader = utils.make_dataloader(
            self.train_dataset, self.batch_tokens, shuffle=True
        )
        self.dev_loader = utils.make_dataloader(
            self.dev_dataset, self.batch_tokens, shuffle=False
        )

        # Optimizer and learning rate scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0,
            betas=betas,
            eps=eps
        )

        def lr_lambda(step: int) -> float:
            """Warmup-inverse-sqrt learning rate schedule."""
            # avoid div by zero
            step_val = max(step, 1)
            return (self.d_model ** -0.5) * min(
                step_val ** -0.5,
                step_val * (self.warmup_steps ** -1.5)
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Training state
        self.global_step = 0

        # Padding token index (assumed 0 by embedding padding_idx)
        self.pad_idx = 0

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional label smoothing.

        Args:
            logits (Tensor): shape (batch_size, tgt_len, vocab_size)
            labels (Tensor): shape (batch_size, tgt_len)

        Returns:
            Tensor: scalar loss.
        """
        batch_size, tgt_len, vocab_size = logits.size()
        # Flatten
        logits_flat = logits.view(-1, vocab_size)      # (N, V)
        labels_flat = labels.view(-1)                  # (N,)

        # Mask padding positions
        mask = labels_flat.ne(self.pad_idx)            # (N,)
        if self.eps_ls > 0.0:
            # Log-probabilities
            log_probs = F.log_softmax(logits_flat, dim=-1)
            # Select non-pad positions
            log_probs = log_probs[mask]                # (N_valid, V)
            targets = labels_flat[mask].unsqueeze(1)   # (N_valid, 1)

            # One-hot and smooth
            smooth_dist = torch.full_like(
                log_probs,
                fill_value=self.eps_ls / float(vocab_size)
            )
            smooth_dist.scatter_(
                dim=1,
                index=targets,
                value=1.0 - self.eps_ls
            )
            # Cross-entropy = -sum(P * log Q) / N_valid
            loss = -(smooth_dist * log_probs).sum(dim=-1).mean()
        else:
            # Standard cross-entropy ignoring pad
            loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=self.pad_idx
            )
        return loss

    def train(self) -> str:
        """
        Run training loop for max_steps and save checkpoints every save_steps.

        Returns:
            str: Path to the last saved checkpoint.
        """
        self.model.train()
        last_ckpt_path = ""

        pbar = tqdm(total=self.max_steps, desc="Training", unit="step")
        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.max_steps:
                    break

                # Move data to device
                src_ids = batch["input_ids"].to(self.device)
                tgt_ids = batch["labels"].to(self.device)

                # Forward pass
                logits = self.model(src_ids, tgt_ids)
                loss = self.compute_loss(logits, tgt_ids)

                # Backward and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Update step and progress bar
                self.global_step += 1
                pbar.update(1)
                pbar.set_postfix(
                    step=self.global_step,
                    loss=f"{loss.item():.4f}"
                )

                # Checkpointing
                if self.global_step % self.save_steps == 0:
                    last_ckpt_path = utils.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.global_step,
                        self.checkpoint_dir
                    )
            # end for batch
        # end while

        pbar.close()
        return last_ckpt_path
