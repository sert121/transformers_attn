## 5. trainer.py

import os
import glob
import re
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    """Trainer encapsulates the training loop, checkpointing, and averaging for the Transformer."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        cfg: Config,
    ) -> None:
        """
        Args:
            model: TransformerModel instance.
            dataloader: DataLoader yielding batches for training.
            cfg: Config object with training hyperparameters.
        """
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        betas = tuple(self.cfg.get("training.optimizer.betas"))
        eps = self.cfg.get("training.optimizer.eps")
        self.optimizer = optim.Adam(
            self.model.parameters(), betas=betas, eps=eps, lr=1.0
        )

        # Noam scheduler
        d_model = self.cfg.get("training.scheduler.d_model")
        warmup_steps = self.cfg.get("training.scheduler.warmup_steps")

        def lr_lambda(step: int) -> float:
            # step starts from 1
            step = max(step, 1)
            return (d_model ** -0.5) * min(
                step ** -0.5, step * (warmup_steps ** -1.5)
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Training schedule
        self.max_steps: int = self.cfg.get("training.max_steps")
        self.save_interval: int = self.cfg.get("training.save_interval_steps")
        self.avg_last_n: int = self.cfg.get("training.average_last_checkpoints")

        # Label smoothing
        self.label_smoothing: float = self.cfg.get("training.label_smoothing", 0.0)
        # Optional gradient clipping
        self.max_grad_norm: Optional[float] = self.cfg.get(
            "training.max_grad_norm", None
        )

        # Checkpoint directory
        self.ckpt_dir: str = self.cfg.get("training.checkpoint_dir", "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self) -> None:
        """Run the training loop, save checkpoints, and average final checkpoints."""
        self.model.train()
        step = 0
        total_tokens = 0
        total_loss_sum = 0.0

        logger.info(
            f"Starting training for {self.max_steps} steps "
            f"(save every {self.save_interval} steps)."
        )
        # main loop
        while step < self.max_steps:
            for batch in self.dataloader:
                step += 1
                # Move batch to device
                src_ids = batch["src_ids"].to(self.device)
                src_mask = batch["src_mask"].to(self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)
                tgt_mask = batch["tgt_mask"].to(self.device)

                # Prepare decoder input and target output (shift by one)
                tgt_input = tgt_ids[:, :-1]
                tgt_output = tgt_ids[:, 1:]
                tgt_input_mask = tgt_mask[:, :-1]
                tgt_output_mask = tgt_mask[:, 1:]

                # Forward
                logits = self.model(
                    src_ids, tgt_input, src_mask=src_mask, tgt_mask=tgt_input_mask
                )  # (B, T, V)

                # Compute loss and backprop
                loss, n_tokens = self._compute_loss(
                    logits, tgt_output, tgt_output_mask
                )
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                # Step optimizer and scheduler
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Logging
                total_loss_sum += loss.item() * n_tokens
                total_tokens += n_tokens

                if step % self.save_interval == 0:
                    ckpt_path = self._save_checkpoint(step)
                    logger.info(f"Step {step}: saved checkpoint to {ckpt_path}")

                if step >= self.max_steps:
                    break
            # end for batch
        # end while

        avg_loss = total_loss_sum / total_tokens if total_tokens > 0 else float("nan")
        logger.info(
            f"Training complete. Steps={step}, average loss per token={avg_loss:.6f}"
        )

        # Checkpoint averaging
        pattern = os.path.join(self.ckpt_dir, "checkpoint_*.pt")
        all_ckpts = glob.glob(pattern)
        if len(all_ckpts) < self.avg_last_n:
            logger.warning(
                f"Found only {len(all_ckpts)} checkpoints, "
                f"less than avg_last_n={self.avg_last_n}. Skipping averaging."
            )
            return

        # Sort by step number
        def _extract_step(path: str) -> int:
            m = re.search(r"checkpoint_(\d+)\.pt$", path)
            return int(m.group(1)) if m else -1

        all_ckpts = sorted(all_ckpts, key=_extract_step)
        last_ckpts = all_ckpts[-self.avg_last_n :]
        avg_path = os.path.join(self.ckpt_dir, "averaged.pt")
        self._average_checkpoints(last_ckpts, avg_path)
        logger.info(f"Averaged {len(last_ckpts)} checkpoints into {avg_path}")

        # Load averaged weights into model
        ckpt = torch.load(avg_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        logger.info("Loaded averaged model weights.")

    def _compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: (B, T, V) raw output from the model.
            target: (B, T) gold token IDs for each position.
            mask: (B, T) bool mask, True for real tokens, False for padding.

        Returns:
            loss: scalar tensor (averaged per non-pad token).
            n_tokens: number of non-pad tokens included in loss.
        """
        bsz, seq_len, vocab_size = logits.size()
        # Flatten
        logits_flat = logits.view(-1, vocab_size)  # (B*T, V)
        target_flat = target.view(-1)  # (B*T,)
        mask_flat = mask.view(-1).to(logits.device)  # (B*T,)

        log_probs = nn.functional.log_softmax(logits_flat, dim=-1)  # (B*T, V)
        # Negative log-likelihood for true classes
        nll = -log_probs.gather(dim=-1, index=target_flat.unsqueeze(1)).squeeze(1)
        # Smooth loss (uniform)
        smooth = -log_probs.sum(dim=-1)

        # Mask out padding tokens
        nll = nll * mask_flat
        smooth = smooth * mask_flat
        n_tokens = int(mask_flat.sum().item())

        if n_tokens == 0:
            return torch.tensor(0.0, device=logits.device), 0

        nll_sum = nll.sum()
        smooth_sum = smooth.sum()
        eps = self.label_smoothing

        # Combined loss
        loss = (1.0 - eps) * nll_sum + (eps * smooth_sum / vocab_size)
        loss = loss / n_tokens
        return loss, n_tokens

    def _save_checkpoint(self, step: int) -> str:
        """
        Save model state_dict and step to a checkpoint file.

        Args:
            step: current training step.

        Returns:
            Path to the saved checkpoint.
        """
        path = os.path.join(self.ckpt_dir, f"checkpoint_{step}.pt")
        torch.save({"model": self.model.state_dict(), "step": step}, path)
        return path

    def _load_checkpoint_states(
        self, paths: List[str]
    ) -> List[dict]:
        """
        Load model state_dicts from a list of checkpoint paths.

        Args:
            paths: list of checkpoint file paths.

        Returns:
            List of state_dicts (model parameters).
        """
        states = []
        for p in paths:
            ckpt = torch.load(p, map_location="cpu")
            if "model" not in ckpt:
                raise KeyError(f"Checkpoint {p} missing 'model' key.")
            states.append(ckpt["model"])
        return states

    def _average_checkpoints(self, paths: List[str], out_path: str) -> None:
        """
        Average model parameters over multiple checkpoints and save to out_path.

        Args:
            paths: list of checkpoint file paths.
            out_path: output file path for the averaged checkpoint.
        """
        states = self._load_checkpoint_states(paths)
        if not states:
            raise ValueError("No checkpoints provided for averaging.")

        # Initialize average state
        avg_state = {}
        for key in states[0].keys():
            # Sum tensors across checkpoints
            stacked = torch.stack([state[key].float() for state in states], dim=0)
            avg_state[key] = (stacked.mean(dim=0)).type(states[0][key].dtype)

        # Save averaged state
        torch.save({"model": avg_state}, out_path)
