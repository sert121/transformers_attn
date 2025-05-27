## trainer.py

import os
import math
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Config, get_lr_scheduler, label_smoothing
from model import TransformerModel


class Trainer:
    """
    Trainer class to handle training and validation loops for the Transformer model.
    """

    def __init__(
        self,
        model: TransformerModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model: The Transformer model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            config: Configuration object.
        """
        # Model and data
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer setup
        optim_cfg = config.get("optimizer")
        opt_type = optim_cfg.get("type", "Adam").lower()
        beta1 = float(optim_cfg.get("beta1", 0.9))
        beta2 = float(optim_cfg.get("beta2", 0.98))
        eps = float(optim_cfg.get("epsilon", 1e-9))
        if opt_type != "adam":
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=(beta1, beta2),
            eps=eps
        )

        # Learning rate scheduler
        lr_cfg = config.get("lr_scheduler")
        d_model = int(lr_cfg.get("d_model", config.get("model.d_model", 512)))
        warmup_steps = int(lr_cfg.get("warmup_steps", config.get("training.warmup_steps", 4000)))
        self.scheduler = get_lr_scheduler(self.optimizer, d_model, warmup_steps)

        # Label smoothing
        self.label_smoothing_eps = float(config.get("training.label_smoothing", 0.0))

        # Padding token id (assumed 0 unless overridden)
        self.pad_idx = int(config.get("data.pad_id", 0))

        # Training parameters
        self.max_steps = int(config.get("training.max_steps", 100000))
        # how often (in steps) to run validation
        self.eval_interval = int(config.get("training.eval_interval", 5000))

        # Checkpointing
        self.ckpt_dir = config.get("training.ckpt_dir", "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_ppl = float("inf")

        # Internal step counter
        self.step = 0

    def train(self) -> None:
        """
        Run the full training loop, including periodic validation
        and checkpointing.
        """
        self.model.train()
        progress_bar = tqdm(total=self.max_steps, desc="Training", unit="step")
        # Loop until we reach max_steps
        while self.step < self.max_steps:
            for batch in self.train_loader:
                if self.step >= self.max_steps:
                    break
                self.step += 1
                loss = self._train_step(batch)
                lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix(
                    loss=f"{loss:.4f}",
                    lr=f"{lr:.6f}"
                )
                progress_bar.update(1)

                # Validation & checkpointing
                if (self.step % self.eval_interval == 0) or (self.step == self.max_steps):
                    val_metrics = self._validate()
                    val_ppl = val_metrics["ppl"]
                    is_best = val_ppl < self.best_ppl
                    if is_best:
                        self.best_ppl = val_ppl
                        ckpt_path = os.path.join(
                            self.ckpt_dir,
                            f"ckpt_step{self.step}_ppl{val_ppl:.2f}.pt"
                        )
                        self.model.save(ckpt_path)
                    print(
                        f"\n[Step {self.step}] Validation Perplexity: {val_ppl:.4f} "
                        f"(best: {self.best_ppl:.4f}) "
                        f"{'[Saved]' if is_best else ''}\n"
                    )
            # end for batch
        # end while
        progress_bar.close()
        print("Training complete.")

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single training step (forward, loss, backward, update).

        Args:
            batch: A dict containing 'src' and 'tgt' tensors.

        Returns:
            The scalar loss value for this step.
        """
        # Unpack and move to device
        src = batch["src"].to(self.device)   # (B, S)
        tgt = batch["tgt"].to(self.device)   # (B, T)

        # Prepare target input and output
        # e.g., input: [<s>, ... , x_{T-1}], output: [..., x_T, </s>]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        logits = self.model(src, tgt_input)  # (B, T-1, V)
        B, Tm1, V = logits.size()

        # Flatten for loss
        logits_flat = logits.contiguous().view(-1, V)       # (B*(T-1), V)
        labels_flat = tgt_output.contiguous().view(-1)      # (B*(T-1),)

        # Compute mask and token count
        non_pad_mask = labels_flat.ne(self.pad_idx)
        n_tokens = non_pad_mask.sum().item()
        if n_tokens == 0:
            # nothing to learn on this batch
            return 0.0

        # Loss computation
        if self.label_smoothing_eps > 0.0:
            # KL divergence with smoothed targets
            # build one-hot
            with torch.no_grad():
                one_hot = torch.zeros_like(logits_flat).scatter_(
                    1, labels_flat.unsqueeze(1), 1.0
                )
                smooth = label_smoothing(one_hot, self.label_smoothing_eps)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = F.kl_div(log_probs, smooth, reduction="sum")
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=self.pad_idx,
                reduction="sum"
            )
        # Normalize by number of non-pad tokens
        loss = loss / n_tokens

        # Backprop & update
        self.optimizer.zero_grad()
        loss.backward()
        # Optionally: gradient clipping here if needed
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def _validate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set to compute perplexity.

        Returns:
            A dict with key 'ppl' and the computed perplexity.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = self.model(src, tgt_input)  # (B, T-1, V)
                B, Tm1, V = logits.size()
                logits_flat = logits.contiguous().view(-1, V)
                labels_flat = tgt_output.contiguous().view(-1)

                # sum-cross-entropy
                loss_sum = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    ignore_index=self.pad_idx,
                    reduction="sum"
                ).item()

                # count valid tokens
                n_tokens = labels_flat.ne(self.pad_idx).sum().item()
                total_loss += loss_sum
                total_tokens += n_tokens

        # restore train mode
        self.model.train()

        # compute perplexity
        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(avg_loss)
        return {"ppl": ppl}
