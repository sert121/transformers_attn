## evaluation.py

"""
Evaluation module for the Transformer.
Loads and averages model checkpoints, performs batched beam search decoding
on a specified data split, and computes corpus-level BLEU score.
"""

import os
import glob
import re
from typing import Dict, List

import torch
import torch.nn.functional as F

from config import Config
from model import TransformerModel
from dataset_loader import DatasetLoader
from utils import Utils


class Evaluation:
    """
    Evaluation harness for a trained Transformer model.

    Attributes:
        model (TransformerModel): The Transformer to evaluate.
        loader (DatasetLoader): Data loader for tokenization and batching.
        config (Config): Experiment configuration.
        device (torch.device): Device on which the model resides.
    """

    def __init__(
        self,
        model: TransformerModel,
        loader: DatasetLoader,
        config: Config
    ) -> None:
        """
        Initialize Evaluation, average the last N checkpoints, and load into model.

        Args:
            model (TransformerModel): The model architecture.
            loader (DatasetLoader): Provides data and tokenizer.
            config (Config): Experiment configuration.
        """
        self.model = model
        self.loader = loader
        self.config = config

        # Determine device from the model parameters
        self.device = next(self.model.parameters()).device

        # Determine checkpoint directory (must match Trainer)
        ckpt_dir = getattr(self.config.training, "checkpoint_dir", "checkpoints")
        pattern = os.path.join(ckpt_dir, "checkpoint_*.pt")
        paths = glob.glob(pattern)
        if not paths:
            raise FileNotFoundError(f"No checkpoints found with pattern: {pattern}")

        # Sort checkpoint paths by step number extracted from filename
        def _step_from_path(path: str) -> int:
            m = re.search(r"checkpoint_(\d+)\.pt$", path)
            if not m:
                return 0
            return int(m.group(1))

        paths = sorted(paths, key=_step_from_path)
        # Select the last N checkpoints to average
        num_avg = self.config.evaluation.checkpoint_average
        selected = paths[-num_avg:]

        # Load and collect state_dicts
        state_dicts = []
        for ckpt_path in selected:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            # Expect saved dict has 'model_state' key
            if "model_state" not in checkpoint:
                raise KeyError(f"'model_state' not found in checkpoint {ckpt_path}")
            state_dicts.append(checkpoint["model_state"])

        # Average the parameters across checkpoints
        avg_state: Dict[str, torch.Tensor] = {}
        param_keys = state_dicts[0].keys()
        for key in param_keys:
            # Sum tensors for this key
            summed = sum(sd[key] for sd in state_dicts)
            avg_state[key] = summed / float(len(state_dicts))

        # Load averaged parameters into model
        self.model.load_state_dict(avg_state)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, split: str) -> Dict[str, float]:
        """
        Run beam search decoding on the specified split and compute BLEU.

        Args:
            split (str): One of "dev" or "test", matching config.data.<split>_src/tgt.

        Returns:
            Dict[str, float]: Dictionary containing "BLEU" score.
        """
        all_hypotheses: List[str] = []
        all_references: List[str] = []

        # Disable gradient computation
        self.model.eval()
        with torch.no_grad():
            dataloader = self.loader.load_data(split)
            for batch in dataloader:
                src_batch = batch["src"].to(self.device)            # (B, src_len)
                tgt_out_batch = batch["tgt_output"].to(self.device) # (B, tgt_len)
                batch_size = src_batch.size(0)

                for i in range(batch_size):
                    # Extract single example
                    src_i = src_batch[i].unsqueeze(0)     # (1, src_len)
                    ref_ids = tgt_out_batch[i].tolist()   # List[int]

                    # Trim reference at EOS and remove padding
                    eos_id = self.loader.eos_id
                    if eos_id in ref_ids:
                        idx = ref_ids.index(eos_id)
                        ref_seq = ref_ids[:idx]
                    else:
                        # Exclude pad tokens if no EOS found
                        pad_id = self.loader.pad_id
                        ref_seq = [tid for tid in ref_ids if tid != pad_id]

                    # Decode reference text
                    # SentencePieceProcessor has DecodeIds method
                    try:
                        ref_text = self.loader.sp.DecodeIds(ref_seq)
                    except AttributeError:
                        # Fallback to generic decode API
                        ref_text = self.loader.sp.decode(ref_seq)  # type: ignore

                    # Perform beam search to get hypothesis token IDs
                    hyp_ids = self._beam_search(src_i)
                    # Decode hypothesis text
                    try:
                        hyp_text = self.loader.sp.DecodeIds(hyp_ids)
                    except AttributeError:
                        hyp_text = self.loader.sp.decode(hyp_ids)  # type: ignore

                    all_references.append(ref_text)
                    all_hypotheses.append(hyp_text)

        # Compute corpus-level BLEU
        bleu_score = Utils.compute_bleu(all_hypotheses, all_references)
        return {"BLEU": bleu_score}

    def _beam_search(self, src: torch.Tensor) -> List[int]:
        """
        Perform beam search decoding for a single source sequence.

        Args:
            src (torch.Tensor): Tensor of shape (1, src_len) containing source IDs.

        Returns:
            List[int]: Decoded token ID sequence (without BOS/EOS).
        """
        # Hyperparameters
        beam_size = self.config.evaluation.beam_size
        length_penalty = self.config.evaluation.length_penalty
        max_offset = self.config.evaluation.max_output_offset

        # Special tokens
        bos_id = self.loader.bos_id
        eos_id = self.loader.eos_id

        # Maximum generation length
        src_len = src.size(1)
        max_len = src_len + max_offset

        # Initialize beams: each is (sequence, score)
        beams: List[Tuple[List[int], float]] = [([bos_id], 0.0)]

        for _ in range(max_len):
            all_candidates: List[Tuple[List[int], float]] = []

            # Expand each beam
            for seq, score in beams:
                # If already ended with EOS, carry over
                if seq[-1] == eos_id:
                    all_candidates.append((seq, score))
                    continue

                # Prepare input tensor for decoder
                tgt_input = torch.tensor([seq], dtype=torch.long, device=self.device)
                # Forward pass: get logits
                logits = self.model(src, tgt_input)  # (1, seq_len, V)
                # Compute log-probs of last time step
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)  # (V,)

                # Get top-K next tokens
                topk_logps, topk_ids = torch.topk(log_probs, beam_size)
                topk_logps = topk_logps.tolist()
                topk_ids = topk_ids.tolist()

                # Form new candidate beams
                for logp, token_id in zip(topk_logps, topk_ids):
                    new_seq = seq + [token_id]
                    new_score = score + logp
                    all_candidates.append((new_seq, new_score))

            # Apply length penalty and select top beams
            scored_candidates: List[Tuple[List[int], float]] = []
            for seq, sc in all_candidates:
                # length penalty as in GNMT: ((5+|seq|)/6)^alpha
                lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                adjusted = sc / lp
                scored_candidates.append((seq, sc, adjusted))

            # Keep top beam_size by adjusted score
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(seq, sc) for seq, sc, _ in scored_candidates[:beam_size]]

            # If all beams have ended, we can stop early
            if all(seq[-1] == eos_id for seq, _ in beams):
                break

        # Select best final beam by adjusted score
        best_seq, best_score = beams[0]
        best_adjusted = best_score / (((5.0 + len(best_seq)) / 6.0) ** length_penalty)
        for seq, sc in beams[1:]:
            adj = sc / (((5.0 + len(seq)) / 6.0) ** length_penalty)
            if adj > best_adjusted:
                best_seq, best_score, best_adjusted = seq, sc, adj

        # Strip BOS and EOS tokens
        # Remove initial BOS
        if best_seq and best_seq[0] == bos_id:
            best_seq = best_seq[1:]
        # Truncate at EOS (if present)
        if eos_id in best_seq:
            eos_pos = best_seq.index(eos_id)
            best_seq = best_seq[:eos_pos]

        return best_seq
