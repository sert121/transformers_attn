## evaluation.py

import os
import glob
import re
from typing import Any, Dict, List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import sacrebleu
from sentencepiece import SentencePieceProcessor

from config import Config
from model import TransformerModel  # Assuming model.py is in the same package


class BeamEntry(NamedTuple):
    """Represents a hypothesis in beam search."""
    seq: List[int]
    logprob: float


class Evaluation:
    """
    Evaluation engine for Transformer models. Performs checkpoint averaging,
    beam-search decoding, and computes corpus BLEU and token-level loss.
    """

    def __init__(
        self,
        model: TransformerModel,
        config: Config,
        test_loader: DataLoader,
    ) -> None:
        """
        Initialize evaluation with a trained model, configuration, and test DataLoader.

        Args:
            model: TransformerModel instance.
            config: Config object with inference and logging settings.
            test_loader: DataLoader yielding batches for evaluation.
        """
        self.model = model
        self.config = config
        self.test_loader = test_loader
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Load and average checkpoints
        self._average_checkpoints()
        # Prepare tokenizer
        self._load_tokenizer()
        # Loss function: sum over non-pad tokens
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.pad_id, reduction="sum"
        )
        # Inference parameters
        inf = config["inference"]
        self.beam_size: int = int(inf["beam_size"])
        self.length_penalty: float = float(inf["length_penalty"])
        self.max_len_offset: int = int(inf["max_len_offset"])

    def _average_checkpoints(self) -> None:
        """
        Load the last N checkpoints and average their model parameters.
        """
        # Determine checkpoint directory
        # Default to 'checkpoints' if not specified
        ckpt_dir = self.config.get("logging.checkpoint_dir", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        # Find all checkpoint files (*.pt or *.pth)
        patterns = ["*.pt", "*.pth"]
        ckpt_paths: List[str] = []
        for pat in patterns:
            ckpt_paths.extend(glob.glob(os.path.join(ckpt_dir, pat)))
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
        # Sort by modification time
        ckpt_paths.sort(key=os.path.getmtime)
        # Select last N
        n_avg = int(self.config["inference.avg_checkpoints"])
        selected = ckpt_paths[-n_avg:]
        # Load and sum parameters
        avg_state: Dict[str, Tensor] = {}
        for idx, path in enumerate(selected):
            chk = torch.load(path, map_location="cpu")
            # Extract model state dict
            state = (
                chk["model_state_dict"]
                if isinstance(chk, dict) and "model_state_dict" in chk
                else chk
            )
            # On first checkpoint, initialize
            if idx == 0:
                for k, v in state.items():
                    avg_state[k] = v.clone().float()
            else:
                for k, v in state.items():
                    avg_state[k] += v.float()
        # Average
        for k in avg_state:
            avg_state[k] /= float(len(selected))
        # Load into model
        self.model.load_state_dict(avg_state)
        self.model.to(self.device)

    def _load_tokenizer(self) -> None:
        """
        Load the SentencePiece model used for tokenization.
        """
        data_cfg = self.config["data"]
        vocab_size = int(data_cfg["vocab_size"])
        # Assume SentencePiece model file is named spm_{vocab_size}.model
        spm_path = f"spm_{vocab_size}.model"
        if not os.path.isfile(spm_path):
            raise FileNotFoundError(f"SentencePiece model not found: {spm_path}")
        sp = SentencePieceProcessor()
        sp.Load(spm_path)
        self.tokenizer = sp
        # Special token IDs
        self.pad_id = sp.pad_id()
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()

    def evaluate(self) -> Dict[str, float]:
        """
        Perform evaluation on the test set: compute average token loss and BLEU.

        Returns:
            A dict with keys "bleu" (float BLEU score) and "loss" (avg loss per token).
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        hypotheses: List[str] = []
        references: List[List[str]] = []
        with torch.no_grad():
            for batch in self.test_loader:
                # Move source tokens to device
                src: Tensor = batch["src"].to(self.device)
                # Prepare references by decoding tgt_output
                tgt_out: Tensor = batch["tgt_output"]
                for row in tgt_out.tolist():
                    # trim at first EOS
                    if self.eos_id in row:
                        cut = row.index(self.eos_id)
                        ids = row[:cut]
                    else:
                        ids = [tok for tok in row if tok != self.pad_id]
                    ref_str = self._decode(ids)
                    references.append([ref_str])
                # Compute loss if target inputs are available
                if "tgt_input" in batch:
                    tgt_in: Tensor = batch["tgt_input"].to(self.device)
                    tgt_out_dev: Tensor = tgt_out.to(self.device)
                    logits: Tensor = self.model(src, tgt_in)
                    B, T, V = logits.size()
                    loss = self.loss_fn(
                        logits.view(B * T, V), tgt_out_dev.view(B * T)
                    )
                    total_loss += loss.item()
                    nonpad = (tgt_out_dev != self.pad_id).sum().item()
                    total_tokens += nonpad
                # Decode each example with beam search
                for i in range(src.size(0)):
                    single_src = src[i : i + 1, :]
                    pred_ids = self._beam_search(single_src)
                    hypotheses.append(self._decode(pred_ids))
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        # sacrebleu expects list of hyps and list of lists of refs (transposed)
        bleu = sacrebleu.corpus_bleu(
            hypotheses, list(zip(*references)), lowercase=False
        )
        return {"bleu": float(bleu.score), "loss": avg_loss}

    def _beam_search(self, src: Tensor) -> List[int]:
        """
        Beam-search decode a single source sequence.

        Args:
            src: Tensor of shape (1, src_len)

        Returns:
            A list of token IDs (excluding initial BOS and final EOS).
        """
        # Encode source once
        memory = self.model.encode(src, src_mask=None)
        # Initialize beam
        beams: List[BeamEntry] = [BeamEntry(seq=[self.bos_id], logprob=0.0)]
        completed: List[BeamEntry] = []
        max_len = src.size(1) + self.max_len_offset

        for _ in range(max_len):
            all_candidates: List[BeamEntry] = []
            for entry in beams:
                seq, logp = entry.seq, entry.logprob
                # If already ended, carry over
                if seq[-1] == self.eos_id:
                    completed.append(entry)
                    continue
                # Prepare decoder input
                tgt = torch.tensor([seq], dtype=torch.long, device=self.device)
                # Decode step
                dec_out = self.model.decode(tgt, memory, src_mask=None, tgt_mask=None)
                # Get last token logits and log-probs
                logits = dec_out[:, -1, :]  # (1, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                # Top-K expansions
                topk_logp, topk_ids = log_probs.topk(self.beam_size)
                for k in range(self.beam_size):
                    new_seq = seq + [int(topk_ids[k].item())]
                    new_logp = logp + float(topk_logp[k].item())
                    all_candidates.append(BeamEntry(seq=new_seq, logprob=new_logp))
            # If no unfinished candidates, break
            if not all_candidates:
                break
            # Apply length penalty and select top beams
            def score_norm(be: BeamEntry) -> float:
                length = len(be.seq)
                lp = ((5.0 + length) ** self.length_penalty) / ((5.0 + 1.0) ** self.length_penalty)
                return be.logprob / lp

            # Keep top beam_size candidates
            beams = sorted(all_candidates, key=score_norm, reverse=True)[: self.beam_size]

        # If we have completed hypotheses, choose best among them
        final_beams = completed if completed else beams
        best = max(final_beams, key=lambda be: score_norm(be))
        # Strip BOS and EOS
        seq = best.seq
        # Remove initial BOS
        if seq and seq[0] == self.bos_id:
            seq = seq[1:]
        # Truncate at EOS if present
        if self.eos_id in seq:
            eos_pos = seq.index(self.eos_id)
            seq = seq[:eos_pos]
        return seq

    def _decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs to a string via the tokenizer.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.
        """
        # Try common SentencePiece decode methods
        if hasattr(self.tokenizer, "decode_ids"):
            return self.tokenizer.decode_ids(ids)
        if hasattr(self.tokenizer, "DecodeIds"):
            return self.tokenizer.DecodeIds(ids)
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(ids)
        # Fallback: join IDs (not ideal)
        return " ".join(str(i) for i in ids)
