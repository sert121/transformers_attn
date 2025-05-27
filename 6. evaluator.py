## 6. evaluator.py

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from utils import build_tokenizer, compute_bleu, compute_f1


class Evaluator:
    """Evaluator for machine translation and constituency parsing tasks
    using beam search and standard metrics (BLEU, F1)."""

    def __init__(
        self, model: torch.nn.Module, dataloader: DataLoader, cfg: Config
    ) -> None:
        """
        Args:
            model: Trained TransformerModel with weights loaded.
            dataloader: DataLoader for the evaluation split.
            cfg: Configuration object.
        """
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer for encoding/decoding
        self.tokenizer = build_tokenizer(cfg)

        # Machine translation inference params
        self.mt_beam_size: int = cfg.get("inference.beam_size")
        self.mt_length_penalty: float = cfg.get("inference.length_penalty")
        self.mt_max_length_offset: int = cfg.get("inference.max_length_offset")

        # Parsing inference params
        self.p_beam_size: int = cfg.get("parsing.beam_size")
        self.p_length_penalty: float = cfg.get("parsing.length_penalty")
        self.p_max_length_offset: int = cfg.get("parsing.max_length_offset")

        # Special token IDs (fall back to pad token or 0)
        self.bos_token_id: int = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else (self.tokenizer.pad_token_id or 0)
        )
        self.eos_token_id: int = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else (self.tokenizer.pad_token_id or 0)
        )

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def evaluate_mt(self) -> Dict[str, float]:
        """Run beam search on the machine translation test set and compute BLEU."""
        all_preds: List[str] = []
        all_refs: List[str] = []

        with torch.no_grad():
            for batch in self.dataloader:
                src_ids = batch["src_ids"].to(self.device)       # (B, S)
                src_mask = batch["src_mask"].to(self.device)     # (B, S)
                tgt_ids = batch["tgt_ids"]                      # (B, T) on CPU

                batch_size = src_ids.size(0)
                for i in range(batch_size):
                    single_src = src_ids[i : i + 1]             # (1, S)
                    single_mask = src_mask[i : i + 1]           # (1, S)
                    max_len = single_src.size(1) + self.mt_max_length_offset

                    pred_ids = self._beam_search(
                        single_src,
                        single_mask,
                        beam_size=self.mt_beam_size,
                        length_penalty=self.mt_length_penalty,
                        max_len=max_len,
                    )
                    # Decode prediction and reference
                    pred_text = self.tokenizer.decode(
                        pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    ref_ids = tgt_ids[i].tolist()
                    ref_text = self.tokenizer.decode(
                        ref_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    all_preds.append(pred_text)
                    all_refs.append(ref_text)

        bleu = compute_bleu(all_preds, all_refs)
        return {"bleu": bleu}

    def evaluate_parsing(self) -> Dict[str, float]:
        """Run beam search on the constituency parsing test set and compute F1."""
        pred_trees: List[str] = []
        gold_trees: List[str] = []

        with torch.no_grad():
            for batch in self.dataloader:
                src_ids = batch["src_ids"].to(self.device)
                src_mask = batch["src_mask"].to(self.device)
                tgt_ids = batch["tgt_ids"]  # bracket sequences

                batch_size = src_ids.size(0)
                for i in range(batch_size):
                    single_src = src_ids[i : i + 1]
                    single_mask = src_mask[i : i + 1]
                    max_len = single_src.size(1) + self.p_max_length_offset

                    pred_ids = self._beam_search(
                        single_src,
                        single_mask,
                        beam_size=self.p_beam_size,
                        length_penalty=self.p_length_penalty,
                        max_len=max_len,
                    )
                    # Tokenizer decodes back to bracketed string
                    pred_tree = self.tokenizer.decode(
                        pred_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
                    )
                    gold_ids = tgt_ids[i].tolist()
                    gold_tree = self.tokenizer.decode(
                        gold_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
                    )
                    pred_trees.append(pred_tree)
                    gold_trees.append(gold_tree)

        f1 = compute_f1(pred_trees, gold_trees)
        return {"f1": f1}

    def _beam_search(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        beam_size: int,
        length_penalty: float,
        max_len: int,
    ) -> List[int]:
        """
        Perform beam search decoding.

        Args:
            src_ids: Tensor of shape (1, S) with source token IDs.
            src_mask: Tensor of shape (1, S) with bool mask (True=real tokens).
            beam_size: Beam width.
            length_penalty: Exponent for length normalization.
            max_len: Maximum number of tokens to generate.

        Returns:
            List of generated token IDs (without BOS/EOS).
        """
        # Each hypothesis is a tuple (tokens, cumulative_logprob)
        beams: List[Tuple[List[int], float]] = [([self.bos_token_id], 0.0)]
        completed: List[Tuple[List[int], float]] = []

        for _ in range(max_len):
            candidates: List[Tuple[List[int], float]] = []
            for tokens, score in beams:
                # If EOS reached, carry forward
                if tokens[-1] == self.eos_token_id:
                    completed.append((tokens, score))
                    continue
                # Prepare decoder input
                tgt_input = torch.tensor(tokens, device=self.device).unsqueeze(0)  # (1, T)
                # Forward pass
                logits = self.model(src_ids, tgt_input, src_mask=src_mask)       # (1, T, V)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)              # (V,)
                topk_logp, topk_ids = log_probs.topk(beam_size)

                # Extend each beam
                for lp, tid in zip(topk_logp.tolist(), topk_ids.tolist()):
                    new_tokens = tokens + [tid]
                    candidates.append((new_tokens, score + lp))

            # If no candidates, break
            if not candidates:
                break

            # Apply length penalty and select top beams
            scored: List[Tuple[float, List[int], float]] = []
            for tokens, cum_logp in candidates:
                length = len(tokens)
                lp_norm = cum_logp / (length**length_penalty)
                scored.append((lp_norm, tokens, cum_logp))
            # Sort and prune
            scored.sort(key=lambda x: x[0], reverse=True)
            beams = [(tokens, logp) for (_, tokens, logp) in scored[:beam_size]]

            # If all beams have ended in EOS, stop early
            if all(tokens[-1] == self.eos_token_id for tokens, _ in beams):
                completed.extend(beams)
                break

        # Choose final beam: prefer completed ones
        final_beams = completed if completed else beams
        # Pick best by normalized score
        best = max(
            final_beams,
            key=lambda x: x[1] / (len(x[0]) ** length_penalty)
        )
        best_tokens = best[0]

        # Strip BOS if present
        if best_tokens and best_tokens[0] == self.bos_token_id:
            best_tokens = best_tokens[1:]
        # Truncate at EOS
        if self.eos_token_id in best_tokens:
            idx = best_tokens.index(self.eos_token_id)
            best_tokens = best_tokens[:idx]

        return best_tokens
