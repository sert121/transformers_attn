## utils.py

"""
Utility functions for mask creation and BLEU evaluation in the Transformer pipeline.
"""

from typing import List, Tuple
import torch
import sacrebleu


class Utils:
    """
    Static utility methods for:
      - Generating attention masks (padding + look-ahead)
      - Computing BLEU scores via sacrebleu
    """

    @staticmethod
    def make_masks(
        src: torch.Tensor,
        tgt: torch.Tensor,
        config
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create encoder and decoder masks for attention.

        Args:
            src (torch.Tensor): Source token IDs, shape (batch_size, src_len).
            tgt (torch.Tensor): Target token IDs, shape (batch_size, tgt_len).
            config: Config object with at least `config.data.pad_token_id` attribute
                    indicating the padding token ID (default 0 if missing).

        Returns:
            src_mask (torch.Tensor): Boolean mask for encoder inputs,
                shape (batch_size, 1, 1, src_len), where True indicates
                a non-padding token.
            tgt_mask (torch.Tensor): Boolean mask for decoder self-attention,
                shape (batch_size, 1, tgt_len, tgt_len), where True indicates
                allowed attention (no padding, no future).
        """
        # Determine pad token ID (default to 0 if not specified)
        pad_token_id = getattr(config.data, 'pad_token_id', 0)

        # Encoder padding mask: (batch, src_len) -> (batch, 1, 1, src_len)
        # True for tokens that are NOT pad
        src_pad_mask = (src != pad_token_id)
        src_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)

        # Decoder padding mask: (batch, tgt_len) -> (batch, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != pad_token_id)
        tgt_pad_mask_exp = tgt_pad_mask.unsqueeze(1).unsqueeze(2)

        # Subsequent (look-ahead) mask: prevent attending to future tokens
        # Create upper-triangular matrix with 1s above diagonal
        tgt_len = tgt.size(1)
        subsequent = torch.triu(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device),
            diagonal=1
        )
        # Expand to (1, 1, tgt_len, tgt_len) for broadcasting
        subsequent_mask = subsequent.unsqueeze(0).unsqueeze(0)

        # Combine padding and subsequent masks:
        # Allowed positions are ones that are not padding and not future
        tgt_mask = tgt_pad_mask_exp & (~subsequent_mask)

        return src_mask, tgt_mask

    @staticmethod
    def compute_bleu(preds: List[str], refs: List[str]) -> float:
        """
        Compute corpus-level BLEU score using sacrebleu.

        Args:
            preds (List[str]): List of hypothesis sentences (detokenized).
            refs (List[str]): List of reference sentences (detokenized).

        Returns:
            float: BLEU score (0.0 - 100.0).
        """
        if len(preds) != len(refs):
            raise ValueError(
                f"Number of hypotheses ({len(preds)}) "
                f"and references ({len(refs)}) must be the same."
            )
        # sacrebleu expects list of references (possibly multiple), so wrap refs
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        return float(bleu.score)
