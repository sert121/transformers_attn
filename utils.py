## utils.py

import torch
from typing import List, Tuple, Optional

import sacrebleu


def create_masks(
    src: torch.LongTensor,
    tgt: Optional[torch.LongTensor],
    pad_idx: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Create boolean masks for source and target sequences to be used
    in the Transformer model's attention mechanisms.

    Args:
        src: Tensor of shape (batch_size, src_len) containing source token IDs.
        tgt: Tensor of shape (batch_size, tgt_len) containing target token IDs,
             or None if only source mask is needed.
        pad_idx: Integer ID of the padding token.

    Returns:
        A tuple (src_mask, tgt_mask) where:
          - src_mask is a boolean tensor of shape (batch_size, 1, 1, src_len),
            True at non-pad positions, False at pad positions.
          - tgt_mask is a boolean tensor of shape (batch_size, 1, tgt_len, tgt_len),
            combining padding mask and look-ahead mask, or None if tgt is None.
    """
    # Source padding mask: True where src != pad_idx
    # Shape: (batch_size, src_len)
    src_pad_mask = (src != pad_idx)
    # Shape: (batch_size, 1, 1, src_len)
    src_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)

    if tgt is None:
        return src_mask, None

    # Target padding mask: True where tgt != pad_idx
    # Shape: (batch_size, tgt_len)
    tgt_pad_mask = (tgt != pad_idx)
    # Shape: (batch_size, 1, 1, tgt_len)
    tgt_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)

    # Look-ahead mask: lower-triangular matrix of shape (tgt_len, tgt_len)
    tgt_len = tgt.size(1)
    device = tgt.device
    # dtype=torch.bool for boolean masking
    look_ahead = torch.tril(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device)
    )
    # Shape: (1, 1, tgt_len, tgt_len)
    look_ahead = look_ahead.unsqueeze(0).unsqueeze(0)

    # Combine padding and look-ahead masks
    # tgt_mask[i, 0, j, k] is True if position k is non-pad and k <= j
    tgt_mask = tgt_pad_mask & look_ahead

    return src_mask, tgt_mask


def compute_bleu(
    references: List[str],
    hypotheses: List[str]
) -> float:
    """
    Compute corpus-level BLEU score using sacrebleu.

    Args:
        references: List of reference sentences (strings).
        hypotheses: List of hypothesis sentences (strings).

    Returns:
        BLEU score as a float, e.g., 28.4.
    """
    if not references or not hypotheses:
        return 0.0
    # sacrebleu expects a list of reference-lists for multi-reference support
    formatted_refs = [references]
    bleu = sacrebleu.corpus_bleu(hypotheses, formatted_refs)
    return float(bleu.score)
