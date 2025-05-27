## evaluator.py

import torch
from torch.utils.data import DataLoader
from typing import List, Dict

from config import Config
from tokenizer import Tokenizer
from model import TransformerModel
from utils import compute_bleu


class Evaluator:
    """
    Evaluator for the Transformer model. Runs beam‐search inference on a
    DataLoader, decodes hypotheses and references, and computes corpus‐level metrics.
    """

    def __init__(
        self,
        model: TransformerModel,
        tokenizer: Tokenizer,
        config: Config
    ) -> None:
        """
        Initializes the Evaluator.

        Args:
            model:     A trained TransformerModel (in eval mode or will be set to eval()).
            tokenizer: Tokenizer used for encoding/decoding token sequences.
            config:    Config object with evaluation and training parameters.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Put model in evaluation mode
        self.model.eval()

        # Inference parameters from training config
        train_params = self.config.get_training_params()
        self.beam_size: int = train_params.get("beam_size", 1)
        self.length_penalty: float = train_params.get("length_penalty", 0.0)

        # Evaluation-specific parameters from config.yaml
        eval_cfg = self.config._cfg.get("evaluation", {})
        # Maximum additional tokens beyond input length
        self.max_length_offset: int = int(eval_cfg.get("max_length_offset", 50))
        # Metrics to compute, e.g., ["bleu"]
        self.metrics: List[str] = eval_cfg.get("metrics", ["bleu"])

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Runs inference and computes metrics on the provided DataLoader.

        Args:
            dataloader: DataLoader yielding batches containing:
                        - "src":       Tensor of source token IDs, shape (B, S)
                        - "tgt_output":Tensor of target token IDs, shape (B, T)

        Returns:
            A dict mapping metric names (e.g., "BLEU") to their computed float values.
        """
        device = next(self.model.parameters()).device

        references: List[str] = []
        hypotheses: List[str] = []

        # Disable gradient computation
        with torch.no_grad():
            for batch in dataloader:
                # Move source to device
                src = batch["src"].to(device)  # (B, S)
                batch_size, src_len = src.size()

                # Determine max generation length per example
                max_len = src_len + self.max_length_offset

                # Generate hypotheses with beam search
                generated = self.model.generate(
                    src=src,
                    max_len=max_len,
                    beam_size=self.beam_size
                )  # List[List[int]] length batch_size

                # Decode each hypothesis and reference
                for i in range(batch_size):
                    # Decode hypothesis (skip special tokens by default)
                    hyp_ids = generated[i]
                    hyp_text = self.tokenizer.decode(hyp_ids)
                    hypotheses.append(hyp_text)

                    # Decode reference from tgt_output (shifted) or full target
                    if "tgt_output" in batch:
                        ref_ids = batch["tgt_output"][i].tolist()
                    elif "tgt_input" in batch:
                        # Fallback to tgt_input if tgt_output missing
                        ref_ids = batch["tgt_input"][i].tolist()
                    else:
                        raise KeyError(
                            "Batch must contain 'tgt_output' or 'tgt_input' for references."
                        )
                    ref_text = self.tokenizer.decode(ref_ids)
                    references.append(ref_text)

        # Compute requested metrics
        results: Dict[str, float] = {}
        # BLEU (case-insensitive matching to config)
        if any(m.lower() == "bleu" for m in self.metrics):
            bleu_score = compute_bleu(references, hypotheses)
            results["BLEU"] = bleu_score

        return results
