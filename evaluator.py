# evaluator.py

import os
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu
from tokenizers import ByteLevelBPETokenizer

from config import Config
from model import TransformerModel
import utils


class Evaluator:
    """Evaluates a trained Transformer model on a test dataset using BLEU."""

    def __init__(
        self,
        config: Config,
        model: TransformerModel,
        dataloader: DataLoader,
    ) -> None:
        """
        Initializes the evaluator.

        Args:
            config: Configuration object.
            model:   A TransformerModel instance.
            dataloader: DataLoader for the test split.
        """
        self.config = config
        self.model = model
        self.dataloader = dataloader

        # Device
        self.device = utils.get_device(self.config)
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer from disk (same as used in DatasetLoader)
        pair = self.config.get("data.language_pair", "")
        if not pair:
            raise ValueError(
                "Language pair not specified in config under 'data.language_pair'."
            )
        tokenizer_dir = os.path.join("tokenizer", pair)
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        merges_path = os.path.join(tokenizer_dir, "merges.txt")
        if not os.path.isfile(vocab_path) or not os.path.isfile(merges_path):
            raise FileNotFoundError(
                f"Tokenizer files not found in '{tokenizer_dir}'. "
                "Ensure you have trained or saved the tokenizer."
            )
        # Instantiate ByteLevelBPETokenizer for decoding
        self.tokenizer = ByteLevelBPETokenizer(
            vocab_path, merges_path, lowercase=False
        )

        # Inference hyperparameters
        self.beam_size: int = self.config.get("inference.beam_size", 4)
        self.length_penalty: float = self.config.get(
            "inference.length_penalty_alpha", 0.0
        )
        self.max_len_offset: int = self.config.get(
            "inference.max_length_offset", 50
        )

    def evaluate(self, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        """
        Runs beam-search decoding on the test set and computes BLEU.

        Args:
            checkpoint_path: Optional path to a model checkpoint to load before evaluation.

        Returns:
            A dict mapping metric names to values, e.g., {'BLEU': 28.4}.
        """
        # Optionally load checkpoint
        if checkpoint_path:
            # Load model weights into the model
            utils.load_checkpoint(
                path=checkpoint_path,
                model=self.model,
                optimizer=None,
                scheduler=None,
                device=self.device,
            )
            self.model.to(self.device)
            self.model.eval()

        all_hypotheses: List[str] = []
        all_references: List[str] = []

        # Iterate over test DataLoader
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            # Source tokens
            src: torch.Tensor = batch["src"].to(self.device)  # (B, src_len)

            # Determine maximum target length for generation
            batch_size, src_len = src.size()
            max_len = src_len + self.max_len_offset

            # Beam search generation
            with torch.no_grad():
                gen_ids_batch: List[List[int]] = self.model.generate(
                    src=src, beam_size=self.beam_size, max_len=max_len
                )

            # Decode predictions and references
            tgt: torch.Tensor = batch["tgt"]  # (B, tgt_len_padded)
            for i in range(batch_size):
                # Hypothesis
                hyp_ids: List[int] = gen_ids_batch[i]
                hyp_text: str = self.tokenizer.decode(
                    hyp_ids, skip_special_tokens=True
                ).strip()
                all_hypotheses.append(hyp_text)

                # Reference (full padded sequence; tokenizer will skip pads)
                ref_ids: List[int] = tgt[i].tolist()
                ref_text: str = self.tokenizer.decode(
                    ref_ids, skip_special_tokens=True
                ).strip()
                all_references.append(ref_text)

        # Compute corpus-level BLEU using sacrebleu
        # sacrebleu expects list of hypotheses and list of reference streams
        bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])

        return {"BLEU": round(bleu.score, 2)}
