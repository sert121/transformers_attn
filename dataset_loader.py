# dataset_loader.py

"""
DatasetLoader: Tokenizer training and dataset preparation for sequence-to-sequence tasks.
Follows the design and hyperparameters specified in config.yaml.
"""

import os
import logging
from typing import Tuple, List

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm


class DatasetLoader:
    """
    Handles subword tokenizer training/loading and creation of PyTorch Datasets
    for training, development, and test splits.

    Public methods:
        load_tokenizer() -> Tokenizer
        load_datasets() -> Tuple[Dataset, Dataset, Dataset]
    """

    def __init__(self, config: object):
        """
        Args:
            config: Config object with a .get(section: str) -> dict interface.
        """
        self.config = config
        # Data file paths and settings
        data_cfg = config.get("data")
        self.train_src_path: str = data_cfg["train_src"]
        self.train_tgt_path: str = data_cfg["train_tgt"]
        self.dev_src_path: str = data_cfg["dev_src"]
        self.dev_tgt_path: str = data_cfg["dev_tgt"]
        self.test_src_path: str = data_cfg["test_src"]
        self.test_tgt_path: str = data_cfg["test_tgt"]
        self.max_seq_length: int = int(data_cfg.get("max_seq_length", 100))

        # Tokenizer settings
        self.tokenizer_cfg = config.get("tokenizer")

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_tokenizer(self) -> Tokenizer:
        """
        Train (or load) a subword tokenizer according to the configuration.
        Supports BPE and WordPiece tokenization.

        Returns:
            Tokenizer: a trained Tokenizer instance with truncation and padding enabled.
        """
        tok_type: str = self.tokenizer_cfg.get("type", "byte-pair")
        vocab_size: int = int(self.tokenizer_cfg.get("vocab_size", 37000))
        special_tokens: List[str] = list(self.tokenizer_cfg.get("special_tokens", []))

        # Initialize tokenizer model
        if tok_type == "byte-pair":
            model = BPE(unk_token="<unk>")
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=True
            )
        elif tok_type == "word-piece":
            model = WordPiece(unk_token="<unk>")
            trainer = WordPieceTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=True
            )
        else:
            raise ValueError(f"Unsupported tokenizer type: '{tok_type}'")

        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = Whitespace()

        # Train tokenizer on concatenated source+target corpora
        files = [self.train_src_path, self.train_tgt_path]
        self.logger.info(
            f"Training {tok_type} tokenizer on {files} with vocab_size={vocab_size}"
        )
        tokenizer.train(files, trainer)

        # Enable truncation and padding
        tokenizer.enable_truncation(max_length=self.max_seq_length)
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("Tokenizer does not contain the '<pad>' token.")
        tokenizer.enable_padding(
            pad_id=pad_id,
            pad_token="<pad>"
        )

        self.logger.info(
            f"Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}, "
            f"Max sequence length: {self.max_seq_length}"
        )
        # Cache for reuse
        self.tokenizer = tokenizer
        return tokenizer

    def load_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create PyTorch Datasets for train, dev, and test splits.
        Each dataset returns dicts with 'input_ids' and 'labels' (both torch.LongTensor).

        Returns:
            (train_dataset, dev_dataset, test_dataset)
        """
        # Ensure tokenizer is loaded
        if not hasattr(self, "tokenizer"):
            self.load_tokenizer()

        tokenizer: Tokenizer = self.tokenizer

        class TranslationDataset(Dataset):
            """
            Simple Dataset wrapping parallel token ID sequences.
            """
            def __init__(self, src_ids: List[List[int]], tgt_ids: List[List[int]]):
                assert len(src_ids) == len(tgt_ids), "Source/target length mismatch"
                self.src_ids = src_ids
                self.tgt_ids = tgt_ids

            def __len__(self) -> int:
                return len(self.src_ids)

            def __getitem__(self, idx: int) -> dict:
                return {
                    "input_ids": torch.tensor(self.src_ids[idx], dtype=torch.long),
                    "labels": torch.tensor(self.tgt_ids[idx], dtype=torch.long)
                }

        def _make_split(src_path: str, tgt_path: str) -> TranslationDataset:
            """
            Reads and encodes a parallel corpus split.

            Args:
                src_path: Path to source language file.
                tgt_path: Path to target language file.

            Returns:
                TranslationDataset for the split.
            """
            self.logger.info(f"Loading and encoding split: src={src_path}, tgt={tgt_path}")
            src_ids: List[List[int]] = []
            tgt_ids: List[List[int]] = []

            # Read line-by-line
            with open(src_path, "r", encoding="utf-8") as src_f, \
                 open(tgt_path, "r", encoding="utf-8") as tgt_f:
                for src_line, tgt_line in tqdm(
                    zip(src_f, tgt_f),
                    desc=f"Encoding {os.path.basename(src_path)}",
                    unit="lines"
                ):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()
                    if not src_line or not tgt_line:
                        continue  # skip empty lines

                    # Encode with tokenizer (adds special tokens, truncates)
                    enc_src = tokenizer.encode(src_line)
                    enc_tgt = tokenizer.encode(tgt_line)

                    # Filter out pairs that still exceed max length (defensive)
                    if len(enc_src.ids) > self.max_seq_length or \
                       len(enc_tgt.ids) > self.max_seq_length:
                        continue

                    src_ids.append(enc_src.ids)
                    tgt_ids.append(enc_tgt.ids)

            self.logger.info(
                f"Finished encoding {os.path.basename(src_path)}: "
                f"{len(src_ids)} examples"
            )
            return TranslationDataset(src_ids, tgt_ids)

        # Build datasets
        train_ds = _make_split(self.train_src_path, self.train_tgt_path)
        dev_ds = _make_split(self.dev_src_path, self.dev_tgt_path)
        test_ds = _make_split(self.test_src_path, self.test_tgt_path)

        return train_ds, dev_ds, test_ds
