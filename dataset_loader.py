## dataset_loader.py

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import sentencepiece as spm

from config import Config


class ParallelTextDataset(Dataset):
    """
    Dataset for parallel text (source-target) pairs using a SentencePiece model.

    Each example returns:
        {
            "src_ids": List[int],
            "tgt_input_ids": List[int],
            "tgt_output_ids": List[int]
        }
    """

    def __init__(
        self, src_path: str, tgt_path: str, sp_model: spm.SentencePieceProcessor
    ) -> None:
        """
        Args:
            src_path: Path to the source language text file (one sentence per line).
            tgt_path: Path to the target language text file (one sentence per line).
            sp_model: A loaded SentencePieceProcessor with special tokens configured.
        """
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")
        if not os.path.isfile(tgt_path):
            raise FileNotFoundError(f"Target file not found: {tgt_path}")

        # Load all lines into memory
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f]
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f]

        if len(self.src_lines) != len(self.tgt_lines):
            raise ValueError(
                f"Source and target files have different number of lines: "
                f"{len(self.src_lines)} vs {len(self.tgt_lines)}"
            )

        self.sp = sp_model
        # Special token IDs
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        # Precompute example lengths (for batching): sum of token counts
        self.lengths = []
        for src, tgt in zip(self.src_lines, self.tgt_lines):
            src_ids = self.sp.encode(src, out_type=int)
            tgt_ids = self.sp.encode(tgt, out_type=int)
            # tgt_input will have +1 for BOS, tgt_output +1 for EOS
            length = len(src_ids) + max(len(tgt_ids) + 1, 1)
            self.lengths.append(length)

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        src_line = self.src_lines[idx]
        tgt_line = self.tgt_lines[idx]

        # Tokenize into pieces
        src_ids = self.sp.encode(src_line, out_type=int)
        tgt_ids = self.sp.encode(tgt_line, out_type=int)

        # Build autoregressive inputs and outputs
        tgt_input_ids = [self.bos_id] + tgt_ids
        tgt_output_ids = tgt_ids + [self.eos_id]

        return {
            "src_ids": src_ids,
            "tgt_input_ids": tgt_input_ids,
            "tgt_output_ids": tgt_output_ids,
        }


class TokenBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups indices into batches
    whose total token count (sum of example lengths) does not exceed a threshold.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens: int,
        shuffle: bool = True,
    ) -> None:
        """
        Args:
            lengths: Sequence of example lengths (number of tokens).
            max_tokens: Maximum sum of lengths per batch.
            shuffle: Whether to shuffle indices at each epoch.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.indices = list(range(len(self.lengths)))

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.indices)).tolist()
            indices = [self.indices[i] for i in perm]
        else:
            indices = list(self.indices)

        batch: List[int] = []
        tokens_in_batch = 0
        for idx in indices:
            length = self.lengths[idx]
            # If adding this example would exceed max_tokens, yield current batch
            if batch and tokens_in_batch + length > self.max_tokens:
                yield batch
                batch = []
                tokens_in_batch = 0
            batch.append(idx)
            tokens_in_batch += length

        if batch:
            yield batch

    def __len__(self) -> int:
        # Approximate number of batches
        total = sum(self.lengths)
        return max(1, (total + self.max_tokens - 1) // self.max_tokens)


class DatasetLoader:
    """
    Responsible for training/loading the SentencePiece model and
    creating PyTorch DataLoaders for train/dev/test splits.
    """

    def __init__(self, config: Config) -> None:
        """
        Args:
            config: A Config instance with dataset and tokenization settings.
        """
        self.config = config
        data_cfg = config["data"]

        # File paths
        self.src_train = data_cfg["src_train"]
        self.tgt_train = data_cfg["tgt_train"]
        self.src_dev = data_cfg["src_dev"]
        self.tgt_dev = data_cfg["tgt_dev"]
        self.src_test = data_cfg["src_test"]
        self.tgt_test = data_cfg["tgt_test"]

        # SentencePiece parameters
        self.vocab_size: int = int(data_cfg["vocab_size"])
        self.tokenization: str = data_cfg["tokenization"]  # "bpe" or "sentencepiece"

        # SentencePiece model prefix and paths
        model_prefix = f"spm_{self.vocab_size}"
        self.sp_model_path = model_prefix + ".model"
        self.sp_vocab_path = model_prefix + ".vocab"

        # Placeholder for loaded SentencePieceProcessor
        self.sp_model: Optional[spm.SentencePieceProcessor] = None

        # Batching parameters
        self.batch_max_tokens: int = int(config["training.batch_max_tokens"])
        # Number of workers for DataLoader
        self.num_workers: int = int(config.get("data.num_workers", 4))

    def prepare_vocab(self) -> None:
        """
        Train or load a SentencePiece model according to the configuration.
        """
        # If already loaded, nothing to do
        if self.sp_model is not None:
            return

        # Check if model file exists
        if os.path.isfile(self.sp_model_path):
            sp = spm.SentencePieceProcessor()
            sp.Load(self.sp_model_path)
            self.sp_model = sp
            return

        # Train a new SentencePiece model
        input_files = f"{self.src_train},{self.tgt_train}"
        model_type = (
            "bpe" if self.tokenization == "bpe" else "unigram"
        )
        spm.SentencePieceTrainer.Train(
            input=input_files,
            model_prefix=os.path.splitext(self.sp_model_path)[0],
            vocab_size=self.vocab_size,
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols="",
            character_coverage=1.0,
            hard_vocab_limit=False,
        )
        # Load the newly trained model
        sp = spm.SentencePieceProcessor()
        sp.Load(self.sp_model_path)
        self.sp_model = sp

    def _collate_fn(
        self, batch: List[Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function to build padded batches from a list of samples.

        Returns a dict with:
            src: Tensor[int64] of shape (B, max_src_len)
            tgt_input: Tensor[int64] of shape (B, max_tgt_len)
            tgt_output: Tensor[int64] of shape (B, max_tgt_len)
        """
        assert self.sp_model is not None, "SentencePiece model not loaded."

        pad_id = self.sp_model.pad_id()

        # Determine max sequence lengths
        max_src_len = max(len(ex["src_ids"]) for ex in batch)
        max_tgt_len = max(len(ex["tgt_input_ids"]) for ex in batch)

        batch_size = len(batch)
        src_batch = torch.full(
            (batch_size, max_src_len), pad_id, dtype=torch.long
        )
        tgt_input_batch = torch.full(
            (batch_size, max_tgt_len), pad_id, dtype=torch.long
        )
        tgt_output_batch = torch.full(
            (batch_size, max_tgt_len), pad_id, dtype=torch.long
        )

        # Populate tensors
        for i, ex in enumerate(batch):
            src_ids = ex["src_ids"]
            tgt_in = ex["tgt_input_ids"]
            tgt_out = ex["tgt_output_ids"]
            src_batch[i, : len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
            tgt_input_batch[i, : len(tgt_in)] = torch.tensor(tgt_in, dtype=torch.long)
            tgt_output_batch[i, : len(tgt_out)] = torch.tensor(
                tgt_out, dtype=torch.long
            )

        return {
            "src": src_batch,
            "tgt_input": tgt_input_batch,
            "tgt_output": tgt_output_batch,
        }

    def load_datasets(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare the vocabulary and return DataLoaders for train, dev, and test.

        Returns:
            (train_loader, dev_loader, test_loader)
        """
        # Ensure subword model is ready
        self.prepare_vocab()
        assert self.sp_model is not None, "Failed to load or train SentencePiece model."

        # Instantiate datasets
        train_ds = ParallelTextDataset(
            self.src_train, self.tgt_train, self.sp_model
        )
        dev_ds = ParallelTextDataset(self.src_dev, self.tgt_dev, self.sp_model)
        test_ds = ParallelTextDataset(
            self.src_test, self.tgt_test, self.sp_model
        )

        # Create samplers
        train_sampler = TokenBatchSampler(
            lengths=train_ds.lengths,
            max_tokens=self.batch_max_tokens,
            shuffle=True,
        )
        dev_sampler = TokenBatchSampler(
            lengths=dev_ds.lengths,
            max_tokens=self.batch_max_tokens,
            shuffle=False,
        )
        test_sampler = TokenBatchSampler(
            lengths=test_ds.lengths,
            max_tokens=self.batch_max_tokens,
            shuffle=False,
        )

        # Build DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_sampler=dev_sampler,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_sampler=test_sampler,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, dev_loader, test_loader
