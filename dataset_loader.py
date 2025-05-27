## dataset_loader.py

"""
Dataset loader for Transformer training and evaluation.
Uses SentencePiece for tokenization and implements dynamic batching
by token counts for source and target sequences.
"""

import os
import random
from typing import List, Dict, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from config import Config


class ExampleDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized source-target examples.
    Each item is a dict with keys: "src", "tgt_input", "tgt_output".
    """

    def __init__(self, examples: List[Tuple[List[int], List[int], List[int]]]) -> None:
        """
        :param examples: List of triples (src_ids, tgt_input_ids, tgt_output_ids)
        """
        super().__init__()
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        src_ids, tgt_in_ids, tgt_out_ids = self.examples[idx]
        return {
            "src": src_ids,
            "tgt_input": tgt_in_ids,
            "tgt_output": tgt_out_ids,
        }


class TokenBatchSampler(Sampler[List[int]]):
    """
    Sampler that groups examples into batches by approximate token counts.
    Ensures that sum of source tokens <= max_src_tokens and
    sum of target tokens <= max_tgt_tokens per batch.
    """

    def __init__(
        self,
        src_lengths: List[int],
        tgt_lengths: List[int],
        max_src_tokens: int,
        max_tgt_tokens: int,
        shuffle: bool = True
    ) -> None:
        """
        :param src_lengths: List of source sequence lengths.
        :param tgt_lengths: List of target sequence lengths.
        :param max_src_tokens: Maximum total source tokens per batch.
        :param max_tgt_tokens: Maximum total target tokens per batch.
        :param shuffle: Whether to shuffle examples each epoch.
        """
        if len(src_lengths) != len(tgt_lengths):
            raise ValueError("src_lengths and tgt_lengths must have the same length")
        self.src_lengths = src_lengths
        self.tgt_lengths = tgt_lengths
        self.max_src_tokens = max_src_tokens
        self.max_tgt_tokens = max_tgt_tokens
        self.shuffle = shuffle
        self.num_examples = len(src_lengths)

    def __iter__(self):
        indices = list(range(self.num_examples))
        if self.shuffle:
            random.shuffle(indices)

        batch: List[int] = []
        accum_src = 0
        accum_tgt = 0
        for idx in indices:
            l_src = self.src_lengths[idx]
            l_tgt = self.tgt_lengths[idx]
            # If adding this example exceeds either limit, yield current batch
            if batch and (
                accum_src + l_src > self.max_src_tokens
                or accum_tgt + l_tgt > self.max_tgt_tokens
            ):
                yield batch
                batch = []
                accum_src = 0
                accum_tgt = 0
            # Add to batch
            batch.append(idx)
            accum_src += l_src
            accum_tgt += l_tgt

        if batch:
            yield batch

    def __len__(self) -> int:
        # Approximate number of batches
        # at least total_tokens / max_tokens
        total_src = sum(self.src_lengths)
        avg_src = total_src / max(1, self.num_examples)
        est_batches = int((total_src / self.max_src_tokens) + 0.5)
        return max(1, est_batches)


def collate_fn(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate function to pad source and target sequences in a batch.

    :param batch: List of dicts with "src", "tgt_input", "tgt_output" as lists of ints.
    :param pad_id: Padding token ID.
    :return: Dict with padded tensors: "src", "tgt_input", "tgt_output".
    """
    batch_size = len(batch)
    # find max lengths
    max_src_len = max(len(ex["src"]) for ex in batch)
    max_tgt_len = max(len(ex["tgt_input"]) for ex in batch)

    # prepare tensors
    src_tensor = torch.full(
        (batch_size, max_src_len), pad_id, dtype=torch.long
    )
    tgt_in_tensor = torch.full(
        (batch_size, max_tgt_len), pad_id, dtype=torch.long
    )
    tgt_out_tensor = torch.full(
        (batch_size, max_tgt_len), pad_id, dtype=torch.long
    )

    # fill in
    for i, ex in enumerate(batch):
        src_ids = ex["src"]
        tgt_in_ids = ex["tgt_input"]
        tgt_out_ids = ex["tgt_output"]

        src_tensor[i, : len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        tgt_in_tensor[i, : len(tgt_in_ids)] = torch.tensor(tgt_in_ids, dtype=torch.long)
        tgt_out_tensor[i, : len(tgt_out_ids)] = torch.tensor(tgt_out_ids, dtype=torch.long)

    return {
        "src": src_tensor,
        "tgt_input": tgt_in_tensor,
        "tgt_output": tgt_out_tensor,
    }


class DatasetLoader:
    """
    Dataset loader that builds/loads a SentencePiece model and provides
    PyTorch DataLoaders for train/dev/test splits with dynamic batching.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize DatasetLoader.

        :param config: Config object with data paths and parameters.
        """
        self.config = config
        data_cfg = config.data

        # SentencePiece model prefix and files
        self.spm_prefix: str = getattr(data_cfg, "spm_model_prefix", "spm_model")
        self.model_file: str = f"{self.spm_prefix}.model"
        self.vocab_size: int = data_cfg.spm_vocab_size

        # Build or load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.build_vocab()

        # Special token IDs
        self.pad_id: int = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.bos_id: int = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_id: int = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2
        self.unk_id: int = self.sp.unk_id() if self.sp.unk_id() >= 0 else 3

    def build_vocab(self) -> None:
        """
        Train a new SentencePiece model if not found, else load existing model.
        """
        if os.path.isfile(self.model_file):
            # Load existing model
            self.sp.load(self.model_file)
            return

        # File paths for training SentencePiece
        src_train = getattr(self.config.data, "train_src", None)
        tgt_train = getattr(self.config.data, "train_tgt", None)
        if not src_train or not tgt_train:
            raise ValueError("train_src and train_tgt must be specified in config.data")

        # Temporary input list for SP training
        combined_input = ",".join([src_train, tgt_train])

        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=combined_input,
            model_prefix=self.spm_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            user_defined_symbols=[]
        )

        # Load the newly trained model
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"SentencePiece model not found after training: {self.model_file}")
        self.sp.load(self.model_file)

    def load_data(self, split: str) -> DataLoader:
        """
        Create a DataLoader for the given split ("train", "dev", or "test").

        :param split: Name of the split.
        :return: PyTorch DataLoader instance.
        """
        # Determine file paths
        src_path = getattr(self.config.data, f"{split}_src", None)
        tgt_path = getattr(self.config.data, f"{split}_tgt", None)
        if not src_path or not tgt_path:
            raise ValueError(f"Config.data must specify {split}_src and {split}_tgt")

        # Read raw lines
        with open(src_path, "r", encoding="utf-8") as f:
            src_lines = [line.strip() for line in f if line.strip()]
        with open(tgt_path, "r", encoding="utf-8") as f:
            tgt_lines = [line.strip() for line in f if line.strip()]

        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"Mismatched line counts: {split}_src has {len(src_lines)}, "
                             f"{split}_tgt has {len(tgt_lines)}")

        # Encode examples
        examples: List[Tuple[List[int], List[int], List[int]]] = []
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_ids = self.sp.encode(src_line, out_type=int)
            tgt_ids = self.sp.encode(tgt_line, out_type=int)
            # Add EOS to source, BOS/EOS to target input/output
            src_seq = src_ids + [self.eos_id]
            tgt_in = [self.bos_id] + tgt_ids
            tgt_out = tgt_ids + [self.eos_id]
            examples.append((src_seq, tgt_in, tgt_out))

        # Compute lengths for batching
        src_lengths = [len(x[0]) for x in examples]
        tgt_lengths = [len(x[1]) for x in examples]

        # Sampler for dynamic batching
        sampler = TokenBatchSampler(
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            max_src_tokens=self.config.training.batch_tokens_src,
            max_tgt_tokens=self.config.training.batch_tokens_tgt,
            shuffle=(split == "train")
        )

        # Collate function binds pad_id
        def _collate(batch):
            return collate_fn(batch, pad_id=self.pad_id)

        # DataLoader
        loader = DataLoader(
            dataset=ExampleDataset(examples),
            batch_sampler=sampler,
            collate_fn=_collate,
            num_workers=getattr(self.config.data, "num_workers", 0),
            pin_memory=getattr(self.config.data, "pin_memory", False),
        )
        return loader
