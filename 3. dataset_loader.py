## 3. dataset_loader.py

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from config import Config
from utils import build_tokenizer


class TranslationDataset(Dataset):
    """Simple Dataset for machine translation (source-target pairs)."""

    def __init__(self, src_seqs: List[List[int]], tgt_seqs: List[List[int]]) -> None:
        if len(src_seqs) != len(tgt_seqs):
            raise ValueError(
                f"Source and target sequence counts differ: "
                f"{len(src_seqs)} vs {len(tgt_seqs)}"
            )
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs

    def __len__(self) -> int:
        return len(self.src_seqs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {
            "src_ids": self.src_seqs[idx],
            "tgt_ids": self.tgt_seqs[idx],
        }


class ParsingDataset(Dataset):
    """Dataset for constituency parsing: input sentences and bracket sequences."""

    def __init__(self, src_seqs: List[List[int]], tgt_seqs: List[List[int]]) -> None:
        if len(src_seqs) != len(tgt_seqs):
            raise ValueError(
                f"Parsing src/tgt counts differ: {len(src_seqs)} vs {len(tgt_seqs)}"
            )
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs

    def __len__(self) -> int:
        return len(self.src_seqs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {
            "src_ids": self.src_seqs[idx],
            "tgt_ids": self.tgt_seqs[idx],
        }


class TokenBucketBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups examples so that each batch has <= tokens_per_batch tokens.

    Tokens per example is computed as len(src_ids) + len(tgt_ids).
    Batches are formed by sorting examples by length and accumulating until limit.
    """

    def __init__(self, dataset: Dataset, tokens_per_batch: int) -> None:
        """
        Args:
            dataset: a Dataset yielding dicts with 'src_ids' and 'tgt_ids'.
            tokens_per_batch: approximate max sum of tokens per batch.
        """
        self.dataset = dataset
        self.tokens_per_batch = tokens_per_batch
        # Precompute lengths
        self.example_lengths: List[int] = []
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            src_len = len(item["src_ids"])
            tgt_len = len(item["tgt_ids"])
            self.example_lengths.append(src_len + tgt_len)
        # Create sorted index order by length ascending
        self.sorted_indices = sorted(
            range(len(self.example_lengths)),
            key=lambda i: self.example_lengths[i],
        )
        # Precompute batches
        self.batches: List[List[int]] = []
        batch: List[int] = []
        batch_tokens = 0
        for idx in self.sorted_indices:
            length = self.example_lengths[idx]
            # start new batch if adding this example would exceed budget
            if batch and batch_tokens + length > self.tokens_per_batch:
                self.batches.append(batch)
                batch = []
                batch_tokens = 0
            batch.append(idx)
            batch_tokens += length
        if batch:
            self.batches.append(batch)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


class DatasetLoader:
    """Central loader for machine translation and parsing datasets.

    Builds tokenizers, reads raw data, numericalizes, and provides DataLoaders
    with dynamic batching by token count.
    """

    def __init__(self, cfg: Config) -> None:
        """
        Args:
            cfg: Config object with required keys under 'data', 'training', 'parsing', etc.
        """
        self.cfg = cfg
        # Build a single tokenizer for both tasks (expects cfg.dataset.* to be set)
        self.tokenizer = build_tokenizer(cfg)
        # Caches
        self.datasets: Dict[str, Dataset] = {}
        self.dataloaders: Dict[str, DataLoader] = {}

    def load_data(self, split: str) -> Dataset:
        """Load or retrieve a Dataset for the given split.

        Args:
            split: Identifier for data split. For MT: 'train', 'dev', 'test'.
                   For parsing: '<wsj|semi>_<train|dev|test>', e.g., 'wsj_train'.

        Returns:
            A torch.utils.data.Dataset.

        Raises:
            KeyError: If split is unknown or required config keys are missing.
        """
        if split in self.datasets:
            return self.datasets[split]

        # Machine Translation splits
        if split in {"train", "dev", "test"}:
            dataset = self._load_mt_split(split)
        # Parsing splits: e.g., 'wsj_train', 'semi_dev'
        elif "_" in split:
            dataset = self._load_parsing_split(split)
        else:
            raise KeyError(f"Unknown data split: '{split}'")

        self.datasets[split] = dataset
        return dataset

    def get_dataloader(self, split: str) -> DataLoader:
        """Get a DataLoader for the given split, with dynamic batching.

        Args:
            split: Same as load_data.

        Returns:
            A DataLoader yielding padded batches.
        """
        if split not in self.datasets:
            self.load_data(split)
        if split in self.dataloaders:
            return self.dataloaders[split]

        dataset = self.datasets[split]
        tokens_per_batch = self.cfg.get("training.tokens_per_batch")
        num_workers = self.cfg.get("data.num_workers", 0)

        batch_sampler = TokenBucketBatchSampler(dataset, tokens_per_batch)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
        )
        self.dataloaders[split] = loader
        return loader

    def _load_mt_split(self, split: str) -> TranslationDataset:
        """Load machine translation (EN-DE) data for split."""
        # Config keys for EN-DE
        src_key = f"data.mt.en_de.{split}_src"
        tgt_key = f"data.mt.en_de.{split}_tgt"
        src_path = self.cfg.get(src_key)
        tgt_path = self.cfg.get(tgt_key)
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"MT source file not found: {src_path}")
        if not os.path.isfile(tgt_path):
            raise FileNotFoundError(f"MT target file not found: {tgt_path}")

        src_seqs: List[List[int]] = []
        tgt_seqs: List[List[int]] = []

        with open(src_path, "r", encoding="utf-8") as fs, open(
            tgt_path, "r", encoding="utf-8"
        ) as ft:
            for src_line, tgt_line in zip(fs, ft):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if not src_line or not tgt_line:
                    continue
                # Tokenize to IDs
                src_ids = self.tokenizer.encode(src_line).ids
                tgt_ids = self.tokenizer.encode(tgt_line).ids
                src_seqs.append(src_ids)
                tgt_seqs.append(tgt_ids)

        return TranslationDataset(src_seqs, tgt_seqs)

    def _load_parsing_split(self, split: str) -> ParsingDataset:
        """Load constituency parsing data for split like 'wsj_train' or 'semi_dev'."""
        prefix, suffix = split.split("_", 1)
        cfg_prefix = prefix if prefix != "semi" else "semi_supervised"
        key = f"data.parsing.{cfg_prefix}.{suffix}_trees"
        file_path = self.cfg.get(key)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Parsing data file not found: {file_path}")

        src_seqs: List[List[int]] = []
        tgt_seqs: List[List[int]] = []

        # Simple bracket parsing to extract leaves and tokens
        def extract_leaves(tree_str: str) -> List[str]:
            # all non-paren tokens are leaves and labels; leaves follow POS tags
            tokens = []
            depth = 0
            parts = tree_str.replace("(", " ( ").replace(")", " ) ").split()
            # leaves are those tokens that are not '(' or ')', but occur at leaf positions:
            # pattern: (... ( POS WORD ) ...)
            for i in range(len(parts) - 2):
                if parts[i] == "(" and parts[i + 2] == ")" and parts[i + 1] != "(":
                    # this is a POS tag followed by WORD: parts[i+1] is POS, parts[i+2] is WORD?
                    # But generic extraction: collect parts[i+1] if the next is not another '('
                    # For simplicity, collect parts[i + 1]
                    tokens.append(parts[i + 1])
            return tokens

        def tokenize_brackets(tree_str: str) -> List[str]:
            # split every paren as separate token
            return tree_str.replace("(", " ( ").replace(")", " ) ").split()

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tree = line.strip()
                if not tree:
                    continue
                # source sentence: leaf words
                words = extract_leaves(tree)
                # target sequence: bracket tokens
                bracket_tokens = tokenize_brackets(tree)
                # numericalize via tokenizer (join tokens with spaces)
                src_text = " ".join(words)
                tgt_text = " ".join(bracket_tokens)
                src_ids = self.tokenizer.encode(src_text).ids
                tgt_ids = self.tokenizer.encode(tgt_text).ids
                src_seqs.append(src_ids)
                tgt_seqs.append(tgt_ids)

        return ParsingDataset(src_seqs, tgt_seqs)

    def _collate_fn(
        self, batch: List[Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """Collate function to pad sequences and create masks."""
        # Extract lists
        src_batch = [ex["src_ids"] for ex in batch]
        tgt_batch = [ex["tgt_ids"] for ex in batch]
        # Compute lengths
        src_lens = [len(x) for x in src_batch]
        tgt_lens = [len(x) for x in tgt_batch]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)
        pad_id: int = self.tokenizer.pad_token_id or 0

        # Prepare padded tensors and masks
        padded_src = []
        padded_tgt = []
        src_masks = []
        tgt_masks = []
        for src, tgt in zip(src_batch, tgt_batch):
            # pad source
            pad_src = src + [pad_id] * (max_src - len(src))
            mask_src = [1] * len(src) + [0] * (max_src - len(src))
            padded_src.append(pad_src)
            src_masks.append(mask_src)
            # pad target
            pad_tgt = tgt + [pad_id] * (max_tgt - len(tgt))
            mask_tgt = [1] * len(tgt) + [0] * (max_tgt - len(tgt))
            padded_tgt.append(pad_tgt)
            tgt_masks.append(mask_tgt)

        # Convert to tensors
        src_ids_tensor = torch.tensor(padded_src, dtype=torch.long)
        tgt_ids_tensor = torch.tensor(padded_tgt, dtype=torch.long)
        src_mask_tensor = torch.tensor(src_masks, dtype=torch.bool)
        tgt_mask_tensor = torch.tensor(tgt_masks, dtype=torch.bool)

        return {
            "src_ids": src_ids_tensor,
            "tgt_ids": tgt_ids_tensor,
            "src_mask": src_mask_tensor,
            "tgt_mask": tgt_mask_tensor,
        }
