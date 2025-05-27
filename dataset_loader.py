# dataset_loader.py

import os
import json
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tokenizers import ByteLevelBPETokenizer

from config import Config
import utils


class WMTDataset(Dataset):
    """Dataset wrapping tokenized source-target pairs."""

    def __init__(self, pairs: List[Tuple[List[int], List[int]]]) -> None:
        """
        Args:
            pairs: List of (src_token_ids, tgt_token_ids).
        """
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_ids, tgt_ids = self.pairs[idx]
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }


class TokenBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups examples by total tokens in source/target."""

    def __init__(
        self,
        dataset: WMTDataset,
        max_src_tokens: int,
        max_tgt_tokens: int,
        shuffle: bool = False,
    ) -> None:
        """
        Args:
            dataset: Instance of WMTDataset.
            max_src_tokens: Maximum total source tokens per batch.
            max_tgt_tokens: Maximum total target tokens per batch.
            shuffle: Whether to shuffle the data each epoch.
        """
        self.dataset = dataset
        self.max_src_tokens = max_src_tokens
        self.max_tgt_tokens = max_tgt_tokens
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        if self.shuffle:
            # In-place shuffle of indices
            torch.random.manual_seed(torch.initial_seed())
            self.indices = torch.randperm(len(self.dataset)).tolist()  # type: ignore

        batch: List[int] = []
        src_count = 0
        tgt_count = 0

        for idx in self.indices:
            item = self.dataset[idx]
            src_len = item["src"].size(0)
            tgt_len = item["tgt"].size(0)
            # If adding this example would exceed token budgets, yield current batch
            if batch and (
                src_count + src_len > self.max_src_tokens
                or tgt_count + tgt_len > self.max_tgt_tokens
            ):
                yield batch
                batch = []
                src_count = 0
                tgt_count = 0

            batch.append(idx)
            src_count += src_len
            tgt_count += tgt_len

        if batch:
            yield batch

    def __len__(self) -> int:
        # Rough estimate (may be off): total_examples / average batch size
        return len(self.dataset)


class DatasetLoader:
    """Loads raw data, builds tokenizer, and provides DataLoaders."""

    def __init__(self, config: Config) -> None:
        """
        Args:
            config: Configuration object.
        """
        self.config = config
        # Parse language pair, e.g., "en-de"
        pair = self.config.get("data.language_pair", "")
        if not pair or "-" not in pair:
            raise ValueError(
                f"Invalid language_pair '{pair}' in config; expected format 'src-tgt'"
            )
        self.src_lang, self.tgt_lang = pair.split("-", 1)
        self.dataset_name = self.config.get("data.train_dataset")
        self.vocab_type = self.config.get("data.vocab.type", "byte-pair")
        self.vocab_size = self.config.get("data.vocab.size", 37000)
        # Batch token limits
        self.max_src_tokens = self.config.get("training.batch.max_source_tokens", 25000)
        self.max_tgt_tokens = self.config.get("training.batch.max_target_tokens", 25000)
        # Number of DataLoader workers
        gpus = self.config.get("hardware.gpus", 0)
        self.num_workers = gpus if isinstance(gpus, int) and gpus > 0 else 0
        # Tokenizer storage directory
        # e.g., "./tokenizer/en-de/"
        self.tokenizer_dir = os.path.join("tokenizer", pair)
        # Prepare or load tokenizer
        self.tokenizer = self._prepare_tokenizer()
        # Padding token id
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.pad_token_id = pad_id if pad_id is not None else 0
        # Device for tensors
        self.device = utils.get_device(self.config)

    def _prepare_tokenizer(self) -> ByteLevelBPETokenizer:
        """
        Loads an existing tokenizer if found, else trains a new one.

        Returns:
            A ByteLevelBPETokenizer instance.
        """
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        vocab_path = os.path.join(self.tokenizer_dir, "vocab.json")
        merges_path = os.path.join(self.tokenizer_dir, "merges.txt")
        # If vocab/merges exist, load
        if os.path.isfile(vocab_path) and os.path.isfile(merges_path):
            tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path, lowercase=False)
            return tokenizer

        # Else train a new tokenizer
        # Expect raw data in data/{dataset_name}/train.{lang}
        data_root = os.path.join("data", self.dataset_name)
        src_file = os.path.join(data_root, f"train.{self.src_lang}")
        tgt_file = os.path.join(data_root, f"train.{self.tgt_lang}")
        if not os.path.isfile(src_file) or not os.path.isfile(tgt_file):
            raise FileNotFoundError(
                f"Training files not found at '{src_file}' or '{tgt_file}'."
            )

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[src_file, tgt_file],
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        )
        # Save to tokenizer_dir
        tokenizer.save_model(self.tokenizer_dir)
        # Reload to ensure consistency
        tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path, lowercase=False)
        return tokenizer

    def _read_raw_data(self, split: str) -> List[Tuple[str, str]]:
        """
        Reads parallel source-target raw text.

        Args:
            split: One of "train", "val", "test".

        Returns:
            List of (src_line, tgt_line) pairs.
        """
        data_root = os.path.join("data", self.dataset_name)
        src_path = os.path.join(data_root, f"{split}.{self.src_lang}")
        tgt_path = os.path.join(data_root, f"{split}.{self.tgt_lang}")
        if not os.path.isfile(src_path) or not os.path.isfile(tgt_path):
            raise FileNotFoundError(
                f"Data files for split '{split}' not found at "
                f"'{src_path}' or '{tgt_path}'."
            )

        pairs: List[Tuple[str, str]] = []
        with open(src_path, "r", encoding="utf-8") as fs, open(
            tgt_path, "r", encoding="utf-8"
        ) as ft:
            for s_line, t_line in zip(fs, ft):
                s = s_line.strip()
                t = t_line.strip()
                if s and t:
                    pairs.append((s, t))
        return pairs

    def _encode_pairs(self, raw_pairs: List[Tuple[str, str]]) -> List[Tuple[List[int], List[int]]]:
        """
        Tokenizes and converts text pairs to ID sequences.

        Args:
            raw_pairs: List of (src_line, tgt_line).

        Returns:
            List of (src_ids, tgt_ids).
        """
        encoded: List[Tuple[List[int], List[int]]] = []
        for src_text, tgt_text in raw_pairs:
            src_enc = self.tokenizer.encode(src_text)
            tgt_enc = self.tokenizer.encode(tgt_text)
            encoded.append((src_enc.ids, tgt_enc.ids))
        return encoded

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collates and pads a list of examples into batched tensors and masks.

        Args:
            batch: List of dicts with 'src' and 'tgt' LongTensors.

        Returns:
            Dict with keys: 'src', 'tgt', 'src_mask', 'tgt_mask'.
        """
        src_seqs = [ex["src"] for ex in batch]
        tgt_seqs = [ex["tgt"] for ex in batch]
        batch_size = len(src_seqs)
        # Determine max lengths
        max_src_len = max([seq.size(0) for seq in src_seqs])
        max_tgt_len = max([seq.size(0) for seq in tgt_seqs])
        # Prepare padded tensors
        padded_src = torch.full(
            (batch_size, max_src_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        padded_tgt = torch.full(
            (batch_size, max_tgt_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        for i, seq in enumerate(src_seqs):
            padded_src[i, : seq.size(0)] = seq
        for i, seq in enumerate(tgt_seqs):
            padded_tgt[i, : seq.size(0)] = seq

        # Create masks
        src_mask = utils.create_padding_mask(padded_src, self.pad_token_id)
        tgt_mask = utils.create_combined_mask(padded_tgt, self.pad_token_id)

        # Move to device
        padded_src = padded_src.to(self.device)
        padded_tgt = padded_tgt.to(self.device)
        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)

        return {
            "src": padded_src,
            "tgt": padded_tgt,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
        }

    def load_data(self) -> Dict[str, DataLoader]:
        """
        Builds DataLoaders for train, val, and test splits.

        Returns:
            Dict mapping split names to DataLoaders.
        """
        loaders: Dict[str, DataLoader] = {}
        for split in ["train", "val", "test"]:
            # 1) Read raw sentences
            raw = self._read_raw_data(split)
            # 2) Encode to ID sequences
            encoded = self._encode_pairs(raw)
            # 3) Build dataset
            dataset = WMTDataset(encoded)
            # 4) Build sampler
            sampler = TokenBatchSampler(
                dataset,
                max_src_tokens=self.max_src_tokens,
                max_tgt_tokens=self.max_tgt_tokens,
                shuffle=(split == "train"),
            )
            # 5) DataLoader
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=self._collate_fn,
                pin_memory=True,
                num_workers=self.num_workers,
            )
            loaders[split] = loader
        return loaders
