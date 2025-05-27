## data_loader.py

import os
import random
from typing import List, Tuple, Dict, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from config import Config
from tokenizer import Tokenizer


class ParallelDataset(Dataset):
    """
    PyTorch Dataset for parallel corpora.
    Each item is a tuple of (source_token_ids, target_token_ids).
    """

    def __init__(self, src_ids: List[List[int]], tgt_ids: List[List[int]]) -> None:
        """
        Args:
            src_ids: List of token ID lists for source sentences.
            tgt_ids: List of token ID lists for target sentences.
        """
        assert len(src_ids) == len(tgt_ids), \
            "Source and target lists must be the same length"
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_ids[idx], self.tgt_ids[idx]


class DynamicBatchSampler(Sampler[List[int]]):
    """
    Sampler that groups dataset indices into batches such that
    the total number of source tokens and target tokens in each batch
    does not exceed specified budgets.
    """

    def __init__(
        self,
        dataset: ParallelDataset,
        batch_source_tokens: int,
        batch_target_tokens: int,
        shuffle: bool = False
    ) -> None:
        """
        Args:
            dataset: ParallelDataset to sample from.
            batch_source_tokens: Maximum total source tokens per batch.
            batch_target_tokens: Maximum total target tokens per batch.
            shuffle: Whether to shuffle data before batching.
        """
        self.dataset = dataset
        self.batch_source_tokens = batch_source_tokens
        self.batch_target_tokens = batch_target_tokens

        # Precompute lengths
        self.src_lens = [len(s) for s in dataset.src_ids]
        self.tgt_lens = [len(t) for t in dataset.tgt_ids]

        # Prepare indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        src_sum = 0
        tgt_sum = 0

        for idx in self.indices:
            src_len = self.src_lens[idx]
            tgt_len = self.tgt_lens[idx]

            # If adding this example exceeds budgets, yield current batch
            if batch and (
                src_sum + src_len > self.batch_source_tokens or
                tgt_sum + tgt_len > self.batch_target_tokens
            ):
                yield batch
                batch = []
                src_sum = 0
                tgt_sum = 0

            batch.append(idx)
            src_sum += src_len
            tgt_sum += tgt_len

        # Yield any remaining batch
        if batch:
            yield batch

    def __len__(self) -> int:
        # Not strictly required, provide an estimate
        total_src = sum(self.src_lens)
        total_tgt = sum(self.tgt_lens)
        # Conservative estimate: average of the two
        est_batches_src = total_src / self.batch_source_tokens
        est_batches_tgt = total_tgt / self.batch_target_tokens
        return int(max(est_batches_src, est_batches_tgt)) + 1


class DatasetLoader:
    """
    Builds PyTorch DataLoaders for train/validation/test splits
    from raw parallel corpora, using dynamic batching by token budgets.
    """

    def __init__(self, config: Config, tokenizer: Tokenizer) -> None:
        """
        Args:
            config: Config object with data paths and training parameters.
            tokenizer: Tokenizer for encoding text to token IDs.
        """
        self.config = config
        self.tokenizer = tokenizer

        paths = self.config.get_data_paths()
        # Load and preprocess each split
        self.train_dataset = self._load_split(
            src_path=paths["train_src"],
            tgt_path=paths["train_tgt"]
        )
        self.val_dataset = self._load_split(
            src_path=paths["val_src"],
            tgt_path=paths["val_tgt"]
        )
        self.test_dataset = self._load_split(
            src_path=paths["test_src"],
            tgt_path=paths["test_tgt"]
        )

    def _load_split(self, src_path: str, tgt_path: str) -> ParallelDataset:
        """
        Reads raw parallel files, encodes them, and returns a ParallelDataset.

        Args:
            src_path: Path to source-language text file.
            tgt_path: Path to target-language text file.

        Returns:
            ParallelDataset with tokenized and numericalized data.
        """
        raw_pairs = self._read_parallel(src_path, tgt_path)
        return self._make_dataset(raw_pairs)

    def _read_parallel(self, src_path: str, tgt_path: str) -> List[Tuple[str, str]]:
        """
        Reads two parallel text files and returns aligned sentence pairs.

        Args:
            src_path: Path to source file.
            tgt_path: Path to target file.

        Returns:
            List of (source_line, target_line) pairs.
        """
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")
        if not os.path.isfile(tgt_path):
            raise FileNotFoundError(f"Target file not found: {tgt_path}")

        pairs: List[Tuple[str, str]] = []
        with open(src_path, "r", encoding="utf-8") as f_src, \
             open(tgt_path, "r", encoding="utf-8") as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if not src_line or not tgt_line:
                    continue
                pairs.append((src_line, tgt_line))
        return pairs

    def _make_dataset(
        self,
        raw_pairs: List[Tuple[str, str]]
    ) -> ParallelDataset:
        """
        Tokenizes and numericalizes raw text pairs to build a ParallelDataset.

        Args:
            raw_pairs: List of (source_str, target_str) pairs.

        Returns:
            ParallelDataset with token ID sequences.
        """
        if not raw_pairs:
            return ParallelDataset([], [])

        src_texts, tgt_texts = zip(*raw_pairs)  # type: ignore
        # Encode batches (includes special tokens)
        src_ids = self.tokenizer.encode_batch(list(src_texts))
        tgt_ids = self.tokenizer.encode_batch(list(tgt_texts))
        return ParallelDataset(src_ids, tgt_ids)

    def _collate_fn(
        self,
        batch: List[Tuple[List[int], List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function to convert a list of examples into padded tensors.

        Args:
            batch: List of (src_ids, tgt_ids) pairs.

        Returns:
            Dict with keys:
              - "src": LongTensor of shape (B, max_src_len)
              - "tgt_input": LongTensor of shape (B, max_tgt_len)
              - "tgt_output": LongTensor of shape (B, max_tgt_len)
              - "src_lengths": LongTensor of shape (B,)
              - "tgt_lengths": LongTensor of shape (B,)
        """
        # Split batch
        src_batch, tgt_batch = zip(*batch)  # type: ignore

        # Prepare target input/output by shifting
        tgt_input_batch = [seq[:-1] for seq in tgt_batch]
        tgt_output_batch = [seq[1:] for seq in tgt_batch]

        # Compute lengths
        src_lens = [len(seq) for seq in src_batch]
        tgt_lens = [len(seq) for seq in tgt_input_batch]

        # Determine max lengths
        max_src_len = max(src_lens) if src_lens else 0
        max_tgt_len = max(tgt_lens) if tgt_lens else 0

        # Get pad token ID from tokenizer
        try:
            pad_id = self.tokenizer.tokenizer.token_to_id("<pad>")
        except Exception:
            pad_id = 0

        batch_size = len(batch)
        # Initialize padded tensors
        src_tensor = torch.full(
            (batch_size, max_src_len),
            pad_id,
            dtype=torch.long
        )
        tgt_input_tensor = torch.full(
            (batch_size, max_tgt_len),
            pad_id,
            dtype=torch.long
        )
        tgt_output_tensor = torch.full(
            (batch_size, max_tgt_len),
            pad_id,
            dtype=torch.long
        )

        # Copy data
        for i, (src_seq, tgt_in_seq, tgt_out_seq) in enumerate(
            zip(src_batch, tgt_input_batch, tgt_output_batch)
        ):
            src_tensor[i, :len(src_seq)] = torch.tensor(src_seq, dtype=torch.long)
            tgt_input_tensor[i, :len(tgt_in_seq)] = torch.tensor(
                tgt_in_seq, dtype=torch.long
            )
            tgt_output_tensor[i, :len(tgt_out_seq)] = torch.tensor(
                tgt_out_seq, dtype=torch.long
            )

        return {
            "src": src_tensor,
            "tgt_input": tgt_input_tensor,
            "tgt_output": tgt_output_tensor,
            "src_lengths": torch.tensor(src_lens, dtype=torch.long),
            "tgt_lengths": torch.tensor(tgt_lens, dtype=torch.long),
        }

    def get_dataloader(self, split: str) -> DataLoader:
        """
        Returns a DataLoader for the specified data split.

        Args:
            split: One of "train", "val", or "test".

        Returns:
            DataLoader yielding batches as dicts from _collate_fn().
        """
        params = self.config.get_training_params()
        batch_src = params["batch_source_tokens"]
        batch_tgt = params["batch_target_tokens"]

        if split == "train":
            dataset = self.train_dataset
            shuffle = True
        elif split == "val":
            dataset = self.val_dataset
            shuffle = False
        elif split == "test":
            dataset = self.test_dataset
            shuffle = False
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")

        sampler = DynamicBatchSampler(
            dataset=dataset,
            batch_source_tokens=batch_src,
            batch_target_tokens=batch_tgt,
            shuffle=shuffle
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self._collate_fn,
            num_workers=4,
            pin_memory=True
        )
