## dataset_loader.py

import os
import json
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from utils import Config


class DatasetLoader:
    """
    Loads parallel corpora, builds or loads subword tokenizers,
    encodes text into integer sequences, and constructs PyTorch
    DataLoaders with dynamic batching by token count.
    """

    def __init__(self, config: Config) -> None:
        # Data file paths
        self.train_src_path: str = config.get("data.train_src")
        self.train_tgt_path: str = config.get("data.train_tgt")
        self.val_src_path: str = config.get("data.val_src")
        self.val_tgt_path: str = config.get("data.val_tgt")
        self.test_src_path: str = config.get("data.test_src")
        self.test_tgt_path: str = config.get("data.test_tgt")

        # Tokenizer settings
        self.vocab_type: str = config.get("data.vocab_type", "bpe")
        self.vocab_size: int = config.get("data.vocab_size", 37000)

        # Batching
        self.max_tokens_per_batch: int = config.get(
            "data.max_tokens_per_batch", 25000
        )

        # Optional: number of worker processes for DataLoader
        self.num_workers: int = config.get("data.num_workers", 4)

        # Tokenizers (to be built or loaded)
        self.src_tokenizer: Tokenizer = None  # type: ignore
        self.tgt_tokenizer: Tokenizer = None  # type: ignore

        # Where to save/load tokenizer files
        self.tokenizer_dir: str = config.get("data.tokenizer_dir", "tokenizers")
        os.makedirs(self.tokenizer_dir, exist_ok=True)

    def build_tokenizers(self) -> None:
        """
        Train or load source and target tokenizers based on config.
        """
        # Filenames for tokenizer artifacts
        src_tok_file = os.path.join(
            self.tokenizer_dir, f"src_{self.vocab_type}_tokenizer.json"
        )
        tgt_tok_file = os.path.join(
            self.tokenizer_dir, f"tgt_{self.vocab_type}_tokenizer.json"
        )

        # If both exist, load and return
        if os.path.isfile(src_tok_file) and os.path.isfile(tgt_tok_file):
            self.src_tokenizer = Tokenizer.from_file(src_tok_file)
            self.tgt_tokenizer = Tokenizer.from_file(tgt_tok_file)
            return

        # Otherwise, train new tokenizers
        # Select model and trainer
        if self.vocab_type.lower() == "bpe":
            src_model = BPE(unk_token="<unk>")
            tgt_model = BPE(unk_token="<unk>")
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            )
        elif self.vocab_type.lower() == "wordpiece":
            src_model = WordPiece(unk_token="<unk>")
            tgt_model = WordPiece(unk_token="<unk>")
            trainer = WordPieceTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            )
        else:
            raise ValueError(f"Unsupported vocab_type: {self.vocab_type}")

        # Train source tokenizer
        self.src_tokenizer = Tokenizer(src_model)
        self.src_tokenizer.pre_tokenizer = Whitespace()
        self.src_tokenizer.train([self.train_src_path], trainer)

        # Train target tokenizer
        self.tgt_tokenizer = Tokenizer(tgt_model)
        self.tgt_tokenizer.pre_tokenizer = Whitespace()
        self.tgt_tokenizer.train([self.train_tgt_path], trainer)

        # Post-processing: add start/end tokens
        src_s_id = self.src_tokenizer.token_to_id("<s>")
        src_e_id = self.src_tokenizer.token_to_id("</s>")
        self.src_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", src_s_id), ("</s>", src_e_id)],
        )

        tgt_s_id = self.tgt_tokenizer.token_to_id("<s>")
        tgt_e_id = self.tgt_tokenizer.token_to_id("</s>")
        self.tgt_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", tgt_s_id), ("</s>", tgt_e_id)],
        )

        # Save tokenizers to disk
        self.src_tokenizer.save(src_tok_file)
        self.tgt_tokenizer.save(tgt_tok_file)

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build tokenizers (if needed), create Datasets and DataLoaders
        for train, validation, and test splits.
        """
        # Ensure tokenizers exist
        if self.src_tokenizer is None or self.tgt_tokenizer is None:
            self.build_tokenizers()

        # Inner dataset for parallel text
        class ParallelTextDataset(Dataset):
            def __init__(
                self,
                src_path: str,
                tgt_path: str,
                src_tokenizer: Tokenizer,
                tgt_tokenizer: Tokenizer,
            ):
                # Read lines
                with open(src_path, "r", encoding="utf-8") as f_src:
                    self.src_lines = [line.strip() for line in f_src]
                with open(tgt_path, "r", encoding="utf-8") as f_tgt:
                    self.tgt_lines = [line.strip() for line in f_tgt]
                assert len(self.src_lines) == len(
                    self.tgt_lines
                ), "Source and target files must have same number of lines."
                self.src_tokenizer = src_tokenizer
                self.tgt_tokenizer = tgt_tokenizer

            def __len__(self) -> int:
                return len(self.src_lines)

            def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
                src_enc = self.src_tokenizer.encode(self.src_lines[idx])
                tgt_enc = self.tgt_tokenizer.encode(self.tgt_lines[idx])
                return {
                    "src_ids": torch.tensor(src_enc.ids, dtype=torch.long),
                    "tgt_ids": torch.tensor(tgt_enc.ids, dtype=torch.long),
                }

        # Instantiate datasets
        train_dataset = ParallelTextDataset(
            self.train_src_path,
            self.train_tgt_path,
            self.src_tokenizer,
            self.tgt_tokenizer,
        )
        val_dataset = ParallelTextDataset(
            self.val_src_path,
            self.val_tgt_path,
            self.src_tokenizer,
            self.tgt_tokenizer,
        )
        test_dataset = ParallelTextDataset(
            self.test_src_path,
            self.test_tgt_path,
            self.src_tokenizer,
            self.tgt_tokenizer,
        )

        # Custom sampler for dynamic batching by token count
        class TokenBucketSampler(Sampler[List[int]]):
            def __init__(self, dataset: ParallelTextDataset, max_tokens: int):
                self.dataset = dataset
                self.max_tokens = max_tokens

            def __iter__(self):
                bucket: List[int] = []
                tokens_in_bucket = 0
                for idx in range(len(self.dataset)):
                    item = self.dataset[idx]
                    # count both src and tgt tokens
                    n_tokens = item["src_ids"].size(0) + item["tgt_ids"].size(0)
                    # if adding this item exceeds budget, yield current bucket
                    if bucket and tokens_in_bucket + n_tokens > self.max_tokens:
                        yield bucket
                        bucket = []
                        tokens_in_bucket = 0
                    bucket.append(idx)
                    tokens_in_bucket += n_tokens
                # yield any remaining
                if bucket:
                    yield bucket

            def __len__(self) -> int:
                # approximate number of batches
                total_tokens = 0
                for idx in range(len(self.dataset)):
                    item = self.dataset[idx]
                    total_tokens += (
                        item["src_ids"].size(0) + item["tgt_ids"].size(0)
                    )
                return max(1, total_tokens // self.max_tokens)

        # Collate function for padding
        def pad_collate_fn(batch_items: List[Dict[str, torch.LongTensor]]):
            # Extract pad IDs
            pad_src = self.src_tokenizer.token_to_id("<pad>")
            pad_tgt = self.tgt_tokenizer.token_to_id("<pad>")

            # Compute max lengths
            max_src_len = max(item["src_ids"].size(0) for item in batch_items)
            max_tgt_len = max(item["tgt_ids"].size(0) for item in batch_items)

            # Prepare batched tensors
            batch_size = len(batch_items)
            src_batch = torch.full(
                (batch_size, max_src_len), pad_src, dtype=torch.long
            )
            tgt_batch = torch.full(
                (batch_size, max_tgt_len), pad_tgt, dtype=torch.long
            )

            for i, item in enumerate(batch_items):
                src_len = item["src_ids"].size(0)
                tgt_len = item["tgt_ids"].size(0)
                src_batch[i, :src_len] = item["src_ids"]
                tgt_batch[i, :tgt_len] = item["tgt_ids"]

            return {"src": src_batch, "tgt": tgt_batch}

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=TokenBucketSampler(train_dataset, self.max_tokens_per_batch),
            collate_fn=pad_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=TokenBucketSampler(val_dataset, self.max_tokens_per_batch),
            collate_fn=pad_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=TokenBucketSampler(test_dataset, self.max_tokens_per_batch),
            collate_fn=pad_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader
