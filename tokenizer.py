## tokenizer.py

import os
from typing import List

from config import Config
from tokenizers import ByteLevelBPETokenizer


class Tokenizer:
    """
    Tokenizer wraps HuggingFace's ByteLevelBPETokenizer to train or load a BPE model,
    encode text sequences into token ID lists, and decode token ID lists back into strings.

    Public methods:
      - train_bpe(corpus_paths: List[str]) -> None
      - encode_batch(texts: List[str]) -> List[List[int]]
      - decode(token_ids: List[int], skip_special_tokens: bool = True) -> str
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Tokenizer. If a trained BPE vocab and merges files exist
        in the output directory, load them. Otherwise defer to train_bpe().

        Args:
            config: Config object providing data paths and BPE hyperparameters.
        """
        data_cfg = config.get_data_paths()
        # Directory to save/load BPE vocab and merges files
        self._tokenizer_dir: str = data_cfg["output_dir"]
        # Filenames for Byte-Pair Encoding model
        self._vocab_file: str = os.path.join(self._tokenizer_dir, "vocab.json")
        self._merges_file: str = os.path.join(self._tokenizer_dir, "merges.txt")
        # Hyperparameters
        self._vocab_size: int = int(data_cfg.get("vocab_size", 0))
        self._bpe_merges: int = int(data_cfg.get("bpe_merges", 0))
        self._tokenizer_type: str = data_cfg.get("tokenizer", "")
        # Underlying tokenizer instance (None until loaded or trained)
        self.tokenizer: ByteLevelBPETokenizer = None  # type: ignore

        # Load existing BPE model if present
        if (
            os.path.isfile(self._vocab_file)
            and os.path.isfile(self._merges_file)
            and self._tokenizer_type == "byte_pair_encoding"
        ):
            self.tokenizer = ByteLevelBPETokenizer(
                self._vocab_file,
                self._merges_file,
                lowercase=False,
                add_prefix_space=True,
            )

    def train_bpe(self, corpus_paths: List[str]) -> None:
        """
        Trains a Byte-Pair Encoding tokenizer on the provided corpora and
        saves the resulting vocab and merges files.

        Args:
            corpus_paths: List of file paths (text files) to train on.

        Raises:
            ValueError: If tokenizer type is not 'byte_pair_encoding'.
            RuntimeError: If training fails.
        """
        if self._tokenizer_type != "byte_pair_encoding":
            raise ValueError(
                f"Unsupported tokenizer type '{self._tokenizer_type}'. "
                "Expected 'byte_pair_encoding'."
            )

        # Ensure output directory exists
        os.makedirs(self._tokenizer_dir, exist_ok=True)

        # Initialize and train the BPE tokenizer
        bpe_tokenizer = ByteLevelBPETokenizer(lowercase=False, add_prefix_space=True)
        try:
            bpe_tokenizer.train(
                files=corpus_paths,
                vocab_size=self._vocab_size,
                min_frequency=2,
                special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
                show_progress=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to train BPE tokenizer: {e}")

        # Save the trained model files
        bpe_tokenizer.save_model(self._tokenizer_dir)

        # Reload into self.tokenizer for encoding/decoding
        self.tokenizer = ByteLevelBPETokenizer(
            self._vocab_file,
            self._merges_file,
            lowercase=False,
            add_prefix_space=True,
        )

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encodes a batch of text strings into lists of token IDs.

        Args:
            texts: List of input strings to tokenize.

        Returns:
            A list where each element is the list of token IDs for the corresponding input string.

        Raises:
            RuntimeError: If the tokenizer is not initialized (loaded or trained).
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call train_bpe() or provide pretrained files.")

        # Tokenize each text, adding special tokens (<s>, </s>)
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        return [enc.ids for enc in encodings]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a sequence of token IDs back into a string.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: If True, special tokens (e.g., <pad>, <s>, </s>, <unk>) are removed.

        Returns:
            The decoded string.

        Raises:
            RuntimeError: If the tokenizer is not initialized.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Cannot decode.")

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
