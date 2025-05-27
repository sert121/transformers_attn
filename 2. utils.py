## 2. utils.py

import os
import json
import logging
import tempfile
import re
from typing import Any, Dict, List, Set, Tuple, Union

import sacrebleu
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, Tokenizer as HFTokenizer
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_json(path: str, obj: Any) -> None:
    """Save a Python object to a JSON file with indentation.

    Args:
        path: File path to write JSON to.
        obj: The Python object (dict, list, etc.) to serialize.

    Raises:
        IOError: If the file cannot be written.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved JSON to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise


def load_json(path: str) -> Any:
    """Load a Python object from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        logger.debug(f"Loaded JSON from {path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def build_tokenizer(cfg: Any) -> PreTrainedTokenizerFast:
    """Build or load a subword tokenizer based on configuration.

    Expects the following config entries under 'dataset':
      - tokenizer_dir: str, path to save or load tokenizer files
      - train_files: List[str], raw text files for training tokenizer
      - tokenizer_type: str, one of {'bpe', 'wordpiece'}
      - vocab_size: int, desired vocabulary size
      - special_tokens: List[str], e.g., ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

    Args:
        cfg: Config object with tokenizer parameters.

    Returns:
        A HuggingFace PreTrainedTokenizerFast.

    Raises:
        KeyError: If required config keys are missing.
        FileNotFoundError: If raw data files are missing.
        RuntimeError: If tokenizer training or loading fails.
    """
    # Read configuration
    tok_dir: str = cfg.get("dataset.tokenizer_dir")
    train_files: List[str] = cfg.get("dataset.train_files")
    tok_type: str = cfg.get("dataset.tokenizer_type", "bpe")
    vocab_size: int = cfg.get("dataset.vocab_size")
    special_tokens: List[str] = cfg.get("dataset.special_tokens", [])

    # Ensure output directory exists
    os.makedirs(tok_dir, exist_ok=True)

    # Check if pretrained tokenizer exists
    try:
        # This will load from tok_dir if a valid tokenizer is there
        hf_tok = PreTrainedTokenizerFast.from_pretrained(tok_dir, use_fast=True)
        logger.info(f"Loaded existing tokenizer from {tok_dir}")
        return hf_tok
    except Exception:
        logger.info(f"No existing tokenizer found in {tok_dir}; training a new one.")

    # Verify raw files exist
    for fpath in train_files:
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Tokenizer training file not found: {fpath}")

    # Train a new tokenizer
    if tok_type.lower() == "bpe":
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=train_files,
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
        )
        # Save the tokenizer model files: vocab.json and merges.txt
        tokenizer.save_model(tok_dir)
    elif tok_type.lower() == "wordpiece":
        tokenizer = BertWordPieceTokenizer(
            lowercase=False, unk_token="[UNK]"
        )
        tokenizer.train(
            files=train_files,
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
        )
        # BertWordPieceTokenizer saves vocab.txt by default
        tokenizer.save_model(tok_dir)
    else:
        raise KeyError(f"Unsupported tokenizer type: {tok_type}")

    # Wrap into a HF PreTrainedTokenizerFast for a unified API
    try:
        hf_tok = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=tokenizer.token_to_id(tokenizer.token_to_id or "[UNK]"),
            pad_token=special_tokens[0] if special_tokens else None,
            cls_token="[CLS]" if "[CLS]" in special_tokens else None,
            sep_token="[SEP]" if "[SEP]" in special_tokens else None,
        )
        # Save the tokenizer for future loads
        hf_tok.save_pretrained(tok_dir)
        logger.info(f"Trained and saved new tokenizer to {tok_dir}")
        return hf_tok
    except Exception as e:
        logger.error(f"Failed to wrap tokenizer with HF PreTrainedTokenizerFast: {e}")
        raise RuntimeError(f"Error in building HF tokenizer: {e}") from e


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score using sacrebleu, as in the paper.

    Args:
        predictions: List of system outputs (detokenized sentences).
        references: List of reference translations.

    Returns:
        BLEU score (float), same units as sacrebleu.score.

    Raises:
        ValueError: If input lists have different lengths.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) != references ({len(references)})"
        )
    # sacrebleu expects list of references as list of list
    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
        force=True,
        lowercase=False,
        tokenize="13a",
    )
    score = float(bleu.score)
    logger.info(f"Computed BLEU score: {score:.2f}")
    return score


def compute_f1(pred_trees: List[str], gold_trees: List[str]) -> float:
    """Compute constituency parsing F1 by span matching.

    Uses a pure-Python bracket parser to extract labeled spans and computes
    global precision, recall and F1 over all sentences.

    Args:
        pred_trees: List of predicted bracketed tree strings.
        gold_trees: List of gold-standard bracketed tree strings.

    Returns:
        F1 score as a percentage (0.0–100.0).

    Raises:
        ValueError: If input lists have different lengths or empty.
    """
    if len(pred_trees) != len(gold_trees):
        raise ValueError(
            f"Predicted tree count ({len(pred_trees)}) != gold tree count ({len(gold_trees)})"
        )
    if not pred_trees:
        return 0.0

    def _tokenize_brackets(s: str) -> List[str]:
        # Split parentheses and labels
        return re.findall(r"\(|\)|[^\s()]+", s)

    class _Node:
        __slots__ = ("label", "children")

        def __init__(self, label: str) -> None:
            self.label = label
            self.children: List["_Node"] = []

    def _parse(tokens: List[str]) -> _Node:
        # Recursive descent parser for bracketed tree
        if not tokens or tokens[0] != "(":
            raise ValueError("Invalid tree format, expected '('")
        tokens.pop(0)  # remove '('
        label = tokens.pop(0)
        node = _Node(label)
        # Read children or terminals until ')'
        while tokens and tokens[0] != ")":
            if tokens[0] == "(":
                child = _parse(tokens)
                node.children.append(child)
            else:
                # leaf token
                leaf = _Node(tokens.pop(0))
                node.children.append(leaf)
        if not tokens:
            raise ValueError("Unbalanced parentheses in tree string")
        tokens.pop(0)  # remove ')'
        return node

    def _collect_spans(
        node: _Node, start: int, spans: Set[Tuple[str, int, int]]
    ) -> int:
        # Returns next index after consuming this node
        if not node.children:
            # leaf: consumes one word
            return start + 1
        cur = start
        for child in node.children:
            cur = _collect_spans(child, cur, spans)
        # record only non-terminal spans
        spans.add((node.label, start, cur))
        return cur

    total_pred_spans: Set[Tuple[str, int, int]] = set()
    total_gold_spans: Set[Tuple[str, int, int]] = set()

    for pred_str, gold_str in zip(pred_trees, gold_trees):
        try:
            pred_tokens = _tokenize_brackets(pred_str)
            gold_tokens = _tokenize_brackets(gold_str)
            pred_root = _parse(pred_tokens.copy())
            gold_root = _parse(gold_tokens.copy())
        except Exception as e:
            logger.warning(f"Skipping tree due to parse error: {e}")
            continue

        pred_spans: Set[Tuple[str, int, int]] = set()
        gold_spans: Set[Tuple[str, int, int]] = set()
        _collect_spans(pred_root, 0, pred_spans)
        _collect_spans(gold_root, 0, gold_spans)

        total_pred_spans.update(pred_spans)
        total_gold_spans.update(gold_spans)

    if not total_pred_spans or not total_gold_spans:
        return 0.0

    tp = len(total_pred_spans & total_gold_spans)
    prec = tp / len(total_pred_spans) if total_pred_spans else 0.0
    rec = tp / len(total_gold_spans) if total_gold_spans else 0.0
    if prec + rec == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    f1_percent = f1 * 100.0
    logger.info(
        f"Computed parsing F1: TP={tp}, Pred={len(total_pred_spans)}, "
        f"Gold={len(total_gold_spans)}, F1={f1_percent:.2f}"
    )
    return f1_percent
