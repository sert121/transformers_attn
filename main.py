## main.py

import argparse
import logging
import os
import sys
from typing import Dict

import torch

from config import Config
from tokenizer import Tokenizer
from data_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluator import Evaluator


def parse_args() -> Dict[str, str]:
    """
    Parse command-line arguments for config file, data paths, output directory,
    and optional checkpoint to resume from.

    Returns:
        A dict mapping argument names to their values.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Transformer model (Attention Is All You Need)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config.yaml)."
    )
    parser.add_argument(
        "--train_src",
        type=str,
        default="",
        help="Path to the training source language file."
    )
    parser.add_argument(
        "--train_tgt",
        type=str,
        default="",
        help="Path to the training target language file."
    )
    parser.add_argument(
        "--val_src",
        type=str,
        default="",
        help="Path to the validation source language file."
    )
    parser.add_argument(
        "--val_tgt",
        type=str,
        default="",
        help="Path to the validation target language file."
    )
    parser.add_argument(
        "--test_src",
        type=str,
        default="",
        help="Path to the test source language file."
    )
    parser.add_argument(
        "--test_tgt",
        type=str,
        default="",
        help="Path to the test target language file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to write outputs (checkpoints, BPE model, etc.)."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to a checkpoint to resume training from."
    )
    args = parser.parse_args()
    # Return as a dict compatible with Config
    return {
        "config": args.config,
        "train_src": args.train_src,
        "train_tgt": args.train_tgt,
        "val_src": args.val_src,
        "val_tgt": args.val_tgt,
        "test_src": args.test_src,
        "test_tgt": args.test_tgt,
        "output_dir": args.output_dir,
        "resume_from": args.resume_from,
    }


def setup_logging() -> None:
    """
    Configure the root logger to output INFO-level logs with timestamps.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )


def main() -> None:
    """
    Main entry point:
      1. Parse CLI arguments and load configuration.
      2. Initialize or train the BPE tokenizer.
      3. Build data loaders for train/val/test splits.
      4. Instantiate the Transformer model.
      5. Optionally resume from a checkpoint.
      6. Train the model.
      7. Evaluate on the test set.
    """
    # -------------------------
    # 1. Setup and configuration
    # -------------------------
    setup_logging()
    args = parse_args()
    logging.info("Loading configuration from %s", args["config"])
    try:
        config = Config(args)
    except Exception as e:
        logging.error("Failed to load config: %s", e)
        sys.exit(1)

    data_paths = config.get_data_paths()
    train_params = config.get_training_params()

    # -------------------------
    # 2. Tokenizer (BPE) setup
    # -------------------------
    logging.info("Initializing tokenizer...")
    tokenizer = Tokenizer(config)
    # If no pretrained BPE model was found, train one
    if tokenizer.tokenizer is None:
        logging.info("Training BPE tokenizer on corpora: %s and %s",
                     data_paths["train_src"], data_paths["train_tgt"])
        tokenizer.train_bpe([data_paths["train_src"], data_paths["train_tgt"]])
        logging.info("BPE tokenizer training complete; model saved to %s",
                     data_paths["output_dir"])
    else:
        logging.info("Loaded existing BPE tokenizer from %s", data_paths["output_dir"])

    # -------------------------
    # 3. Data loaders
    # -------------------------
    logging.info("Preparing datasets and data loaders...")
    try:
        data_loader = DatasetLoader(config, tokenizer)
        train_loader = data_loader.get_dataloader("train")
        val_loader = data_loader.get_dataloader("val")
        test_loader = data_loader.get_dataloader("test")
    except Exception as e:
        logging.error("Failed to prepare data loaders: %s", e)
        sys.exit(1)

    # -------------------------
    # 4. Model instantiation
    # -------------------------
    logging.info("Building Transformer model...")
    model_params = config.get_model_params()
    # Add vocabulary and special token indices
    model_params["vocab_size"] = int(data_paths.get("vocab_size", 0))
    # Retrieve special token IDs from tokenizer (fallback to defaults)
    try:
        pad_idx = tokenizer.tokenizer.token_to_id("<pad>")
        bos_idx = tokenizer.tokenizer.token_to_id("<s>")
        eos_idx = tokenizer.tokenizer.token_to_id("</s>")
    except Exception:
        pad_idx, bos_idx, eos_idx = 0, 1, 2
    model_params["pad_idx"] = pad_idx
    model_params["bos_idx"] = bos_idx
    model_params["eos_idx"] = eos_idx
    model_params["length_penalty"] = float(train_params.get("length_penalty", 0.0))

    model = TransformerModel(model_params)

    # -------------------------
    # 5. Resume from checkpoint
    # -------------------------
    resume_ckpt = args.get("resume_from", "")
    if resume_ckpt:
        if os.path.isfile(resume_ckpt):
            logging.info("Resuming training from checkpoint %s", resume_ckpt)
            # Load via Trainer after instantiation
        else:
            logging.warning(
                "Requested resume checkpoint '%s' not found. Starting from scratch.",
                resume_ckpt
            )

    # -------------------------
    # 6. Training
    # -------------------------
    logging.info("Starting training for %d steps.",
                 train_params.get("train_steps"))
    trainer = Trainer(model, config, train_loader, val_loader)
    if resume_ckpt and os.path.isfile(resume_ckpt):
        try:
            trainer.load_checkpoint(resume_ckpt)
        except Exception as e:
            logging.error("Failed to load checkpoint: %s", e)
            sys.exit(1)
    trainer.train()
    logging.info("Training complete.")

    # -------------------------
    # 7. Evaluation on test set
    # -------------------------
    if data_paths["test_src"] and data_paths["test_tgt"]:
        logging.info("Evaluating on test set...")
        evaluator = Evaluator(model, tokenizer, config)
        results = evaluator.evaluate(test_loader)
        for metric, score in results.items():
            logging.info("%s = %.2f", metric, score)
    else:
        logging.warning("Test files not provided; skipping evaluation.")

    logging.info("Done.")


if __name__ == "__main__":
    main()
