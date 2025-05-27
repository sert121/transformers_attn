import argparse
import os
import random
import logging
from typing import Optional, List

import numpy as np
import torch

from config import Config
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluator import Evaluator

# Set up root logger
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("main")


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def override_model_hparams_from_parsing(cfg: Config) -> None:
    """
    Override the 'model' section in cfg with parsing-specific hyperparameters.
    Expects cfg to contain a 'parsing' subsection with keys:
      num_layers, d_model, d_ff, num_heads, dropout
    """
    # Retrieve parsing hyperparams
    parsing_keys = ["num_layers", "d_model", "d_ff", "num_heads", "dropout"]
    for key in parsing_keys:
        try:
            val = cfg.get(f"parsing.{key}")
        except KeyError as e:
            raise KeyError(f"Missing parsing hyperparameter '{key}'") from e
        # Override model section
        # Access internal dict directly since Config has no setter
        cfg._cfg.setdefault("model", {})[key] = val
    logger.info("Overrode model hyperparameters with parsing settings")


def run_machine_translation(cfg: Config) -> None:
    """Train and evaluate the Transformer on the WMT machine translation task."""
    logger.info("Starting machine translation task")

    # Prepare data loaders
    loader = DatasetLoader(cfg)
    train_loader = loader.get_dataloader("train")
    dev_loader = loader.get_dataloader("dev")
    test_loader = loader.get_dataloader("test")

    # Build model
    vocab_size = loader.tokenizer.vocab_size
    model = TransformerModel(cfg, vocab_size)

    # Train
    trainer = Trainer(model, train_loader, cfg)
    trainer.train()

    # Evaluate on test set
    evaluator = Evaluator(model, test_loader, cfg)
    mt_metrics = evaluator.evaluate_mt()
    logger.info(f"MT Test BLEU: {mt_metrics['bleu']:.2f}")


def run_parsing(cfg: Config, prefixes: Optional[List[str]] = None) -> None:
    """
    Train and evaluate the Transformer on constituency parsing.
    Runs for each 'prefix' in prefixes (e.g., 'wsj', 'semi').
    """
    if prefixes is None:
        prefixes = ["wsj", "semi"]

    for prefix in prefixes:
        logger.info(f"Starting parsing task for prefix '{prefix}'")

        # Override model hyperparameters for parsing
        override_model_hparams_from_parsing(cfg)

        # Instantiate a fresh loader for parsing data (different vocab, tokenizer)
        loader = DatasetLoader(cfg)
        split_train = f"{prefix}_train"
        split_dev = f"{prefix}_dev"
        split_test = f"{prefix}_test"

        train_loader = loader.get_dataloader(split_train)
        dev_loader = loader.get_dataloader(split_dev)
        test_loader = loader.get_dataloader(split_test)

        # Build model
        vocab_size = loader.tokenizer.vocab_size
        model = TransformerModel(cfg, vocab_size)

        # Train (monitoring dev F1 is beyond this script; uses fixed max_steps)
        trainer = Trainer(model, train_loader, cfg)
        trainer.train()

        # Evaluate on test set
        evaluator = Evaluator(model, test_loader, cfg)
        parse_metrics = evaluator.evaluate_parsing()
        logger.info(
            f"Parsing ({prefix}) Test F1: {parse_metrics['f1']:.2f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce 'Attention Is All You Need': "
                    "train/evaluate Transformer for MT and parsing."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["mt", "parsing", "all"],
        default="all",
        help="Which task to run: 'mt' (machine translation), "
             "'parsing' (constituency parsing), or 'all'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--parsing_prefixes",
        type=str,
        nargs="+",
        default=None,
        help="Prefixes for parsing splits (e.g., wsj, semi). "
             "Defaults to both ['wsj','semi'] if not set.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    cfg = Config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Set random seeds if provided
    if args.seed is not None:
        set_random_seeds(args.seed)

    # Run the requested task(s)
    if args.task in ("mt", "all"):
        run_machine_translation(cfg)

    if args.task in ("parsing", "all"):
        run_parsing(cfg, prefixes=args.parsing_prefixes)

    logger.info("All tasks completed.")


if __name__ == "__main__":
    main()
