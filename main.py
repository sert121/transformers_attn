# main.py

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from config import Config
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluator import Evaluator
import utils


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate the Transformer model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "all"],
        default="all",
        help="Mode of operation: 'train', 'eval', or 'all' (default: all).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to load for resuming training or for evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.info("Starting Transformer experiment")

    # Load configuration
    logging.info(f"Loading configuration from '{args.config}'")
    config = Config(args.config)

    # Set random seeds
    seed = config.get("training.seed", 42)
    logging.info(f"Setting random seed to {seed}")
    set_seed(seed)

    # Select device
    device = utils.get_device(config)
    logging.info(f"Using device: {device}")

    # Prepare data
    logging.info("Loading datasets and tokenizers...")
    data_loader = DatasetLoader(config)
    dataloaders = data_loader.load_data()
    logging.info(
        f"Loaded splits: {', '.join(dataloaders.keys())}; "
        f"Train batches: {len(dataloaders['train'])}"
    )

    # Build model
    logging.info("Building Transformer model...")
    model = TransformerModel(config)
    model.to(device)

    # Multi-GPU support
    num_gpus = config.get("hardware.gpus", 0)
    if device.type == "cuda" and isinstance(num_gpus, int) and num_gpus > 1:
        logging.info(f"Wrapping model with DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)

    # Training phase
    if args.mode in ("train", "all"):
        logging.info("Initializing Trainer...")
        trainer = Trainer(config, model, dataloaders)
        if args.checkpoint:
            logging.info(f"Resuming training from checkpoint '{args.checkpoint}'")
            trainer.load_checkpoint(args.checkpoint)
        trainer.train()

    # Evaluation phase
    if args.mode in ("eval", "all"):
        # Determine which checkpoint to load
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpt_dir = config.get("training.checkpoint_dir", "checkpoints")
            ckpt_path = os.path.join(ckpt_dir, "ckpt_final.pt")
        logging.info(f"Evaluating with checkpoint '{ckpt_path}'")

        evaluator = Evaluator(config, model, dataloaders["test"])
        metrics = evaluator.evaluate(checkpoint_path=ckpt_path)
        logging.info("Evaluation results:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value}")

    logging.info("Experiment finished.")


if __name__ == "__main__":
    main()
