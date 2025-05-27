## main.py

import argparse
import os
import sys

import torch

from utils import Config
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluation import Evaluator


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate the Transformer model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "train_eval"],
        default="train_eval",
        help="Operation mode: 'train', 'eval', or 'train_eval'."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a model checkpoint to load before training/evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (checkpoints, logs, metrics). "
             "If not set, uses config.training.ckpt_dir."
    )
    return parser.parse_args()


def main():
    """
    Main entry point for training and evaluation.
    """
    args = parse_args()

    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    output_dir = args.output_dir or config.get("training.ckpt_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # Build data loaders
    print("Loading data and building tokenizers...")
    loader = DatasetLoader(config)
    train_loader, val_loader, test_loader = loader.load_data()
    print("Data loaders are ready.")

    # Initialize model
    print("Initializing Transformer model...")
    model = TransformerModel(config)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs, using DataParallel.")
        model = torch.nn.DataParallel(model)

    # Load checkpoint if provided
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            print(f"Checkpoint file not found: {args.ckpt}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading checkpoint from {args.ckpt} ...")
        # if DataParallel, unwrap
        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        m.load(args.ckpt)
        print("Checkpoint loaded.")

    # TRAINING
    if args.mode in ("train", "train_eval"):
        print("Starting training...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        trainer.train()
        print("Training finished.")

    # EVALUATION
    if args.mode in ("eval", "train_eval"):
        # Ensure model is in eval mode
        model.eval()
        print("Starting evaluation...")
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            config=config
        )
        metrics = evaluator.evaluate()
        bleu = metrics.get("bleu", None)
        if bleu is not None:
            print(f"Corpus BLEU = {bleu:.2f}")
        else:
            print("No BLEU score computed.", file=sys.stderr)
        # Save metrics to file
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"Metrics saved to {metrics_path}")

    print("All done.")


if __name__ == "__main__":
    main()
