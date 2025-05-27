"""
main.py

Entry point for reproducing the Transformer experiments from
"Attention Is All You Need". This script will:
  1. Parse command-line arguments to get a config file path.
  2. Load and validate the experiment configuration.
  3. Build the dataset (SentencePiece vocab, data loaders).
  4. Instantiate the Transformer model and optimizer.
  5. Train the model according to the specified schedule.
  6. Evaluate the model on the development set and report BLEU.
"""

import argparse
import sys
import time
from typing import Optional

import torch

from config import Config
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluation import Evaluation


class Main:
    """
    Main orchestrator for the Transformer experiment.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the experiment:
          - Load configuration
          - Build/load vocabulary
          - Prepare data loaders
          - Instantiate model, optimizer, trainer, evaluator

        Args:
            config_path (str): Path to YAML or JSON config file.
        """
        start_time = time.time()
        # 1) Load and validate configuration
        print(f"[INFO] Loading configuration from: {config_path}")
        self.config = Config(config_path)
        print("[INFO] Configuration:")
        self.config.dump()

        # 2) Prepare dataset and tokenizer
        print("[INFO] Building/loading SentencePiece vocabulary")
        self.loader = DatasetLoader(self.config)

        # Verify that train/dev splits are accessible (will raise if misconfigured)
        print("[INFO] Verifying data splits")
        _ = self.loader.load_data("train")
        _ = self.loader.load_data("dev")

        # 3) Instantiate model
        print("[INFO] Instantiating Transformer model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerModel(self.config).to(self.device)

        # 4) Build optimizer
        print("[INFO] Setting up optimizer")
        self.optimizer = self._build_optimizer()

        # 5) Instantiate trainer and evaluator
        print("[INFO] Initializing Trainer")
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loader=self.loader,
            config=self.config
        )
        print("[INFO] Initializing Evaluation harness")
        self.evaluator = Evaluation(
            model=self.model,
            loader=self.loader,
            config=self.config
        )

        elapsed = time.time() - start_time
        print(f"[INFO] Initialization complete in {elapsed:.2f}s")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build the optimizer as specified in config.optimizer.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        Raises:
            ValueError: If optimizer type is unsupported.
        """
        opt_cfg = self.config.optimizer
        optim_type = opt_cfg.type.lower()
        if optim_type == "adam":
            optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                eps=opt_cfg.eps
            )
            return optimizer
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_cfg.type}")

    def run_experiment(self) -> None:
        """
        Execute the full experiment: training and evaluation.
        """
        print("[INFO] Starting training")
        self.trainer.train()
        print("[INFO] Training finished. Starting evaluation on dev set")
        results = self.evaluator.evaluate(split="dev")
        bleu = results.get("BLEU", None)
        if bleu is not None:
            print(f"[RESULT] Development set BLEU = {bleu:.2f}")
        else:
            print("[WARN] BLEU score not found in evaluation results")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with attribute `config`.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce 'Attention Is All You Need' experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration file (YAML or JSON)."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the script.
    """
    args = parse_args()
    try:
        experiment = Main(config_path=args.config)
        experiment.run_experiment()
    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
