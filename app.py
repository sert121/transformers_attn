"""app.py

Entry point for training and evaluating the Transformer model
as described in 'Attention Is All You Need'.
"""

import os
import argparse
import logging
import sys
from typing import Any

import yaml

import utils
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluator import Evaluator


class Config:
    """
    Simple configuration handler wrapping a YAML file.
    """

    def __init__(self, path: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML configuration file.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            self._cfg = yaml.safe_load(f)
        if not isinstance(self._cfg, dict):
            raise ValueError(f"Config file {path} is not a valid YAML mapping")

    def get(self, section: str) -> Any:
        """
        Retrieve a configuration section.

        Args:
            section (str): Top-level section name.

        Returns:
            Any: The value of the section (usually a dict).

        Raises:
            KeyError: If the section is not present.
        """
        if section not in self._cfg:
            raise KeyError(f"Section '{section}' not found in configuration")
        return self._cfg[section]


class App:
    """
    Application orchestrator for data loading, model training, and evaluation.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the application.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Load configuration
        self.config = Config(config_path)

        # Setup logging directories
        logging_cfg = self.config.get("logging")
        log_dir = logging_cfg.get("log_dir", "logs/")
        ckpt_dir = logging_cfg.get("checkpoint_dir", "checkpoints/")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Configure root logger to also write to a file
        log_file = os.path.join(log_dir, "app.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"App initialized with config: {config_path}")

    def run(self) -> None:
        """
        Execute the full pipeline: set seed, load data, build model,
        train, and evaluate.
        """
        # 1. Set random seed for reproducibility
        training_cfg = self.config.get("training")
        seed = int(training_cfg.get("seed", 42))
        utils.set_seed(seed)

        # 2. Prepare data: tokenizer and datasets
        loader = DatasetLoader(self.config)
        tokenizer = loader.load_tokenizer()
        train_ds, dev_ds, test_ds = loader.load_datasets()
        self.logger.info(
            f"Datasets loaded: train={len(train_ds)}, "
            f"dev={len(dev_ds)}, test={len(test_ds)}"
        )

        # 3. Instantiate model
        model_cfg = self.config.get("model")
        vocab_size = tokenizer.get_vocab_size()
        model = TransformerModel(model_cfg, vocab_size)
        self.logger.info(
            f"Model created: TransformerModel(d_model={model_cfg.get('d_model')}, "
            f"num_layers={model_cfg.get('num_layers')}, "
            f"vocab_size={vocab_size})"
        )

        # 4. Train the model
        trainer = Trainer(model, (train_ds, dev_ds), self.config)
        ckpt_path = trainer.train()
        self.logger.info(f"Training complete. Last checkpoint saved at: {ckpt_path}")

        # 5. Evaluate on test set
        evaluator = Evaluator(model, tokenizer, self.config)
        metrics = evaluator.evaluate(split="test")
        self.logger.info(f"Test evaluation metrics: {metrics}")

        # 6. Print final metrics
        print("\nFinal evaluation metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


def main() -> None:
    """
    Parse arguments and launch the application.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Transformer model "
                    "from 'Attention Is All You Need'."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    try:
        app = App(args.config)
        app.run()
    except Exception as e:
        logging.getLogger("App").error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
