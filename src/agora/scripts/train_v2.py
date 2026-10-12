#!/usr/bin/env python3
import argparse
import datetime
import sys
import traceback

import mlflow
import yaml

from agora.training.base_config import CONFIG_REGISTRY

# from agora.training.config_upd import Config
from agora.training.base_trainer import TRAINER_REGISTRY
from agora.utils.training import flatten_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MRI classification model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model-type", type=str, required=True, help="Model type used for training"
    )
    parser.add_argument(
        "--seed", type=int, default=None
    )  # Only used when submitting multiple jobs at the same time.
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and validate configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        required_keys = {"model", "training"}
        if not all(key in config for key in required_keys):
            raise ValueError(
                f"Configuration must contain all required keys: {required_keys}"
            )

        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_trainer(model_type: str):
    # import importlib

    if model_type not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(TRAINER_REGISTRY)}"
        )

    return TRAINER_REGISTRY[model_type]
    # module_path, class_name = TRAINER_REGISTRY[model_type].rsplit(".", 1)
    # module = importlib.import_module(module_path)
    # return getattr(module, class_name)


def get_config_class(model_type: str):
    if model_type not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(CONFIG_REGISTRY)}"
        )
    return CONFIG_REGISTRY[model_type]


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    try:
        CfgClass = get_config_class(args.model_type)
        cfg = CfgClass.from_yaml(args.config)
        # cfg = Config.from_yaml(args.config)

        if args.seed is not None:
            cfg.training.seed = args.seed

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{cfg.training.seed}"

        mlflow.set_experiment(f"Agora/{args.model_type}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(flatten_config(cfg.training, prefix="training"))
            mlflow.log_params(flatten_config(cfg.model, prefix="model"))

            # Create trainer
            TrainerClass = get_trainer(args.model_type)  # only import the necessary Trainer
            trainer = TrainerClass(cfg)

            # Start training
            trainer.train()

    except Exception as e:
        print(f"Error during training: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
