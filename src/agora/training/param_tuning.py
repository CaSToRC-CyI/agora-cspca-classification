"""
Modular Ray Tune hyperparameter optimization script.

Supports multiple trainer types via TRAINER_REGISTRY and SEARCH_SPACE_REGISTRY.
Each trainer has its own search space; static training config comes from a YAML file.

Usage:
    python tune.py --config path/to/config.yaml --trainer omnirad
    python tune.py --config path/to/config.yaml --trainer cnn3d --num-samples 30
"""

import argparse
import datetime
import os
import sys
import traceback
from functools import partial
from typing import Any, Dict

import ray
import torch
from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from rich.traceback import install

# ---------------------------------------------------------------------------
# Trainer + Config + Registries imports
# ---------------------------------------------------------------------------
from agora.training.base_config import (
    CONFIG_REGISTRY,
    BaseConfig,
)
from agora.training.base_trainer import (
    TRAINER_REGISTRY,
)

install(show_locals=True)


def get_trainer_class(model_type: str):
    if model_type not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: {list(TRAINER_REGISTRY.keys())}"
        )
    return TRAINER_REGISTRY[model_type]


def get_config_class(model_type: str):
    if model_type not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: {list(CONFIG_REGISTRY.keys())}"
        )
    return CONFIG_REGISTRY[model_type]


# ---------------------------------------------------------------------------
# TuneTrainerMixin
#
# Mix this in *before* the concrete trainer in MRO so that train() and
# setup_directories() resolve here first.
#
#   class OmniRadTuneTrainer(TuneTrainerMixin, OmniRadTrainer): pass
# ---------------------------------------------------------------------------


class TuneTrainerMixin:
    """
    Mixin that adds Ray Tune reporting and checkpoint resumption to any
    BaseTrainer subclass.  Override `train()`.
    """

    # ------------------------------------------------------------------
    # Reporting helper
    # ------------------------------------------------------------------

    def tune_report(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        is_best: bool,
        val_metrics: Dict[str, float] | None = None,
    ):
        """Report metrics (and optionally a checkpoint) back to Ray Tune."""
        metrics_to_report = {}

        if val_metrics:
            metrics_to_report["val_f1_score"] = val_metrics["f1"]
            metrics_to_report["val_loss"] = val_metrics["loss"]
            metrics_to_report["val_accuracy"] = val_metrics["accuracy"]
            metrics_to_report["val_precision"] = val_metrics["precision"]
            metrics_to_report["val_recall"] = val_metrics["recall"]
            metrics_to_report["val_roc_auc"] = val_metrics.get("roc_auc", 0.0)
            metrics_to_report["val_pr_auc"] = val_metrics.get("pr_auc", 0.0)
            metrics_to_report["best_threshold"] = val_metrics.get("best_threshold", 0.5)
        else:
            # Fall back to training metrics when no validation set is used
            metrics_to_report["val_f1_score"] = train_metrics["f1"]
            metrics_to_report["val_loss"] = train_metrics["loss"]

        checkpoint = None
        if is_best and val_metrics:
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),  # type: ignore[attr-defined]
                "optimizer_state_dict": self.optimizer.state_dict(),  # type: ignore[attr-defined]
                "metrics": val_metrics,
            }
            torch.save(
                checkpoint_state,
                os.path.join(self.checkpoint_dir, "model.pt"),
            )
            checkpoint = Checkpoint.from_directory(str(self.checkpoint_dir))

        tune.report(metrics_to_report, checkpoint=checkpoint)

    # ------------------------------------------------------------------
    # Training loop — mirrors BaseTrainer.train() with tune additions
    # ------------------------------------------------------------------

    def train(self):
        """Main training loop with Ray Tune checkpoint resume + reporting."""
        print(f"Starting training on {self.device}")  # type: ignore[attr-defined]

        start_time = datetime.datetime.now()
        patience = getattr(self.cfg.training, "early_stop_patience", 10)  # type: ignore[attr-defined]
        self.no_improvement_count = 0  # type: ignore[attr-defined]

        # ---- Resume from Ray Tune checkpoint if one exists ---------------
        ray_checkpoint = tune.get_checkpoint()
        if ray_checkpoint:
            with ray_checkpoint.as_directory() as ckpt_dir:
                ckpt = torch.load(
                    os.path.join(ckpt_dir, "model.pt"),
                    weights_only=False,
                )
                start_epoch = ckpt["epoch"] + 1
                self.model.load_state_dict(ckpt["model_state_dict"])  # type: ignore[attr-defined]
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore[attr-defined]
                print(f"Resumed from epoch {start_epoch}")
        else:
            start_epoch = getattr(self, "start_epoch", 0)

        # ---- Main loop ---------------------------------------------------
        for epoch in range(start_epoch, self.cfg.training.num_epochs):  # type: ignore[attr-defined]
            train_metrics = self.train_epoch(epoch)  # type: ignore[attr-defined]
            val_metrics = None

            if self.val_loader is not None:  # type: ignore[attr-defined]
                val_metrics = self.validate(epoch)  # type: ignore[attr-defined]
                current_metric = val_metrics["f1"]
                current_loss = val_metrics["loss"]

                print(f"\nEpoch {epoch + 1}/{self.cfg.training.num_epochs}")
                print("-" * 50)
                print(
                    "Training   | Loss: {:.4f}, Acc: {:.2f}%, F1: {:.2f}%, "
                    "Precision: {:.2f}%, Recall: {:.2f}%".format(
                        train_metrics["loss"],
                        train_metrics["accuracy"] * 100,
                        train_metrics["f1"] * 100,
                        train_metrics["precision"] * 100,
                        train_metrics["recall"] * 100,
                    )
                )
                print(
                    "Validation | Loss: {:.4f}, Acc: {:.2f}%, F1: {:.2f}%, "
                    "Precision: {:.2f}%, Recall: {:.2f}%".format(
                        val_metrics["loss"],
                        val_metrics["accuracy"] * 100,
                        val_metrics["f1"] * 100,
                        val_metrics["precision"] * 100,
                        val_metrics["recall"] * 100,
                    )
                )
            else:
                current_metric = train_metrics["f1"]
                current_loss = train_metrics["loss"]
                print(f"\nEpoch {epoch + 1}/{self.cfg.training.num_epochs}")
                print("-" * 50)
                print(
                    "Training   | Loss: {:.4f}, Acc: {:.2f}%, F1: {:.2f}%, "
                    "Precision: {:.2f}%, Recall: {:.2f}%".format(
                        train_metrics["loss"],
                        train_metrics["accuracy"] * 100,
                        train_metrics["f1"] * 100,
                        train_metrics["precision"] * 100,
                        train_metrics["recall"] * 100,
                    )
                )

            # Append to CSV log
            if val_metrics is not None:
                log_row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "train_f1": train_metrics["f1"],
                    "train_precision": train_metrics["precision"],
                    "train_recall": train_metrics["recall"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "train_pr_auc": train_metrics["pr_auc"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_f1": val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_best_thr": self.best_threshold,
                    "val_best_thr_tgt_recall": self.best_threshold_tgt_recall,
                    # "val_best_recall": val_best_rec,
                }
            else:
                log_row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "train_f1": train_metrics["f1"],
                    "train_precision": train_metrics["precision"],
                    "train_recall": train_metrics["recall"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "train_pr_auc": train_metrics["pr_auc"],
                }
            self.append_csv_log(self.csv_log_path, log_row)

            is_best = current_metric > self.best_metric  # type: ignore[attr-defined]
            is_best_loss = current_loss < self.best_loss  # type: ignore[attr-defined]

            if is_best:
                print(
                    f"=> New best metric: {current_metric:.4f} "
                    f"(was {self.best_metric:.4f})"  # type: ignore[attr-defined]
                )
                self.best_metric = current_metric  # type: ignore[attr-defined]

            if is_best_loss:
                self.best_loss = current_loss  # type: ignore[attr-defined]

            if is_best or is_best_loss:
                self.no_improvement_count = 0  # type: ignore[attr-defined]
            else:
                self.no_improvement_count += 1  # type: ignore[attr-defined]

            print("-" * 50 + "\n")

            # Report to Ray Tune every epoch
            self.tune_report(
                epoch=epoch,
                train_metrics=train_metrics,
                is_best=is_best,
                val_metrics=val_metrics,
            )

            # Early stopping
            if self.no_improvement_count >= patience:  # type: ignore[attr-defined]
                print(
                    f"\nEarly stopping triggered after {epoch + 1} epochs "
                    f"({patience} epochs without improvement)."
                )
                break

        self.writer.close()  # type: ignore[attr-defined]
        elapsed = datetime.datetime.now() - start_time
        h, rem = divmod(elapsed.total_seconds(), 3600)
        m, s = divmod(rem, 60)
        print(f"Training took {int(h)}h {int(m)}m {s:.2f}s")
        print("Training completed!")


# ---------------------------------------------------------------------------
# Dynamically build tune-aware trainer classes for every registered trainer
# ---------------------------------------------------------------------------


def _make_tune_trainer(trainer_cls):
    """Return a new class that mixes TuneTrainerMixin into trainer_cls."""
    return type(
        f"{trainer_cls.__name__}Tune",
        (TuneTrainerMixin, trainer_cls),
        {},
    )


TUNE_TRAINER_REGISTRY: Dict[str, Any] = {
    name: _make_tune_trainer(cls) for name, cls in TRAINER_REGISTRY.items()
}


# ---------------------------------------------------------------------------
# Search spaces
#
# Each entry is a dict of tune.* sampling expressions.
# Static params (data paths, num_epochs, etc.) stay in the YAML config.
# Tune-able params defined here will be overlaid on top of the loaded config.
# ---------------------------------------------------------------------------

# ---- Shared / common params used by most trainers -------------------------
_COMMON_SEARCH_SPACE: Dict[str, Any] = {
    # "loss_fn": tune.choice(["focal", "bce"]),
    "learning_rate": tune.loguniform(5e-5, 1e-2),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "weighted_sampling": tune.choice([True, False]),
}

# ---- Per-trainer search spaces --------------------------------------------
# TODO: fill in model-specific params for each trainer as needed.

SEARCH_SPACE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "omnirad": {
        **_COMMON_SEARCH_SPACE,
        "batch_size": tune.choice([4, 8]),
        # Separate LR / WD for the pretrained encoder vs the rest
        "finetune_lr": tune.loguniform(1e-6, 1e-4),
        "finetune_wd": tune.loguniform(1e-6, 1e-3),
        # TODO: add omnirad-specific architectural params here
        # e.g. "finetune_t": tune.choice(["v0", "v1"]),
        "finetune_t": tune.choice(["v2", "v5"]),
    },
    "radiodino": {
        **_COMMON_SEARCH_SPACE,
        "batch_size": tune.choice([4, 8]),
        "finetune_lr": tune.loguniform(1e-6, 1e-4),
        "finetune_wd": tune.loguniform(1e-6, 1e-3),
        # TODO: add radiodino-specific params here
        "finetune_t": tune.choice(["v2", "v5"]),
    },
    "cnn3d": {
        **_COMMON_SEARCH_SPACE,
        "batch_size": tune.choice([8, 16, 32]),
        # TODO: add CNN architectural params here
        # e.g. "initial_filters": tune.choice([8, 16, 32]),
        #      "dropout_rate": tune.uniform(0.1, 0.5),
    },
    "resnet3d": {
        **_COMMON_SEARCH_SPACE,
        "batch_size": tune.choice([8, 16, 32]),
        # TODO: add ResNet-specific params here
        # e.g. "dropout_rate": tune.uniform(0.1, 0.5),
    },
}


# ---------------------------------------------------------------------------
# Config builder
#
# Applies the tune trial config dict on top of the base YAML config.
# Extend `_apply_tune_config` below when adding new trainers / params.
# ---------------------------------------------------------------------------


def _apply_tune_config(cfg: BaseConfig, tune_config: Dict[str, Any]) -> BaseConfig:
    """
    Overlay the Ray Tune trial params onto the loaded BaseConfig.

    All params that live under cfg.training are handled generically via
    setattr.  Trainer-specific or nested params (e.g. cfg.model.*) should
    be handled in the explicit block below.
    """
    # ---- Generic training-level params ------------------------------------
    # TRAINING_PARAMS = {
    #     "loss_fn",
    #     "learning_rate",
    #     "weight_decay",
    #     "batch_size",
    #     "weighted_sampling",
    #     "finetune_lr",
    #     "finetune_wd",
    #     "optimizer",
    # }
    # for key in TRAINING_PARAMS:
    #     if key in tune_config:
    #         setattr(cfg.training, key, tune_config[key])

    # ---- Model-level params -----------------------------------------------
    # TODO: Add model-config overrides here as you expand search spaces.
    # Example:
    #   if "dropout_rate" in tune_config:
    #       cfg.model.dropout_rate = tune_config["dropout_rate"]
    #   if "initial_filters" in tune_config:
    #       cfg.model.initial_filters = tune_config["initial_filters"]
    #
    for key, value in tune_config.items():
        if hasattr(cfg.training, key):
            setattr(cfg.training, key, value)
        elif hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
        else:
            print(f"Warning: '{key}' not found, skipping.")

    return cfg


# ---------------------------------------------------------------------------
# Trainable function (one trial)
# ---------------------------------------------------------------------------


def tune_params(tune_config: Dict[str, Any], *, model_type: str, base_cfg: BaseConfig):
    """
    Entrypoint for a single Ray Tune trial.

    Args:
        tune_config:  Dict sampled by Ray Tune from the search space.
        model_type:   Trainer key, e.g. "omnirad".
        base_cfg:     Pre-loaded config object (from YAML).  We deep-copy it
                      so each trial gets its own independent instance.
    """
    import copy

    try:
        cfg = copy.deepcopy(base_cfg)
        cfg = _apply_tune_config(cfg, tune_config)

        TuneTrainerCls = TUNE_TRAINER_REGISTRY[model_type]
        trainer = TuneTrainerCls(cfg)
        trainer.train()

    except Exception:
        print(f"Trial failed with config: {tune_config}")
        print(traceback.format_exc())
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ray Tune hyperparameter search — modular trainer support"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the static YAML configuration file.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
        choices=list(TRAINER_REGISTRY.keys()),
        help="Which trainer / model type to tune.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Total number of hyperparameter trials (default: 20).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent trials — tune to your GPU count (default: 3).",
    )
    parser.add_argument(
        "--cpus-per-trial",
        type=int,
        default=10,
        help="CPU cores allocated per trial (default: 10).",
    )
    parser.add_argument(
        "--gpus-per-trial",
        type=int,
        default=1,
        help="GPUs allocated per trial (default: 1).",
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=10,
        help="ASHA grace period in epochs before early stopping kicks in (default: 10).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override num_epochs from YAML for the scheduler's max_t.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # ---- Load base config from YAML ---------------------------------------
    try:
        CfgClass = get_config_class(args.trainer)
        base_cfg = CfgClass.from_yaml(args.config)
    except Exception:
        print(f"Failed to load config from '{args.config}':")
        print(traceback.format_exc())
        sys.exit(1)

    # ---- Resolve search space for this trainer ----------------------------
    if args.trainer not in SEARCH_SPACE_REGISTRY:
        raise ValueError(
            f"No search space defined for trainer '{args.trainer}'. "
            f"Add one to SEARCH_SPACE_REGISTRY."
        )
    search_space = SEARCH_SPACE_REGISTRY[args.trainer]

    # ---- ASHA scheduler ---------------------------------------------------
    max_t = args.max_epochs or base_cfg.training.num_epochs
    scheduler = ASHAScheduler(
        metric="val_f1_score",
        mode="max",
        max_t=max_t,
        grace_period=args.grace_period,
        reduction_factor=2,
    )

    # ---- Ray init ---------------------------------------------------------
    ray.init(address="auto")

    # ---- Build trainable --------------------------------------------------
    # Use partial so tune_params only receives the tune-sampled config dict
    trainable = partial(tune_params, model_type=args.trainer, base_cfg=base_cfg)

    trainable_with_resources = tune.with_resources(
        trainable,
        resources={
            "cpu": args.cpus_per_trial,
            "gpu": args.gpus_per_trial,
        },
    )

    # ---- Tuner ------------------------------------------------------------
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent,
            scheduler=scheduler,
        ),
        run_config=tune.RunConfig(
            verbose=2,  # status and brief trial results
            failure_config=tune.FailureConfig(max_failures=2),
            # log_to_file=True,
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    # ---- Report best trial ------------------------------------------------
    best = results.get_best_result("val_f1_score", "max", scope="all")
    print("\n" + "=" * 60)
    print(f"Best trial — trainer: {args.trainer}")
    print(f"  Config:     {best.config}")
    print(f"  Val Loss:   {best.metrics.get('val_loss', 'N/A'):.4f}")
    print(f"  Val F1:     {best.metrics.get('val_f1_score', 'N/A'):.4f}")
    print(f"  Val ROC-AUC:{best.metrics.get('val_roc_auc', 'N/A'):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
