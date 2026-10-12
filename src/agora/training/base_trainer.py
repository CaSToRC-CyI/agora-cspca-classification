import csv
import datetime
import json
import random
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Union

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.sampler
from scipy import interpolate
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

# ----------------- Misceleneous Specific Imports -----------------
from agora.data.dataloader import get_datasets

# ----------------- Model Specific Imports -----------------
from agora.models.model_3dcnn import CNN3D, CNN3DClinical
from agora.models.model_resnet3d import ResNet3D, ResNet3DClinical
from agora.models.omnirad import OmniRadProstateCsPCa, OmniRadProstateCsPCaClinical
from agora.models.utils import FINETUNE_MAP, count_trainable

# ----------------- Config Specific Imports -----------------
from agora.training.base_config import (
    BaseConfig,
    CNNConfig,
    OmniRadConfig,
    ResNetConfig,
)
from agora.utils.losses import SigmoidFocalLoss

CLINICAL_COLS: List[str] = ["patient_age", "psa", "psad", "prostate_volume"]


def main():
    print("testing")


if __name__ == "__main__":
    main()


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BaseTrainer(ABC):
    """Abstract base class for model training. Provides a common training framework
    including optimizer setup, loss function, checkpointing, scheduling, logging,
    and a full training loop. Model instantiation and architecture-specific logic
    are delegated to subclasses.

    Args:
        cfg (BaseConfig): Base configuration object. Subclasses should pass
            their specific config type (e.g. OmniRadConfig, CNNConfig).

    Raises:
        NotImplementedError: If subclass does not implement _build_model()
        NotImplementedError: If subclass does not implement _build_model_id()
        NotImplementedError: If subclass does not implement _project_name()
        NotImplementedError: If subclass does not implement _get_optimizer_param_groups()
        TypeError: If BaseTrainer is instantiated directly instead of through a subclass.

    Example:
        class OmniRadTrainer(BaseTrainer):
            def _build_model(self): ...
            def _build_model_id(self): ...
            def _project_name(self): ...
            def _get_optimizer_param_groups(self): ...

        cfg = OmniRadConfig.from_yaml("config.yaml")
        trainer = OmniRadTrainer(cfg)
        trainer.train()
    """

    model: nn.Module  # declaration for Pylance

    @classmethod
    def build_model_only(cls, cfg: BaseConfig):
        """Lightweight init — only builds the model.
        Essentially just a hacky way to load the proper models for inference/testing
        """
        instance = cls.__new__(cls)  # skip __init__
        instance.cfg = cfg
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.use_clinical = cfg.model.use_clinical
        instance.att_fusion = cfg.model.att_fusion
        instance._build_model()
        return instance

    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler(device=self.device)

        # Only used for pylance recognition
        self.use_clinical = cfg.model.use_clinical
        self.att_fusion = cfg.model.att_fusion

        # Reproducibility seed
        self.rep_seed = cfg.training.seed

        set_global_seeds(self.rep_seed)

        self.setup_directories()

        # Create datasets and dataloaders
        self.setup_data()

        self._build_model()  # subclass implements

        self._set_loss_fn()

        self._set_optimizer()

        # Load existing checkpoint if it exists
        self._load_checkpoint()

        self._set_scheduler()

        # Save training parameters to JSON
        self.save_params_json(cfg)

        # Temporary
        self.target_recall = 0.7

        # Metrics tracking
        self.best_threshold = float("-inf")
        self.best_threshold_tgt_recall = float("inf")
        self.best_metric = float("-inf")
        self.best_loss = float("inf")

    # ----------------- Initialization Methods -----------------

    def setup_directories(self):
        """Create necessary directories for logs and checkpoints."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.timestamp = timestamp
        self.model_id = self._build_model_id()  # subclass defines this
        project_name = self._project_name()  # subclass defines this

        unique_id = uuid.uuid4().hex[:6]  # Prevent job collision on slurm
        dir_name = "_".join([unique_id, timestamp, self.model_id])
        self.checkpoint_dir = (
            self.cfg.training.output_dir / "checkpoints" / project_name / dir_name
        )
        self.metrics_dir = (
            self.cfg.training.output_dir / "metrics" / project_name / dir_name
        )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.csv_log_path = self.metrics_dir / "training_log.csv"
        self.params_json_path = self.metrics_dir / "training_params.json"

        if not self.csv_log_path.exists():
            self.init_csv_logger(self.csv_log_path)

    @abstractmethod
    def _build_model_id(self) -> str:
        """Load model_id from model config in Trainer subclasses

        Raises:
            NotImplementedError: Function should be defined in trainer subclasses

        Returns:
            str: model id retrieved from model config
        """
        raise NotImplementedError

    @abstractmethod
    def _project_name(self) -> str:
        """Load project parent directory name from model config dataset(s) in Trainer subclasses

        Raises:
            NotImplementedError: Function should be defined in trainer subclasses

        Returns:
            str: Project parent directory name for logging
        """
        raise NotImplementedError

    def setup_data(self):
        """Setup datasets and dataloaders."""
        train_dataset, val_dataset, test_dataset, self.pos_weight = get_datasets(
            self.cfg, self.rep_seed, use_clinical=self.use_clinical
        )

        # Capture the train-split clinical normalization stats so they can be
        # persisted in the checkpoint and reused for external-dataset evaluation.
        self.clinical_stats = getattr(train_dataset.dataset, "clinical_stats", None)

        sampler, shuffle = None, True
        if self.cfg.training.weighted_sampling:
            sampler = self._get_weighted_sampler(train_dataset)
            shuffle = False

        g = torch.Generator()
        g.manual_seed(self.rep_seed)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.training.num_workers,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            generator=g,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            generator=g,
        )

    @abstractmethod
    def _build_model(self) -> None:
        """Build model specific to the subclass

        Raises:
            NotImplementedError: Function should be defined in trainer subclasses

        Returns:
            Pytorch model
        """
        raise NotImplementedError

    def _set_loss_fn(self):
        # Mapping from name to loss function
        loss_fn_dict = {
            "focal": SigmoidFocalLoss,
            "bce": nn.BCEWithLogitsLoss,
            "ce": nn.CrossEntropyLoss,
        }

        loss_fn_class = loss_fn_dict[self.cfg.training.loss_fn]

        pos = True
        pos_w = self.pos_weight if pos else None

        # Set the correct kwargs based on the loss type
        if self.cfg.training.loss_fn == "focal":
            loss_fn = loss_fn_class(
                alpha=self.pos_weight / (1 + self.pos_weight), gamma=0, reduction="mean"
            )  # a=0.75
        elif self.cfg.training.loss_fn == "bce":
            loss_fn = loss_fn_class(pos_weight=pos_w, reduction="mean")
        elif self.cfg.training.loss_fn == "ce":
            loss_fn = loss_fn_class(reduction="mean")  # can add pos_weight
        else:
            raise ValueError(f"Unsupported loss function: {self.cfg.training.loss_fn}")

        self.criterion = loss_fn

    def _set_optimizer(self):
        # Mapping from name to optimizer class
        optimizer_dict = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop,
            "adagrad": optim.Adagrad,
            # Add more if needed
        }

        optimizer_class = optimizer_dict[self.cfg.training.optimizer]

        # Set common optimizer arguments
        opt_kwargs = {
            "lr": self.cfg.training.learning_rate,
            "weight_decay": self.cfg.training.weight_decay,
        }

        param_groups = self._get_optimizer_param_groups()

        # If optimizer requires additional arguemnts add them like
        # opt_kwargs['arg'] = ....
        self.optimizer = optimizer_class(
            param_groups,
            **opt_kwargs,
        )

    @abstractmethod
    def _get_optimizer_param_groups(self) -> List[Dict]:
        """Load optimizer parameter groups based on Trainer subclasses model architecture.
        As an example we return all model parameters in base trainer

        Returns:
            List[dict]: List of dictionaries of parameter groups
        """
        param_groups = [
            {
                "params": self.model.parameters(),
                "lr": self.cfg.training.learning_rate,
                "weight_decay": self.cfg.training.weight_decay,
            },
        ]

        return param_groups

    def _load_checkpoint(self):
        # Load existing checkpoint if it exists
        if self.cfg.training.checkpoint_dir:
            print(f"Loading checkpoint from {self.cfg.training.checkpoint_dir}.")

            if self.cfg.training.checkpoint_dir.suffix == ".pt":
                self.chkpt_path = self.cfg.training.checkpoint_dir
            else:
                self.chkpt_path = self.cfg.training.checkpoint_dir / "best_model.pt"

            # Ideally we would load the "best_model.pth" but for now we load arbitrary ones
            checkpoint = torch.load(
                self.chkpt_path,
                weights_only=False,
            )

            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            self.start_epoch = 0

    def _set_scheduler(self):
        # TODO: Add a parameter in the config file to choose scheduler.
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", factor=0.8, patience=5
        # )
        warmup_epochs = int(
            self.cfg.training.warmup_frac * self.cfg.training.num_epochs
            - self.start_epoch
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.05,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),  # warmup
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.cfg.training.num_epochs
                    - warmup_epochs
                    - self.start_epoch,
                ),
            ],
            milestones=[warmup_epochs],
        )

    def save_params_json(self, config: Union[BaseConfig, OmniRadConfig, CNNConfig]):
        from dataclasses import is_dataclass

        def serialize(obj):
            """Recursively convert dataclass/Path objects to JSON-serializable types."""
            if is_dataclass(obj) and not isinstance(obj, type):
                return {k: serialize(v) for k, v in vars(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            return obj

        params: Any = serialize(config)
        params["class_weights"] = {
            "neg": float(1),
            "pos": float(self.pos_weight),
        }
        params["training"]["seed"] = int(self.rep_seed)
        params["model"]["model_id"] = self.model_id

        with self.params_json_path.open("w") as f:
            json.dump(params, f, indent=2)

    # ----------------- Training Related Methods -----------------

    def evaluate(self, best_val_threshold: float | None, best_epoch: int | None):
        self.model.eval()
        y_pred = []
        y_true = []
        y_pred_prob = []
        true_gs = []
        paths = []
        # For loss calc
        total_loss = 0.0
        total_samples = 0
        pred_thresh = best_val_threshold if best_val_threshold is not None else 0.5

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].float().to(self.device)
                true_gss = batch["true_gs"]
                paths_batch = batch["paths"]

                if self.use_clinical:
                    clinical = batch["clinical"].to(self.device)
                    outputs = self.model(images, clinical).squeeze()
                else:
                    outputs = self.model(images).squeeze()
                y_pred_prob.extend(torch.sigmoid(outputs).cpu().numpy())
                y_pred.extend((torch.sigmoid(outputs) > pred_thresh).cpu().numpy())
                y_true.extend(labels.cpu().numpy())
                true_gs.extend(true_gss)
                paths.extend(paths_batch[0])

                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

        # Calculate loss
        epoch_loss = total_loss / total_samples
        # self.analyze_mistakes(y_true, y_pred, true_gs, epoch)
        self.analyze_mistakes_paths(y_true, y_pred, true_gs, paths, best_epoch)

        # ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred)
        # plt.savefig(self.metrics_dir / f"confusion_matrix_{epoch}.png")
        conf_out = self.metrics_dir / f"confusion_matrix_{best_epoch}.png"
        class_names = ["non-csPCa", "csPCa"]
        self.plot_confusion_matrix(
            y_true, y_pred, out_path=conf_out, class_names=class_names
        )
        mlflow.log_artifact(str(conf_out), artifact_path="test_plots")

        fig, ax = plt.subplots(figsize=(8, 6))

        # opt_thresh, opt_tpr, opt_fpr = self.compute_optimal_threshold(y_true, y_pred_prob, method="youden")
        opt_tpr, opt_fpr = self.get_tpr_fpr_from_threshold(
            pred_thresh, y_true, y_pred_prob
        )

        ax.plot(
            opt_fpr,
            opt_tpr,
            "ro",
            label=f"Optimal Threshold = {pred_thresh:.2f}",
        )

        RocCurveDisplay.from_predictions(
            y_true, y_pred_prob, name="csPCa Classifier", plot_chance_level=True, ax=ax
        )
        roc_out = self.metrics_dir / f"roc_curve_{best_epoch}.png"
        fig.savefig(roc_out)
        plt.close(fig)
        mlflow.log_artifact(str(roc_out), artifact_path="test_plots")

        fig, ax = plt.subplots(figsize=(8, 6))

        # opt_thresh, opt_pr, opt_rec = self.compute_optimal_threshold(y_true, y_pred_prob, method="f1")
        opt_pr, opt_rec = self.get_pr_rec_from_threshold(
            pred_thresh, y_true, y_pred_prob
        )

        ax.plot(
            opt_rec,
            opt_pr,
            "ro",
            label=f"Optimal Threshold = {pred_thresh:.2f}",
        )

        PrecisionRecallDisplay.from_predictions(
            y_true, y_pred_prob, name="csPCa Classifier", plot_chance_level=True, ax=ax
        )
        pr_out = self.metrics_dir / f"pr_curve_{best_epoch}.png"
        fig.savefig(pr_out)
        plt.close(fig)
        mlflow.log_artifact(str(pr_out), artifact_path="test_plots")

        print("Test metric plots saved.")

        # Can modify compute_metrics to work with test set later on
        return {
            "loss": float(epoch_loss),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division="warn"),
            "recall": recall_score(y_true, y_pred, zero_division="warn"),
            "f1": f1_score(y_true, y_pred, zero_division="warn"),
            "roc_auc": roc_auc_score(y_true, y_pred_prob),
            "pr_auc": average_precision_score(y_true, y_pred_prob),
        }

    def analyze_mistakes_paths(self, y_true, y_pred, true_gs, paths, epoch):
        error_dict = defaultdict(int)
        error_paths = []
        for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            # if FP or FN (i.e. mistakes) keep track
            if y_t != y_p:
                error_dict[true_gs[i]] += 1
                error_paths.append((paths[i], true_gs[i], y_pred[i]))

        # Probably create a histogram plot of the errors
        # Extract keys and values
        # Sort the dictionary items by count in descending order
        sorted_items = sorted(error_dict.items(), key=lambda x: x[1], reverse=True)

        # Unpack the sorted items into labels and counts
        sorted_gs, sorted_counts = zip(*sorted_items)

        # Optional: convert to lists
        sorted_gs = list(sorted_gs)
        sorted_counts = list(sorted_counts)

        # Plot the histogram
        fig, ax = plt.subplots()
        ax.bar(sorted_gs, sorted_counts, color="skyblue", edgecolor="black")
        ax.set_xticks(
            ax.get_xticks(),
            [t.get_text() for t in ax.get_xticklabels()],
            rotation=45,
            ha="center",
        )
        ax.set_title(f"Histogram of model errors at epoch {epoch}")
        ax.set_xlabel("GS")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        gs_out = self.metrics_dir / f"GS_errors_{epoch}.png"
        fig.savefig(gs_out)
        plt.close(fig)
        mlflow.log_artifact(str(gs_out), artifact_path="test_plots")
        # plt.show()

        # with open(self.metrics_dir / f"err_paths_{epoch}.txt", "w") as f:
        #     f.write("\n".join(error_paths))
        with open(self.metrics_dir / f"err_paths_{epoch}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["Path", "True GS", "Predicted Label"])
            # Write data
            writer.writerows(error_paths)

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_threshold": self.best_threshold,
            "best_val_threshold_tgt_recall": self.best_threshold_tgt_recall,
            "target_recall": self.target_recall,
            # Train-split clinical normalization stats, reused for external eval.
            "clinical_stats": getattr(self, "clinical_stats", None),
        }

        # Save latest checkpoint
        save_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)

        # If best model, save separately
        if is_best:
            self.best_ckpt_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, self.best_ckpt_path)

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_outputs = []
        total_loss = 0.0
        total_samples = 0

        for i, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].float().to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                if self.use_clinical:
                    clinical = batch["clinical"].to(self.device)
                    outputs = self.model(images, clinical).squeeze(-1)
                else:
                    outputs = self.model(images).squeeze(-1)

                loss = self.criterion(outputs, labels)

            # Backward pass
            # loss.backward()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(
                self.optimizer
            )  # Uncomment if we add gradient clipping
            if i % 100 == 0:  # every 100 steps
                self.log_grad_norms(
                    self.model,
                    prefix=f"[Epoch {epoch} Step {epoch * len(self.train_loader) + i}] ",
                )

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
            if i % 100 == 0:  # every 100 steps
                self.log_grad_norms(
                    self.model,
                    prefix=f"[After clip - Epoch {epoch} Step {epoch * len(self.train_loader) + i}] ",
                )
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate predictions for metric calculation
            all_outputs.append(outputs)
            all_labels.append(labels)

            # Weighted loss accumulation
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            running_loss += loss.item()

            # Log every N batches
            if i % self.cfg.training.log_frequency == 0:
                avg_loss = running_loss / (i + 1)
                print(
                    f"Epoch [{epoch}/{self.cfg.training.num_epochs}] "
                    f"Batch [{i}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )

                step = epoch * len(self.train_loader) + i
                mlflow.log_metric("train/batch_loss", loss.item(), step=step)

        # Combine all batches into one tensor
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        # Compute Epoch level metrics from accumulated predictions
        metrics = self.compute_metrics(all_outputs, all_labels, val=False)
        # Compute average loss
        metrics["loss"] = total_loss / total_samples

        mlflow.log_metrics({f"train/{k}": v for k, v in metrics.items()}, step=epoch)

        return metrics

    def validate(self, epoch: int):
        """Run validation."""
        self.model.eval()

        all_labels = []
        all_outputs = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].float().to(self.device)

                if self.use_clinical:
                    clinical = batch["clinical"].to(self.device)
                    outputs = self.model(images, clinical).squeeze()
                else:
                    outputs = self.model(images).squeeze()

                loss = self.criterion(outputs, labels)

                all_outputs.append(outputs)
                all_labels.append(labels)
                # true_gs.extend(true_gss)

                # Weighted loss accumulation
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

        # Combine all batches into one tensor
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        metrics = self.compute_metrics(
            all_outputs, all_labels, val=True, tgt_recall=self.target_recall
        )

        # Compute average loss
        metrics["loss"] = total_loss / total_samples

        # Set the current best threshold for testing
        self.best_threshold = metrics["best_threshold"]
        self.best_threshold_tgt_recall = metrics["best_threshold_tgt_recall"]

        self.last_lr = self.scheduler.get_last_lr()

        # self.scheduler.step(metrics=metrics["loss"])
        self.scheduler.step()  # Cosine Annealing with warmup

        self.cur_lr = self.scheduler.get_last_lr()

        if self.last_lr != self.cur_lr:
            print(f"lr changed to: {self.cur_lr}")

        mlflow.log_metrics({f"val/{k}": v for k, v in metrics.items()}, step=epoch)

        return metrics

    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(
            f"{'Validation: ' + str(self.cfg.training.val_split * 100) + '%' if self.cfg.training.val_split else 'No validation'}"
        )

        # Log the time it took to train
        start_time = datetime.datetime.now()

        # Initialize early stopping variables
        patience = getattr(
            self.cfg.training, "early_stop_patience", 10
        )  # default patience
        self.no_improvement_count = 0

        for epoch in range(self.cfg.training.num_epochs):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            val_metrics = None

            mlflow.log_metric("learning_rate", self.scheduler.get_last_lr()[0], step=epoch)

            # Validation phase
            if self.val_loader is not None:
                val_metrics = self.validate(epoch)
                # Select on PR-AUC: it is threshold-free, so checkpoint
                # selection no longer depends on the (unstable) per-epoch F1
                # threshold, and it tracks the minority (positive) class under
                # class imbalance better than ROC-AUC.
                current_metric = val_metrics["pr_auc"]
                current_loss = val_metrics["loss"]

                print(f"\nEpoch {epoch + 1}/{self.cfg.training.num_epochs}")
                print("-" * 50)
                print(
                    "Training   | Loss: {:.4f}, Acc: {:.2f}%, F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%".format(
                        train_metrics["loss"],
                        train_metrics["accuracy"] * 100,
                        train_metrics["f1"] * 100,
                        train_metrics["precision"] * 100,
                        train_metrics["recall"] * 100,
                    )
                )
                print(
                    "Validation | Loss: {:.4f}, Acc: {:.2f}%, F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%".format(
                        val_metrics["loss"],
                        val_metrics["accuracy"] * 100,
                        val_metrics["f1"] * 100,
                        val_metrics["precision"] * 100,
                        val_metrics["recall"] * 100,
                    )
                )
            else:
                current_metric = train_metrics["pr_auc"]
                current_loss = train_metrics["loss"]
                print(f"\nEpoch {epoch + 1}/{self.cfg.training.num_epochs}")
                print("-" * 50)
                print(
                    "Training   | Loss: {:.4f}, Acc: {:.2f}%, F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%".format(
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

            # Save checkpoint
            is_best = False
            loss_threshold = 1.2  # allow up to 20% worse than best loss
            # prevent threshold exploit on the validation set
            if (
                current_metric > self.best_metric
                and current_loss < self.best_loss * loss_threshold
            ):
                is_best = True
            # is_best = current_metric > self.best_metric
            is_best_loss = current_loss < self.best_loss
            if is_best:
                print(
                    f"=> New val best metric: {current_metric} | Old val best metric: {self.best_metric}!"
                )
                self.best_metric = current_metric
                # self.save_metric_plots(epoch=epoch)

            if is_best_loss:
                self.best_loss = current_loss

            if is_best or is_best_loss:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if epoch % self.cfg.training.save_frequency == 0 or is_best:
                self.save_checkpoint(
                    epoch, val_metrics if self.val_loader else train_metrics, is_best
                )

            # Marks end of epoch
            print("-" * 50 + "\n")

            # Early stopping check
            if self.no_improvement_count >= patience:
                print(
                    f"\nEarly stopping triggered after {epoch + 1} epochs. No improvement in PR-AUC or Loss for {patience} consecutive epochs."
                )
                break

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        total_seconds = elapsed_time.total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training took {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Best Val PR-AUC: {self.best_metric:.3f}")
        if self.best_ckpt_path:
            best_val_thr = None
            best_epoch = None
            print(f"Loading best checkpoint: {self.best_ckpt_path}")
            ckpt = torch.load(
                self.best_ckpt_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "best_val_threshold" in ckpt:
                best_val_thr = ckpt["best_val_threshold"]
            if "epoch" in ckpt:
                best_epoch = ckpt["epoch"]

            thr = float(best_val_thr) if best_val_thr is not None else 0.5
            print(f"Using decision threshold {thr:.3f} for test metrics (Balances F1).")
            test_metrics = self.evaluate(best_val_thr, best_epoch)

            # Save metrics JSON
            metrics_path = self.metrics_dir / "test_metrics.json"
            out = {
                "checkpoint": str(self.best_ckpt_path),
                "threshold": thr,
                "metrics": test_metrics,
                "seed": int(self.rep_seed),
                # "target_recall": float(target_recall),
                # "threshold_at_target_recall": float(thr_tr) if thr_tr is not None else None,
                # "metrics_at_target_recall": metrics_tr,
            }
            with metrics_path.open("w") as f:
                json.dump(out, f, indent=2)
            print(f"\nSaved test metrics to {metrics_path}")
            mlflow.log_metrics({f"test/{k}": v for k, v in test_metrics.items()})
        print("Training completed!")
        return test_metrics

    # ----------------- Static (Will see) Methods -----------------

    @staticmethod
    def init_csv_logger(csv_path: Path):
        """
        Initialize a CSV file for per-epoch logs.
        """
        header = [
            "epoch",
            "train_loss",
            "train_acc",
            "train_f1",
            "train_precision",
            "train_recall",
            "train_roc_auc",
            "train_pr_auc",
            "val_loss",
            "val_acc",
            "val_roc_auc",
            "val_pr_auc",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_best_thr",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    @staticmethod
    def append_csv_log(csv_path: Path, row: dict):
        """
        Append one epoch's metrics to the CSV log.
        """
        fields = [
            "epoch",
            "train_loss",
            "train_acc",
            "train_f1",
            "train_precision",
            "train_recall",
            "train_roc_auc",
            "train_pr_auc",
            "val_loss",
            "val_acc",
            "val_roc_auc",
            "val_pr_auc",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_best_thr",
        ]
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row[k] for k in fields])

    @staticmethod
    def _get_weighted_sampler(
        subset: torch.utils.data.Subset, replacement: bool = True
    ):
        dataset = subset.dataset
        indices = subset.indices

        nclasses = dataset.num_classes

        # Count number of data per class
        all_labels = dataset.cached_labels
        subset_labels = torch.from_numpy(all_labels[indices])
        # cached_lables_tensor = torch.from_numpy(dataset.cached_labels)
        class_counts = torch.bincount(subset_labels, minlength=nclasses).float()
        # Compute weights: inverse of class frequency
        class_weights = 1.0 / class_counts
        # Assign weight to each sample
        sample_weights = class_weights[subset_labels]

        print(
            f"Weighted Sampling Class weights: {class_weights} | Replacement: {replacement}"
        )

        return torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=replacement,
        )

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, out_path: Path, class_names=None):
        if class_names is None:
            class_names = ["neg", "pos"]

        fig, ax = plt.subplots(figsize=(4, 4))

        # Use sklearn's ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            display_labels=class_names,
            cmap="Blues",
            ax=ax,
        )

        # Customize the plot
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Test Confusion Matrix")

        # Add text annotations with appropriate colors
        cm = disp.confusion_matrix
        thresh = cm.max() / 2.0
        # Assert for ruff warning
        assert disp.text_ is not None, (
            "ConfusionMatrixDisplay.text_ was not populated after plotting"
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                disp.text_[i, j].set_color("white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def get_tpr_fpr_from_threshold(threshold, y_true, y_scores):
        # Get FPR, TPR, and thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # Interpolate tpr and fpr values
        tpr_intrp = interpolate.interp1d(thresholds, tpr)
        fpr_intrp = interpolate.interp1d(thresholds, fpr)

        # Clip to avoid crushing
        threshold = np.clip(threshold, thresholds.min(), thresholds.max())

        return tpr_intrp(threshold), fpr_intrp(threshold)

    @staticmethod
    def get_pr_rec_from_threshold(threshold, y_true, y_scores):
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # Interpolate precision and recall values
        pr_intrp = interpolate.interp1d(thresholds, precision[:-1])
        rec_intrp = interpolate.interp1d(thresholds, recall[:-1])

        # Clip to avoid crushing
        threshold = np.clip(threshold, thresholds.min(), thresholds.max())

        return pr_intrp(threshold), rec_intrp(threshold)

    @staticmethod
    def compute_metrics(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        val: bool = False,
        tgt_recall: float = 0.7,
    ) -> Dict[str, float]:
        """Compute training metrics for binary classification."""
        labels_np = labels.cpu().numpy()

        if not val:
            # Convert logits to probabilities and predictions
            with torch.no_grad():
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                predictions = (probs >= 0.5).astype(int)
            # probs = torch.sigmoid(outputs).detach().cpu().numpy()
            # predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

            return {
                "accuracy": accuracy_score(labels_np, predictions),
                "precision": precision_score(labels_np, predictions, zero_division=0),
                "recall": recall_score(labels_np, predictions, zero_division=0),
                "f1": f1_score(labels_np, predictions, zero_division=0),
                "roc_auc": roc_auc_score(labels_np, probs),
                "pr_auc": average_precision_score(labels_np, probs),
            }
        else:
            probs = torch.sigmoid(outputs).cpu().numpy()

            precision, recall, thresholds = precision_recall_curve(labels_np, probs)
            # precision/recall have len(thresholds)+1 entries; the trailing
            # point (precision=1, recall=0) has no associated threshold. Align
            # F1 with `thresholds` by dropping that last, threshold-less point.
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_t = f1[:-1]  # F1 values aligned 1:1 with `thresholds`

            # Tie-break instead of a plain argmax. As the model overfits, the
            # predicted-prob distribution saturates toward 0/1 and the
            # F1-vs-threshold curve goes nearly flat near its peak, so the
            # unique argmax becomes noise-sensitive and teleports between epochs
            # (we saw it jump 0.06 <-> 0.97) even when the model barely changed.
            # `f1_tol` is a tolerance knob, NOT a float guard: among all
            # thresholds whose F1 is within `f1_tol` of the best, pick the one
            # closest to 0.5 -- i.e. give up at most `f1_tol` F1 to get the
            # least extreme, most defensible and stable operating point.
            f1_tol = 0.02
            near_best = np.flatnonzero(f1_t >= f1_t.max() - f1_tol)
            best_idx = near_best[np.argmin(np.abs(thresholds[near_best] - 0.5))]
            best_thresh = thresholds[best_idx]
            print(f"Optimal F1 Threshold: {best_thresh}")

            ###
            valid_mask = recall[:-1] >= tgt_recall  # exclude last point (no threshold)
            if valid_mask.any():
                # Highest-F1 threshold that still achieves recall >= target.
                # Use a separate index so it does not clobber `best_idx` (the
                # F1 operating point reported below).
                valid_f1 = np.where(valid_mask, f1_t, -1)
                tgt_idx = valid_f1.argmax()
                best_threshold_tgt_recall = thresholds[tgt_idx]
                print(
                    f"Optimal Threshold (recall >= {tgt_recall}): {best_threshold_tgt_recall}"
                )
            else:
                # Fallback: no threshold satisfies the constraint, reuse F1 thr
                print(
                    f"Warning: cannot achieve recall >= {tgt_recall}. Using max F1 threshold."
                )
                best_threshold_tgt_recall = best_thresh
            ####
            preds = (probs > best_thresh).astype(int)

            return {
                "best_threshold": best_thresh,
                "best_threshold_tgt_recall": best_threshold_tgt_recall,
                "f1": f1_t[best_idx],
                "precision": precision[best_idx],
                "recall": recall[best_idx],
                "accuracy": accuracy_score(labels_np, preds),
                "roc_auc": roc_auc_score(labels_np, probs),
                "pr_auc": average_precision_score(labels_np, probs),
            }

    # Debugging:
    @staticmethod
    def log_grad_norms(model, logger=None, prefix=""):
        norms = {}
        for name, module in model.named_children():
            total_norm = 0.0
            n_params = 0
            for p in module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    n_params += p.numel()
            total_norm = total_norm**0.5
            norms[name] = total_norm
            if logger:
                logger.info(f"{prefix}{name} grad_norm: {total_norm:.3e}")
            else:
                print(f"{prefix}{name} grad_norm: {total_norm:.3e}")
        return norms


class OmniRadTrainer(BaseTrainer):
    cfg: OmniRadConfig  # Fix class type annotarion for pylance
    model: Union[OmniRadProstateCsPCa, OmniRadProstateCsPCaClinical]

    def __init__(self, cfg: OmniRadConfig):
        self.use_clinical = cfg.model.use_clinical
        self.att_fusion = cfg.model.att_fusion
        self.use_dense_features = cfg.model.use_dense_features

        super().__init__(cfg)

        self.print_train_info()

        # model freeze backbone should be set to off if using this
        if cfg.training.finetune:
            # assert not cfg.model.freeze_backbone
            finetune_fn = FINETUNE_MAP[cfg.training.finetune_t]
            finetune_fn(self.model.encoder)
            count_trainable(self.model.encoder)
        else:
            finetune_fn = FINETUNE_MAP["v0"]
            finetune_fn(self.model.encoder)
            count_trainable(self.model.encoder)

    def print_train_info(self):
        # print(f"Modalities used: {self.modal}")
        print(f"Loss function pos_weight: {self.pos_weight}")
        print(f"Logging to: {self.timestamp}")
        print(f"Learning Rate: {self.cfg.training.learning_rate}")
        print(f"Loss Function: {self.cfg.training.loss_fn}")
        print(f"Optimizer: {self.cfg.training.optimizer}")
        print(f"Weight Decay: {self.cfg.training.weight_decay}")
        print(f"csPCa Threshold: {self.cfg.training.cspca_threshold}")
        print(f"Reproducibility seed: {self.rep_seed}")
        # self.model.print_model_architecture(self.device)
        model_stats = self.model.get_model_summary(
            input_shape=self.cfg.training.target_shape,
            clinical_shape=self.cfg.model.clin_input_dim,
        )
        print(model_stats)

    def _build_model(self):
        """Build model specific to the OmniRad trainer"""
        if self.use_clinical:
            self.model = OmniRadProstateCsPCaClinical(self.cfg.model).to(self.device)
        else:
            self.model = OmniRadProstateCsPCa(self.cfg.model).to(self.device)

    def _build_model_id(self) -> str:
        """Load model_id from OmniRad model config

        Raises:
            RuntimeError: Error when building the model id from config

        Returns:
            str: model id retrieved from model config
        """
        try:
            cfg = self.cfg

            model_name = cfg.model.model_name.split("/")[-1]
            loss = cfg.training.loss_fn

            finetuned = "finetuned" if cfg.training.finetune else "transfer"
            finetune_type = cfg.training.finetune_t if cfg.training.finetune else "v0"

            parts = [
                # "Debug",
                model_name,
                loss,
                finetuned,
                finetune_type,
                str(cfg.training.cspca_threshold),
            ]

            if self.use_clinical:
                parts.append("clinical")

            if self.use_clinical and self.att_fusion:
                parts.append("att")

            if not self.use_dense_features:
                parts.append("cls")

            self.model_id = "_".join(parts)
            return self.model_id

        except (AttributeError, KeyError, TypeError) as e:
            raise RuntimeError("Invalid configuration while building model_id") from e

    def _project_name(self) -> str:
        """Get project parent directory based on dataset(s) used

        Raises:
            ValueError: Raise error if invalid dataset is used

        Returns:
            str: Project parent directory for logging
        """
        dataset_map = {
            "picai": "OmniRad",
            "goc": "OmniRad-g",
            "picai+goc": "OmniRad-pg",
            "promis": "OmniRad-pr",
            "all": "OmniRad-all",
        }

        try:
            dataset = self.cfg.training.datasets
            project_name = dataset_map[dataset]
            return project_name+"-all"
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError("Invalid dataset configuration") from e

    def _get_optimizer_param_groups(self) -> List[Dict]:
        """Load optimizer parameter groups based on Trainer subclasses model architecture.

        Returns:
            List[dict]: List of dictionaries of parameter groups
        """
        param_groups = [
            {
                "params": self.model.encoder.parameters(),
                "lr": self.cfg.training.finetune_lr,
                "weight_decay": self.cfg.training.finetune_wd,
            },
            {"params": self.model.aggregator.parameters()},
            {"params": self.model.classifier.parameters()},
        ]

        # Add clinical component only if enabled
        if self.use_clinical:
            param_groups.append({"params": self.model.clinical_mlp.parameters()})

        if self.att_fusion:
            att_params = (
                list(self.model.cross_attention.parameters())
                + list(self.model.clinical_projection.parameters())
                + list(self.model.attention_norm.parameters())
            )
            param_groups.append({"params": att_params})

        return param_groups


class RadioDinoTrainer(OmniRadTrainer):
    def __init__(self, cfg: OmniRadConfig):
        super().__init__(cfg)

    def _project_name(self) -> str:
        """Get project parent directory based on dataset(s) used

        Raises:
            ValueError: Raise error if invalid dataset is used

        Returns:
            str: Project parent directory for logging
        """
        dataset_map = {
            "picai": "RadioDino",
            "goc": "RadioDino-g",
            "picai+goc": "RadioDino-pg",
            "promis": "RadioDino-pr",
            "all": "RadioDino-all",
        }

        try:
            dataset = self.cfg.training.datasets
            project_name = dataset_map[dataset]
            return project_name+"-all"
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError("Invalid dataset configuration") from e


class CNNTrainer(BaseTrainer):
    cfg: CNNConfig  # Fix class type annotarion for pylance
    model: Union[CNN3D, CNN3DClinical]

    def __init__(self, cfg: CNNConfig):
        self.use_clinical = cfg.model.use_clinical
        self.att_fusion = cfg.model.att_fusion

        super().__init__(cfg)

        self.print_train_info()

    def print_train_info(self):
        # print(f"Modalities used: {self.modal}")
        print(f"Loss function pos_weight: {self.pos_weight}")
        print(f"Logging to: {self.timestamp}")
        print(f"Learning Rate: {self.cfg.training.learning_rate}")
        print(f"Loss Function: {self.cfg.training.loss_fn}")
        print(f"Optimizer: {self.cfg.training.optimizer}")
        print(f"Weight Decay: {self.cfg.training.weight_decay}")
        print(f"csPCa Threshold: {self.cfg.training.cspca_threshold}")
        print(f"Reproducibility seed: {self.rep_seed}")
        # self.model.print_model_architecture(self.device)
        model_stats = self.model.get_model_summary(
            input_shape=self.cfg.training.target_shape,
            clinical_shape=self.cfg.model.clin_input_dim,
        )
        print(model_stats)

    def _build_model(self):
        """Build model specific to the 3D CNN trainer"""
        if self.use_clinical:
            self.model = CNN3DClinical(self.cfg.model).to(self.device)
        else:
            self.model = CNN3D(self.cfg.model).to(self.device)

    def _build_model_id(self) -> str:
        """Load model_id from 3D CNN model config

        Raises:
            RuntimeError: Error when building the model id from config

        Returns:
            str: model id retrieved from model config
        """
        try:
            cfg = self.cfg

            model_name = cfg.model.model_name.split("/")[-1]
            model_version = cfg.model.model_v
            loss = cfg.training.loss_fn

            parts = [
                model_name,
                model_version,
                loss,
                str(cfg.training.cspca_threshold),
            ]

            if self.use_clinical:
                parts.append("clinical")

            if self.use_clinical and self.att_fusion:
                parts.append("att")

            self.model_id = "_".join(parts)
            return self.model_id

        except (AttributeError, KeyError, TypeError) as e:
            raise RuntimeError("Invalid configuration while building model_id") from e

    def _project_name(self) -> str:
        """Get project parent directory based on dataset(s) used

        Raises:
            ValueError: Raise error if invalid dataset is used

        Returns:
            str: Project parent directory for logging
        """
        dataset_map = {
            "picai": "3DCNN",
            "goc": "3DCNN-g",
            "picai+goc": "3DCNN-pg",
            "promis": "3DCNN-pr",
            "all": "3DCNN-all",
        }

        try:
            dataset = self.cfg.training.datasets
            project_name = dataset_map[dataset]
            return project_name+"-all"
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError("Invalid dataset configuration") from e

    def _get_optimizer_param_groups(self) -> List[Dict]:
        """Load optimizer parameter groups based on Trainer subclasses model architecture.

        Returns:
            List[dict]: List of dictionaries of parameter groups
        """
        # TODO maybe change this later to not give all model params
        param_groups = [
            {"params": self.model.single_channel_block_t2w.parameters()},
            {"params": self.model.single_channel_block_adc.parameters()},
            {"params": self.model.single_channel_block_dwi.parameters()},
            {"params": self.model.features.parameters()},
            {"params": self.model.classifier.parameters()},
        ]

        # Add clinical component only if enabled
        if self.use_clinical:
            param_groups.append({"params": self.model.clinical_mlp.parameters()})

        if self.att_fusion:
            att_params = (
                list(self.model.cross_attention.parameters())
                + list(self.model.clinical_projection.parameters())
                + list(self.model.attention_norm.parameters())
            )
            param_groups.append({"params": att_params})

        return param_groups


class ResNetTrainer(BaseTrainer):
    # pass
    cfg: ResNetConfig  # Fix class type annotarion for pylance
    model: Union[ResNet3D, ResNet3DClinical]

    def __init__(self, cfg: ResNetConfig):
        self.use_clinical = cfg.model.use_clinical
        self.att_fusion = cfg.model.att_fusion

        super().__init__(cfg)

        self.print_train_info()

    def print_train_info(self):
        # print(f"Modalities used: {self.modal}")
        print(f"Loss function pos_weight: {self.pos_weight}")
        print(f"Logging to: {self.timestamp}")
        print(f"Learning Rate: {self.cfg.training.learning_rate}")
        print(f"Loss Function: {self.cfg.training.loss_fn}")
        print(f"Optimizer: {self.cfg.training.optimizer}")
        print(f"Weight Decay: {self.cfg.training.weight_decay}")
        print(f"csPCa Threshold: {self.cfg.training.cspca_threshold}")
        print(f"Reproducibility seed: {self.rep_seed}")
        # self.model.print_model_architecture(self.device)
        model_stats = self.model.get_model_summary(
            input_shape=self.cfg.training.target_shape,
            clinical_shape=self.cfg.model.clin_input_dim,
        )
        print(model_stats)

    def _build_model(self):
        """Build model specific to the 3D CNN trainer"""
        if self.use_clinical:
            self.model = ResNet3DClinical(self.cfg.model).to(self.device)
        else:
            self.model = ResNet3D(self.cfg.model).to(self.device)

    def _build_model_id(self) -> str:
        """Load model_id from 3D CNN model config

        Raises:
            RuntimeError: Error when building the model id from config

        Returns:
            str: model id retrieved from model config
        """
        try:
            cfg = self.cfg

            model_name = cfg.model.model_name.split("/")[-1]
            model_version = cfg.model.model_v
            loss = cfg.training.loss_fn

            parts = [
                model_name,
                model_version,
                loss,
                str(cfg.training.cspca_threshold),
            ]

            if self.use_clinical:
                parts.append("clinical")

            if self.use_clinical and self.att_fusion:
                parts.append("att")

            self.model_id = "_".join(parts)
            return self.model_id

        except (AttributeError, KeyError, TypeError) as e:
            raise RuntimeError("Invalid configuration while building model_id") from e

    def _project_name(self) -> str:
        """Get project parent directory based on dataset(s) used

        Raises:
            ValueError: Raise error if invalid dataset is used

        Returns:
            str: Project parent directory for logging
        """
        dataset_map = {
            "picai": "ResNet3D",
            "goc": "ResNet3D-g",
            "picai+goc": "ResNet3D-pg",
            "promis": "ResNet3D-pr",
            "all": "ResNet3D-all",
        }

        try:
            dataset = self.cfg.training.datasets
            project_name = dataset_map[dataset]
            return project_name+"-all"
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError("Invalid dataset configuration") from e

    def _get_optimizer_param_groups(self) -> List[Dict]:
        """Load optimizer parameter groups based on Trainer subclasses model architecture.

        Returns:
            List[dict]: List of dictionaries of parameter groups
        """
        # TODO maybe change this later to not give all model params
        param_groups = [
            {"params": self.model.t2w_block.parameters()},
            {"params": self.model.dwi_block.parameters()},
            {"params": self.model.adc_block.parameters()},
            {"params": self.model.features.parameters()},
            {"params": self.model.classifier.parameters()},
        ]

        # Add clinical component only if enabled
        if self.use_clinical:
            param_groups.append({"params": self.model.clinical_mlp.parameters()})

        if self.att_fusion:
            att_params = (
                list(self.model.cross_attention.parameters())
                + list(self.model.clinical_projection.parameters())
                + list(self.model.attention_norm.parameters())
            )
            param_groups.append({"params": att_params})

        return param_groups


TRAINER_REGISTRY = {
    "omnirad": OmniRadTrainer,
    "radiodino": RadioDinoTrainer,
    "cnn3d": CNNTrainer,
    "resnet3d": ResNetTrainer,
}
