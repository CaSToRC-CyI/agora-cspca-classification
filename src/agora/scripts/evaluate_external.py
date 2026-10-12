#!/usr/bin/env python3
"""Evaluate a trained csPCa model on an external dataset.

Loads a finished training run (config + best checkpoint), builds a single
no-split loader over an external cohort described by an external-eval YAML, runs
inference and writes results to ``--out-dir``:

    metrics.json        (only when the cohort is labeled)
    predictions.csv     (study paths, probability, label-if-present)
    roc_curve.png / pr_curve.png / confusion_matrix.png   (labeled only)

The decision threshold is the validation-selected ``best_val_threshold`` stored
in the checkpoint. Clinical features are normalized with the SOURCE model's
train-split stats (also from the checkpoint); see ``--clinical-mode``.

Example:
    uv run python src/agora/scripts/evaluate_external.py \\
        --run-dir /path/to/metrics/Agora-omnirad/<uuid>_<ts>_<modelid> \\
        --model-type omnirad \\
        --external-config configs/external_eval_example.yaml \\
        --out-dir /path/to/external_results
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from agora.data.dataloader import get_datasets, get_external_dataset
from agora.scripts.train_v2 import get_config_class, get_trainer
from agora.training.base_config import ExternalEvalConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        required=True,
        help="Metrics run dir of the trained model "
        "(contains training_params.json).",
    )
    p.add_argument("--model-type", required=True, help="omnirad|radiodino|cnn3d|resnet3d")
    p.add_argument(
        "--external-config",
        required=True,
        help="YAML with an 'external_eval' block.",
    )
    p.add_argument("--out-dir", required=True, help="Directory for results.")
    p.add_argument(
        "--clinical-mode",
        choices=["impute", "require"],
        default="impute",
        help="impute: mean-fill missing clinical with training stats; "
        "require: fail if any clinical value is missing.",
    )
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild external cache.")
    return p.parse_args()


def resolve_checkpoint(run_dir: Path) -> Path:
    """Map a metrics run dir to its best_model.pt under checkpoints/."""
    model_dir = run_dir.parent.stem          # project dir, e.g. Agora-omnirad
    output_dir = run_dir.parent.parent.parent  # training output_dir
    ckpt_path = output_dir / "checkpoints" / model_dir / run_dir.name / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best_model.pt not found at {ckpt_path}")
    return ckpt_path


def load_state_dict_compat(model, state_dict):
    """Handle the legacy spatial_aggregator -> aggregator rename."""
    if any(k.startswith("spatial_aggregator") for k in state_dict):
        print("Old checkpoint format: renaming spatial_aggregator -> aggregator")
        state_dict = {
            k.replace("spatial_aggregator", "aggregator"): v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)


def recover_clinical_stats(config):
    """Fallback for old checkpoints: recompute train-split clinical stats."""
    print(
        "WARNING: checkpoint has no 'clinical_stats'. Recomputing from the original "
        "training data (rebuilds/loads the training cache)."
    )
    train_dataset, _, _, _ = get_datasets(config, config.training.seed, use_clinical=True)
    return getattr(train_dataset.dataset, "clinical_stats", None)


@torch.no_grad()
def run_inference(model, loader, device, use_clinical):
    model.eval()
    probs, labels, paths = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        if use_clinical:
            outputs = model(images, batch["clinical"].to(device)).squeeze()
        else:
            outputs = model(images).squeeze()
        outputs = torch.atleast_1d(outputs)
        probs.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
        paths.extend(batch["paths"][0])  # T2 path per case
    return np.asarray(probs), np.asarray(labels), paths


def compute_and_plot_metrics(y_true, y_prob, threshold, out_dir):
    y_pred = (y_prob > threshold).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

    RocCurveDisplay.from_predictions(y_true, y_prob, name="csPCa", plot_chance_level=True)
    plt.savefig(out_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob, name="csPCa")
    plt.savefig(out_dir / "pr_curve.png", bbox_inches="tight")
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["non-csPCa", "csPCa"]
    )
    plt.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    return metrics


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load trained run config + checkpoint ---
    params_path = run_dir / "training_params.json"
    print(params_path)
    if not params_path.exists():
        raise FileNotFoundError(f"training_params.json not found in {run_dir}")
    config = get_config_class(args.model_type).from_json(params_path)
    use_clinical = config.model.use_clinical

    ckpt_path = resolve_checkpoint(run_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    threshold = ckpt.get("best_val_threshold", 0.5)
    print(f"Decision threshold (from validation): {threshold:.4f}")

    clinical_stats = ckpt.get("clinical_stats")
    if use_clinical and clinical_stats is None:
        clinical_stats = recover_clinical_stats(config)

    # --- External cohort ---
    ext_cfg = ExternalEvalConfig.from_yaml(args.external_config)
    if tuple(ext_cfg.target_shape) != tuple(config.training.target_shape):
        raise ValueError(
            f"target_shape mismatch: external {ext_cfg.target_shape} != "
            f"training {config.training.target_shape}. They must match the model input."
        )
    print(f"Evaluating external dataset '{ext_cfg.name}' (labeled={ext_cfg.labeled})")

    _, loader = get_external_dataset(
        ext_cfg,
        use_clinical=use_clinical,
        clinical_stats=clinical_stats,
        clinical_mode=args.clinical_mode,
        force_rebuild=args.force_rebuild,
    )

    # --- Build model + load weights ---
    model = get_trainer(args.model_type).build_model_only(config).model
    load_state_dict_compat(model, ckpt["model_state_dict"])
    model.to(device)

    # --- Inference ---
    y_prob, y_true, paths = run_inference(model, loader, device, use_clinical)

    # --- Predictions CSV (always) ---
    pred_path = out_dir / "predictions.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t2_path", "probability"]
        if ext_cfg.labeled:
            header += ["label", "predicted"]
        writer.writerow(header)
        for i, prob in enumerate(y_prob):
            row = [paths[i], f"{prob:.6f}"]
            if ext_cfg.labeled:
                row += [int(y_true[i]), int(prob > threshold)]
            writer.writerow(row)
    print(f"Wrote {pred_path}")

    # --- Metrics (labeled only) ---
    if ext_cfg.labeled:
        metrics = compute_and_plot_metrics(y_true, y_prob, threshold, out_dir)
        metrics["dataset"] = ext_cfg.name
        metrics["clinical_mode"] = args.clinical_mode if use_clinical else "n/a"
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nExternal metrics:")
        print(json.dumps(metrics, indent=2))
    else:
        print("Unlabeled cohort: predictions written, metrics skipped.")


if __name__ == "__main__":
    main()
