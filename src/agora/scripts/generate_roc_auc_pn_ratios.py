import argparse
import json
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.utils.data.sampler
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

# from src.data.dataloader import CachedMRIDatasetCCTest
from agora.data.dataloader import get_datasets
from agora.scripts.train_v2 import get_config_class, get_trainer
from agora.training.base_config import BaseConfig

# --- Log-driven selection config (mode="logs") ---------------------------------
# The metrics dir mixes many configs/ablations and duplicated seeds, so scanning
# it blindly (mode="dir") lumps everything together. Instead we select runs from
# the training logs, which name the seed and print where the model was saved.
# Conventions mirror notebooks/stat_analysis.ipynb.
DEFAULT_LOGS_DIR = "results/logs"
# Seeds to select, in run order. The run number in the log name is the seed's
# 1-based position here (mri_train_multi.sub names run N = array index + 1), so
# switching seed sets (e.g. 42-46 -> 7-11) only requires editing this list.
SEEDS = [7, 8, 9, 10, 11]
RUN_OF_SEED = {seed: i + 1 for i, seed in enumerate(SEEDS)}

# Curve sets for the combined ROC graph. Each maps an output model name to
# (model_type, log-filename pattern with {s}=seed, {r}=run). One JSON is written
# per model: "{model_name}_{set_name}_roc.json", with keys seed_{seed}.
CURVE_SETS = {
    "thr7": {
        "3DCNN-pg": ("cnn3d", "cnn_v0_seed{s}_run{r}"),
        "ResNet3D-pg": ("resnet3d", "resnet_v0_seed{s}_run{r}"),
        "RadioDino-pg": ("radiodino", "radiodino_v5_flat_seed{s}_run{r}"),
        "OmniRad-pg": ("omnirad", "omnirad_v5_flat_seed{s}_run{r}"),
    },
    "thr6": {
        "3DCNN-pg": ("cnn3d", "cnn_v0_thr6_seed{s}_run{r}"),
        "ResNet3D-pg": ("resnet3d", "resnet_v0_thr6_seed{s}_run{r}"),
        "RadioDino-pg": ("radiodino", "radiodino_v5_thr6_seed{s}_run{r}"),
        "OmniRad-pg": ("omnirad", "omnirad_v5_thr6_seed{s}_run{r}"),
    },
}

_SAVED_RE = re.compile(r"Saved test metrics to (\S+test_metrics\.json)")

# MLflow run-summary line for a full-data (Fraction_1.00) learning-curve repeat.
# In an lr-curve log the 'Saved test metrics to ...' line for a run is printed
# immediately *before* its 'View run Fraction_1.00_repeat_N' line, so each
# Fraction_1.00 view line pairs with the most recent Saved line above it. The two
# lines' timestamps differ by ~1s, so order-based pairing (not timestamp joins)
# is what's reliable here.
_VIEW_FRAC1_RE = re.compile(r"View run Fraction_1\.00_repeat_(\d+)_")

# Known model types across all curve sets, for validating --lr-logs specs.
_KNOWN_MODEL_TYPES = {mt for curves in CURVE_SETS.values() for mt, _ in curves.values()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["dir", "logs"],
        default="dir",
        help="dir: scan a metrics base dir (legacy). logs: select runs from "
        "training logs by config/seed (recommended for the *-pg-all results).",
    )
    p.add_argument(
        "--model_base_dir",
        "-mbd",
        type=str,
        default=str(Path.cwd()),
        help="[dir mode] Model base path directory.",
    )
    p.add_argument(
        "--model-type",
        type=str,
        nargs="+",
        default=None,
        help="dir mode: model type used for testing (required; single value). "
        "logs mode: optional filter to emit only curves of these model types "
        "(e.g. cnn3d resnet3d); omit to emit all models.",
    )
    p.add_argument(
        "--logs-dir",
        type=str,
        default=DEFAULT_LOGS_DIR,
        help="[logs mode] Directory containing the training .log files.",
    )
    p.add_argument(
        "--curve-set",
        choices=[*CURVE_SETS.keys(), "all", "none"],
        default="all",
        help="[logs mode] Which seed-patterned curve set(s) to emit "
        "('none' skips them, e.g. when only --lr-logs is wanted).",
    )
    p.add_argument(
        "--lr-logs",
        nargs="+",
        type=str,
        default=None,
        help="[logs mode] One or more lr-curve logs to emit ROC curves for via "
        "their Fraction_1.00 (full-data) point. Each entry is "
        "'MODEL_NAME:MODEL_TYPE=PATH' (e.g. 'OmniRad-pg:omnirad=/…/omnirad_lr_curves_v3.log'). "
        "Written as {MODEL_NAME}_{--lr-set-name}_roc.json.",
    )
    p.add_argument(
        "--lr-set-name",
        type=str,
        default="thr7",
        help="[logs mode] Curve-set name used in the --lr-logs output filename "
        "(default: thr7, so it replaces the old seed-42 thr7 curves).",
    )
    p.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=str(Path.cwd()),
        help="JSON output directory.",
    )
    args = p.parse_args()
    if args.mode == "dir":
        if not args.model_type:
            p.error("--model-type is required in dir mode")
        if len(args.model_type) != 1:
            p.error("--model-type takes a single value in dir mode")
    if args.mode == "logs" and args.model_type:
        unknown = sorted(set(args.model_type) - _KNOWN_MODEL_TYPES)
        if unknown:
            p.error(
                f"--model-type {unknown} not in any curve set; "
                f"choose from {sorted(_KNOWN_MODEL_TYPES)} or omit to emit all."
            )
    if args.mode == "dir" and args.lr_logs:
        p.error("--lr-logs is only valid in logs mode")
    if args.lr_logs:
        for spec in args.lr_logs:
            name_part, sep, _ = spec.partition("=")
            model_type = name_part.partition(":")[2]
            if not sep or ":" not in name_part or not model_type:
                p.error(
                    f"--lr-logs entry must be 'MODEL_NAME:MODEL_TYPE=PATH': {spec!r}"
                )
            if model_type not in _KNOWN_MODEL_TYPES:
                p.error(
                    f"--lr-logs model type {model_type!r} unknown; "
                    f"choose from {sorted(_KNOWN_MODEL_TYPES)}"
                )
    return args


def metrics_dir_from_log(logs_dir: Path, log_name: str) -> Path:
    """Resolve a (group, seed) log to its metrics run directory.

    Parses the 'Saved test metrics to .../test_metrics.json' line printed at the
    end of training and returns its parent (the per-run metrics directory that
    get_tpr_fpr expects).
    """
    log_path = logs_dir / f"{log_name}.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        m = _SAVED_RE.search(fh.read())
    if not m:
        raise ValueError(f"No 'Saved test metrics to' line in {log_path}")
    return Path(m.group(1)).parent


def discover_lr_frac1_runs(lr_log: Path):
    """Return the Fraction_1.00 (full-data) repeats of one lr-curve log.

    An lr-curve log bundles every fraction × repeat in a single file, so the
    per-seed `metrics_dir_from_log` (first Saved line wins) can't isolate them.
    Here we walk the file top-to-bottom, remembering the most recent
    'Saved test metrics to ...' dir, and whenever a 'View run
    Fraction_1.00_repeat_N' line appears we pair that repeat with the remembered
    dir (see _VIEW_FRAC1_RE for why order, not timestamps, is the reliable join).

    Returns a list of (repeat, run_dir) sorted by repeat.
    """
    pairs = []
    last_saved = None
    with open(lr_log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m_saved = _SAVED_RE.search(line)
            if m_saved:
                last_saved = Path(m_saved.group(1)).parent
                continue
            m_view = _VIEW_FRAC1_RE.search(line)
            if m_view and last_saved is not None:
                pairs.append((int(m_view.group(1)), last_saved))
    return sorted(pairs)


def _seed_from_run_dir(run_dir: Path, fallback: int) -> int:
    """Read the seed recorded in a run's test_metrics.json (repeat index fallback)."""
    metrics_path = run_dir / "test_metrics.json"
    try:
        with metrics_path.open() as f:
            return int(json.load(f).get("seed", fallback))
    except (OSError, ValueError, json.JSONDecodeError):
        return fallback


def get_pred_and_prob(model, loader, device, use_clinical=False):
    model.eval()
    y_true = []
    y_pred_prob = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device)

            # Best threshold is derived from the latest validation set
            if use_clinical:
                clinical = batch["clinical"].to(device)
                outputs = model(images, clinical).squeeze()
            else:
                outputs = model(images).squeeze()
            y_pred_prob.extend(torch.sigmoid(outputs).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    return y_pred_prob, y_true


def validate_args(args):
    mrp = Path(args.model_base_dir)
    out_dir = Path(args.out_dir)

    if not mrp.is_dir():
        raise ValueError(f"Wrong path: {mrp} is not a directory.")
    if not mrp.is_dir():
        raise ValueError(f"Wrong path: {out_dir} is not a directory.")


def extract_model_id(model_root_path):
    parts = model_root_path.name.split("_")

    # Check if its the old or new naming format
    # A date segment matches YYYY-MM-DD, a uuid hex does not
    if re.match(r"^\d{4}-\d{2}-\d{2}$", parts[0]):
        # Old format: date_time_modelid — model_id starts at part 3
        model_id = "_".join(parts[2:])
    else:
        # New format: uuid_date_time_modelid — model_id starts at part 4
        model_id = "_".join(parts[3:])

    return model_id


# This is a static method in BaseTrainer so maybe theres a way to not duplicate
def _get_weighted_sampler(subset: torch.utils.data.Subset, replacement: bool = True):
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


def make_dataloaders(cfg: BaseConfig):
    """Setup datasets and dataloaders."""

    train_dataset, val_dataset, test_dataset, pos_weight = get_datasets(
        cfg, cfg.training.seed, use_clinical=cfg.model.use_clinical
    )

    sampler = None
    shuffle = True

    # Add weighted sampling (Undersample non csPCa)
    if cfg.training.weighted_sampling:
        sampler = _get_weighted_sampler(train_dataset)
        shuffle = False

    g = torch.Generator()
    g.manual_seed(cfg.training.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        sampler=sampler,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        generator=g,
    )

    return train_loader, val_loader, test_loader


def get_tpr_fpr(
    model_root_path, model_type
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    mrp = Path(model_root_path)
    model_dir = mrp.parent.stem
    pp_dir = mrp.parent.parent.parent
    ckpt_dir = pp_dir / "checkpoints" / model_dir / mrp.name

    model_id = extract_model_id(mrp)
    print(f"Using Model ID: {model_id}")

    # Optional checks
    parts = mrp.name.split("_")
    # Only used when we want to skip a specific threshold
    # if "7.0" in parts:
    #     print(f"Directory {mrp.name}: Skipped because threshold is 7.0")
    #     return None, None

    if "Debug" in mrp.name:
        print(f"Directory {mrp.name}: contains 'Debug' and is skipped.")
        return None, None

    # Load params JSON saved during training
    params_path = mrp / "training_params.json"
    if not params_path.exists():
        print(f"FileNotFoundError: Params JSON not found: {params_path}")
        return None, None
        # raise FileNotFoundError(f"Params JSON not found: {params_path}")

    CfgClass = get_config_class(model_type)
    config = CfgClass.from_json(params_path)

    # Load best checkpoint
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path:
        raise FileNotFoundError(f"No best_model.pt found in {ckpt_dir}")
    print(f"Using checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_val_thr = ckpt.get("best_val_threshold", 0.5)
    print(f"Using decision threshold {best_val_thr:.3f} for test metrics.")

    _, _, test_loader = make_dataloaders(config)

    # Build model
    TrainerClass = get_trainer(model_type)
    trainer = TrainerClass.build_model_only(config)
    model = trainer.model

    state_dict = ckpt["model_state_dict"]

    # Check if old naming exists -> Past model compatibility
    has_spatial = any(k.startswith("spatial_aggregator") for k in state_dict.keys())

    if has_spatial:
        print(
            "Old checkpoint format detected. Renaming spatial_aggregator → aggregator"
        )

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("spatial_aggregator", "aggregator")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    y_pred_prob, y_true = get_pred_and_prob(
        model,
        test_loader,
        device,
        use_clinical=config.model.use_clinical,
    )

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    return fpr, tpr


def run_dir_mode(args):
    """Legacy mode: treat every subdir of model_base_dir as one seed of one model."""
    validate_args(args)
    mbr = Path(args.model_base_dir)

    # Dictionary to save tpr and fpr for all seeds of a particular model
    model_data = {}

    for idx, subdir in enumerate(sorted(mbr.iterdir())):
        if subdir.is_dir():
            print("----------------------------------------")
            print(f"Calculating TPR and FPR for {subdir}")
            print("----------------------------------------")
            fpr, tpr = get_tpr_fpr(subdir, args.model_type[0])
            if tpr is not None and fpr is not None:
                model_data[f"seed_{idx}"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": auc(fpr, tpr),
                }

    model_dir = mbr.stem
    # When all is done we save a json per model
    out_path = f"{args.out_dir}/{model_dir}_roc.json"
    with open(out_path, "w") as f:
        json.dump(model_data, f)

    print(f"\nSaved ROC curve to {out_path}")


def run_logs_mode(args):
    """Select runs from the training logs by config/seed and emit one JSON per model."""
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    if args.curve_set == "all":
        set_names = list(CURVE_SETS)
    elif args.curve_set == "none":
        set_names = []
    else:
        set_names = [args.curve_set]

    for set_name in set_names:
        for model_name, (model_type, pattern) in CURVE_SETS[set_name].items():
            # Optional filter: only emit curves for the requested model type(s).
            if args.model_type and model_type not in args.model_type:
                continue

            print("=" * 60)
            print(f"[{set_name}] {model_name} ({model_type})")
            print("=" * 60)

            model_data = {}
            for seed in SEEDS:
                log_name = pattern.format(s=seed, r=RUN_OF_SEED[seed])
                try:
                    run_dir = metrics_dir_from_log(logs_dir, log_name)
                except (FileNotFoundError, ValueError) as e:
                    print(f"  seed {seed}: SKIP ({e})")
                    continue

                print(f"  seed {seed}: {run_dir}")
                fpr, tpr = get_tpr_fpr(run_dir, model_type)
                if fpr is None or tpr is None:
                    print(f"  seed {seed}: SKIP (no tpr/fpr returned)")
                    continue

                model_data[f"seed_{seed}"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": auc(fpr, tpr),
                }

            out_path = out_dir / f"{model_name}_{set_name}_s7-11_roc.json"
            with open(out_path, "w") as f:
                json.dump(model_data, f)
            print(f"Saved {len(model_data)}/{len(SEEDS)} seeds to {out_path}")


def run_lr_logs_mode(args):
    """Emit ROC JSONs from the Fraction_1.00 point of explicitly listed lr-curve logs.

    Each --lr-logs entry is 'MODEL_NAME:MODEL_TYPE=PATH'. Output is written as
    {MODEL_NAME}_{--lr-set-name}_roc.json with keys seed_{seed}, mirroring
    run_logs_mode so the combined-ROC notebook consumes it unchanged.
    """
    out_dir = Path(args.out_dir)
    set_name = args.lr_set_name

    for spec in args.lr_logs:
        name_part, _, path = spec.partition("=")
        model_name, _, model_type = name_part.partition(":")

        # Optional filter: only emit curves for the requested model type(s).
        if args.model_type and model_type not in args.model_type:
            continue

        lr_log = Path(path)
        if not lr_log.exists():
            print(f"⚠️  lr-curve log not found: {lr_log}")
            continue

        print("=" * 60)
        print(f"[{set_name}] {model_name} ({model_type})  <- {lr_log.name}")
        print("=" * 60)

        pairs = discover_lr_frac1_runs(lr_log)
        if not pairs:
            print(f"  ⚠️  No Fraction_1.00 runs found in {lr_log.name}")
            continue

        model_data = {}
        for repeat, run_dir in pairs:
            seed = _seed_from_run_dir(run_dir, fallback=repeat)
            print(f"  repeat {repeat} (seed {seed}): {run_dir}")
            fpr, tpr = get_tpr_fpr(run_dir, model_type)
            if fpr is None or tpr is None:
                print(f"  seed {seed}: SKIP (no tpr/fpr returned)")
                continue

            model_data[f"seed_{seed}"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": auc(fpr, tpr),
            }

        out_path = out_dir / f"{model_name}_{set_name}_roc.json"
        with open(out_path, "w") as f:
            json.dump(model_data, f)
        print(f"Saved {len(model_data)}/{len(pairs)} seeds to {out_path}")


def main():
    args = parse_args()

    if args.mode == "logs":
        run_logs_mode(args)
        if args.lr_logs:
            run_lr_logs_mode(args)
    else:
        run_dir_mode(args)

    print("\n" + "=" * 80)
    print("✅ Calculations complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
