import argparse
import datetime
import sys
import traceback

import mlflow
import pandas as pd
import yaml
from tqdm import tqdm

from agora.training.base_config import CONFIG_REGISTRY
from agora.training.base_trainer import TRAINER_REGISTRY
from agora.training.learning_curve_trainer import LearningCurveTrainer
from agora.utils.training import flatten_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run learning curve experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model-type", type=str, required=True, help="Model type used for training"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", type=bool, default=False)
    parser.add_argument(
        "--repeat-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Specific repeat indices to run (default: all 0..repeats-1). "
            "Each repeat uses seed = base_seed + repeat_id, so passing a subset "
            "reproduces exactly the same splits as a full run. Use e.g. "
            "'--repeat-ids 4' to resume only the final repeat."
        ),
    )
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
    if model_type not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(TRAINER_REGISTRY)}"
        )
    return TRAINER_REGISTRY[model_type]


def get_config_class(model_type: str):
    if model_type not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(CONFIG_REGISTRY)}"
        )
    return CONFIG_REGISTRY[model_type]


def verify_learning_curve_setup(
    trainer: LearningCurveTrainer,
    fractions: list[float],
    repeat_ids: list[int],
) -> None:
    print("\n=== Dry Run: Learning Curve Setup Verification ===\n")
    all_checks_passed = True
    base_seed = trainer.rep_seed

    # Track the previous repeat's largest subset to verify nesting within a repeat
    for j in repeat_ids:
        seed_j = base_seed + j

        # Re-split train/val/test for this repeat, exactly as the real loop does
        trainer.resplit_data(seed=seed_j)

        # Reference class ratio from THIS repeat's full training pool
        full_labels = trainer._full_train_dataset.dataset.cached_labels[
            trainer._full_train_dataset.indices
        ]
        full_n_pos = int((full_labels == 1).sum())
        full_n_neg = int((full_labels == 0).sum())
        full_ratio = full_n_pos / len(full_labels)
        print(
            f"Repeat {j} | seed {seed_j} | full train set | n={len(full_labels)} | "
            f"neg={full_n_neg} pos={full_n_pos} | pos_ratio={full_ratio:.3f}"
        )

        prev_indices = None  # smaller fraction's indices, to check nesting
        for fraction in fractions:
            checks = []  # collect per-run issues

            # 1. Update fraction and inspect subset
            subset = trainer.update_fraction(fraction=fraction, seed=seed_j)
            n_total = len(subset)
            # subset_labels = trainer._get_subset_labels(subset)
            n_pos = int((subset.labels == 1).sum())
            n_neg = int((subset.labels == 0).sum())
            ratio = n_pos / n_total if n_total > 0 else 0.0

            # Check both classes are present
            if n_pos == 0:
                checks.append("WARN: no positive samples")
                all_checks_passed = False
            if n_neg == 0:
                checks.append("WARN: no negative samples")
                all_checks_passed = False

            # Check ratio is within 5% of the full dataset ratio
            if abs(ratio - full_ratio) > 0.05:
                checks.append(
                    f"WARN: pos_ratio {ratio:.3f} deviates from full ratio {full_ratio:.3f}"
                )
                all_checks_passed = False

            # Check nesting: the previous (smaller) fraction must be a subset of
            # this one within the same repeat.
            if prev_indices is not None:
                if not set(prev_indices.tolist()).issubset(set(subset.indices.tolist())):
                    checks.append(
                        "WARN: subset is NOT nested within the larger fraction"
                    )
                    all_checks_passed = False
            prev_indices = subset.indices

            # 2. Reset loader and pull one batch
            trainer.reset_train_loader(subset)
            try:
                batch = next(iter(trainer.train_loader))
                images = batch["image"].to(trainer.device)
                labels = batch["label"].float().to(trainer.device)
                true_gss = batch["true_gs"]
                paths_batch = batch["paths"]

                img_shape = tuple(images.shape)
                lbl_shape = tuple(labels.shape)
                n_true_gs = len(true_gss)
                n_paths = len(paths_batch)

                clinical_shape = None
                if trainer.use_clinical:
                    clinical = batch["clinical"].to(trainer.device)
                    clinical_shape = tuple(clinical.shape)

            except Exception as e:
                checks.append(f"ERROR: batch loading failed — {e}")
                all_checks_passed = False
                img_shape = lbl_shape = None
                clinical_shape = None
                n_true_gs = n_paths = None

            # 3. Print summary row
            status = "✓" if not checks else "✗"
            clinical_str = f" clinical={clinical_shape}" if clinical_shape else ""
            print(
                f"{status} Fraction {fraction:.2f} | repeat {j} | seed {seed_j} | "
                f"n={n_total} | neg={n_neg} pos={n_pos} | pos_ratio={ratio:.3f} | "
                f"image={img_shape} label={lbl_shape}{clinical_str} | "
                f"n_true_gs={n_true_gs} n_paths={n_paths}"
            )

            # 4. Print any warnings/errors
            for issue in checks:
                print(f"    └─ {issue}")

        print()  # blank line between repeats for readability

    if all_checks_passed:
        print("\n✓ All checks passed. Ready to run full experiment.")
    else:
        print("\n✗ Some checks failed. Review warnings above before running.")


def main():
    args = parse_args()
    try:
        CfgClass = get_config_class(args.model_type)
        cfg = CfgClass.from_yaml(args.config)

        if args.seed is not None:
            cfg.training.seed = args.seed

        base_seed = cfg.training.seed

        # --- Wrap the registered trainer with LearningCurveTrainer ---
        # LearningCurveTrainer expects to be used as a mixin-style subclass.
        # We dynamically create a combined class so registry-defined trainers
        # (which implement the abstract methods) are still used, while
        # LearningCurveTrainer adds update_fraction / reset_run on top.
        BaseTrainerClass = get_trainer(args.model_type)
        DynamicLCTrainer = type(
            "DynamicLCTrainer",
            (
                LearningCurveTrainer,
                BaseTrainerClass,
            ),  # LC methods take priority via MRO
            {},
        )

        fractions = [0.1, 0.33, 0.55, 0.78, 1.0]
        repeats = 5
        repeat_ids = args.repeat_ids if args.repeat_ids is not None else list(range(repeats))

        if args.dry_run:
            trainer = DynamicLCTrainer.build_for_dry_run(cfg)
            verify_learning_curve_setup(trainer, fractions, repeat_ids)
            return

        trainer = DynamicLCTrainer(cfg)

        mlflow.set_experiment(f"Agora/{args.model_type}/lr_curves_v4")

        # Output path is computed up front so results can be written
        # incrementally (see below). metrics_dir.parent.parent is the stable
        # per-project metrics dir (metrics_dir itself changes each run). When
        # only a subset of repeats is requested, suffix the filename so a
        # resume run never clobbers the canonical full-run CSV.
        suffix = "" if args.repeat_ids is None else "_repeats-" + "-".join(
            str(r) for r in repeat_ids
        )
        out_path = (
            trainer.metrics_dir.parent.parent
            / f"learning_curve_results_{args.model_type}_v4{suffix}.csv"
        )
        first_write = True

        # Repeat/seed is the OUTER loop: each repeat re-splits train/val/test
        # with a fresh seed (resplit_data), so cross-repeat variance reflects
        # the data split, not just model init. Fractions are the INNER loop and
        # subsample nested prefixes from this repeat's fixed training pool, all
        # evaluated on this repeat's fixed test set.
        for j in tqdm(repeat_ids, desc="Repeat"):
            seed_j = base_seed + j
            trainer.resplit_data(seed=seed_j)

            for fraction in fractions:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_name = f"Fraction_{fraction:.2f}_repeat_{j}_{timestamp}"

                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params(flatten_config(cfg.training, prefix="training"))
                    mlflow.log_params(flatten_config(cfg.model, prefix="model"))
                    mlflow.log_params({"fraction": fraction, "repeat": j, "seed": seed_j})

                    subset = trainer.update_fraction(fraction=fraction, seed=seed_j)
                    trainer.reset_run(seed=seed_j)
                    trainer.reset_train_loader(subset)

                    test_metrics = trainer.train()

                if test_metrics is not None:
                    row = {
                        "fraction": fraction,
                        "repeat": j,
                        "seed": seed_j,
                        **test_metrics,
                    }
                    # Append each completed run to disk immediately so a mid-run
                    # failure (e.g. MLflow server 503s) never loses finished
                    # rows. The first successful write of this invocation
                    # (over)writes the file with a header; the rest append.
                    pd.DataFrame([row]).to_csv(
                        out_path,
                        mode="w" if first_write else "a",
                        header=first_write,
                        index=False,
                    )
                    first_write = False

        print(f"Saved learning curve results to {out_path}")

    except Exception as e:
        print(f"Error during training: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
