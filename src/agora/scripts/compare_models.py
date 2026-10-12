import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: only writes the PNG to disk, no interactive window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

# Get current filename
this_file = Path(__file__).resolve()
log_dir = os.path.join(os.getcwd(), "log")

log_filepath = "{}/{}.log".format(log_dir, this_file.stem)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(name=f"{__name__}_CompareModels")

# --- Log-driven discovery (mode="logs") ---------------------------------------
# Training runs are no longer laid out under a predictable
# {ModelType}{datasetcode}/ metrics tree. Instead we discover runs from the
# per-run training logs in training_results/logs, which are named
#   {group}_seed{SEED}_run{RUN}.log
# and which print the absolute path of the saved test_metrics.json. We group the
# 5 seeds of each ablation by the {group} prefix and aggregate across them.
DEFAULT_LOGS_DIR = "results/logs"

# {group}_seed{seed}_run{run}.log  -> capture group / seed / run.
# Anything with an extra suffix (e.g. ..._run5_b) or no seed/run (e.g.
# *_split_thr7, *_lr_curves) is intentionally not matched.
_LOGNAME_RE = re.compile(r"^(?P<group>.+)_seed(?P<seed>\d+)_run(?P<run>\d+)$")

# Line printed at the end of training pointing at the per-run metrics dir.
_SAVED_RE = re.compile(r"Saved test metrics to (\S+test_metrics\.json)")

# MLflow run-summary line for a full-data (Fraction_1.00) learning-curve repeat.
# In an lr-curve log the 'Saved test metrics to ...' line for a run is printed
# immediately *before* its 'View run Fraction_1.00_repeat_N' line, so we pair
# each Fraction_1.00 view line with the most recent Saved line above it. The two
# lines' timestamps differ by ~1s, so order-based pairing (not timestamp joins)
# is what's reliable here.
_VIEW_FRAC1_RE = re.compile(r"View run Fraction_1\.00_repeat_(\d+)_")

METRIC_COLS = ["auroc", "pr_auc", "accuracy", "f1", "precision", "recall"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare model performances. Two modes: 'logs' discovers "
        "runs from the per-seed training logs (recommended); 'dir' scans a "
        "{ModelType}{dataset}/ metrics tree (legacy)."
    )
    parser.add_argument(
        "--mode",
        choices=["logs", "dir"],
        default="logs",
        help="logs: aggregate across the per-seed training logs (recommended). "
        "dir: scan a legacy metrics base dir by model-type/dataset.",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the ranked panel plot.",
    )

    # logs mode
    parser.add_argument(
        "--logs-dir",
        "-l",
        type=str,
        default=DEFAULT_LOGS_DIR,
        help="[logs mode] Directory containing the per-run training .log files.",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="[logs mode] Optional regex; keep only groups matching it.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="[logs mode] Optional regex; drop groups matching it.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="[logs mode] Keep only these seeds from the per-run logs "
        "(the group name is seed-agnostic, so this is the only way to split "
        "e.g. thr6 seed 42-46 from seed 7-11). Does not affect --lr-logs.",
    )
    parser.add_argument(
        "--lr-logs",
        nargs="+",
        type=str,
        default=None,
        help="[logs mode] One or more lr-curve logs to fold in via their "
        "Fraction_1.00 (full-data) point. Each entry is 'path' or 'label=path' "
        "(label becomes the table row). Bypasses --include/--exclude.",
    )

    # dir mode (legacy)
    parser.add_argument(
        "--model-types",
        "-m",
        nargs="+",
        type=str,
        default=None,
        help="[dir mode] Model types to compare (omnirad|resnet|cnn|radiodino).",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        type=str,
        default=None,
        help="[dir mode] Dataset/s to compare (picai|goc|picai+goc|promis|all).",
    )
    parser.add_argument(
        "--metrics-base-dir",
        type=str,
        default="results/training_outputs/metrics/",
        help="[dir mode] Base metrics directory holding {ModelType}{dataset}/ dirs.",
    )

    args = parser.parse_args()
    if args.mode == "dir" and (not args.model_types or not args.datasets):
        parser.error("--model-types and --datasets are required in dir mode")
    return args


# ============================================================================
# logs mode
# ============================================================================
def metrics_dir_from_log(log_path: Path) -> Path:
    """Resolve a per-run log to its metrics run directory.

    Parses the 'Saved test metrics to .../test_metrics.json' line printed at the
    end of training and returns its parent (the per-run metrics directory). If a
    log contains several such lines (re-runs appended), the last one wins.
    """
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        matches = _SAVED_RE.findall(fh.read())
    if not matches:
        raise ValueError(f"No 'Saved test metrics to' line in {log_path}")
    return Path(matches[-1]).parent


def discover_runs(
    logs_dir: Path, include: str = None, exclude: str = None, seeds=None
):
    """Find all {group}_seed{S}_run{R}.log files and group them by prefix.

    Returns a dict {group: [(seed, run, log_path), ...]} sorted by seed.
    """
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None
    seed_set = set(seeds) if seeds else None

    groups: dict[str, list] = defaultdict(list)
    for log_path in sorted(logs_dir.glob("*.log")):
        m = _LOGNAME_RE.match(log_path.stem)
        if not m:
            continue
        group = m.group("group")
        if include_re and not include_re.search(group):
            continue
        if exclude_re and exclude_re.search(group):
            continue
        if seed_set is not None and int(m.group("seed")) not in seed_set:
            continue
        groups[group].append((int(m.group("seed")), int(m.group("run")), log_path))

    for g in groups:
        groups[g].sort(key=lambda t: t[0])
    return dict(sorted(groups.items()))


def build_row_from_run_dir(
    group: str, run_dir: Path, run: int, log_name: str, seed_fallback: int
):
    """Load one run's test metrics + training params from its metrics dir into a row."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "test_metrics.json"
    params_path = run_dir / "training_params.json"

    if not metrics_path.exists():
        logger.info(f"⚠️  No test_metrics.json in {run_dir} (from {log_name})")
        return None

    with metrics_path.open("r") as f:
        data = json.load(f)

    params = {}
    if params_path.exists():
        with params_path.open("r") as f:
            params = json.load(f)

    model_cfg = params.get("model", {})
    training_cfg = params.get("training", {})
    metrics = data.get("metrics", {})

    return {
        "group": group,
        "model_type": group,
        "model_id": model_cfg.get("model_id", run_dir.name),
        "model_name": model_cfg.get("model_name", None),
        "seed": data.get("seed", seed_fallback),
        "run": run,
        # config context (useful for labelling tables / sanity checks)
        "datasets": training_cfg.get("datasets", None),
        "cspca_threshold": training_cfg.get("cspca_threshold", None),
        "loss_fn": training_cfg.get("loss_fn", None),
        "finetune_t": training_cfg.get("finetune_t", model_cfg.get("model_v", None)),
        "use_clinical": model_cfg.get("use_clinical", None),
        "att_fusion": model_cfg.get("att_fusion", None),
        # metrics (validation-best threshold)
        "threshold": data.get("threshold", None),
        "auroc": metrics.get("roc_auc", None),
        "pr_auc": metrics.get("pr_auc", None),
        "test_loss": metrics.get("loss", None),
        "accuracy": metrics.get("accuracy", None),
        "f1": metrics.get("f1", None),
        "precision": metrics.get("precision", None),
        "recall": metrics.get("recall", None),
        "run_dir": str(run_dir),
        "log": log_name,
    }


def load_run(group: str, seed: int, run: int, log_path: Path):
    """Resolve and load one per-seed run's metrics into a row."""
    try:
        run_dir = metrics_dir_from_log(log_path)
    except ValueError as e:
        logger.info(f"⚠️  {log_path.name}: {e}")
        return None
    return build_row_from_run_dir(group, run_dir, run, log_path.name, seed)


def run_logs_mode(args) -> pd.DataFrame:
    """Discover per-seed runs from the training logs and build the run table."""
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs dir not found: {logs_dir}")

    groups = discover_runs(logs_dir, args.include, args.exclude, args.seeds)
    if not groups:
        logger.info("No per-seed run logs discovered!")
        return pd.DataFrame()

    logger.info("=" * 80)
    logger.info(f"DISCOVERED {len(groups)} GROUPS IN {logs_dir}")
    logger.info("=" * 80)

    all_results = []
    for group, runs in groups.items():
        rows = [
            row
            for seed, run, log_path in runs
            if (row := load_run(group, seed, run, log_path)) is not None
        ]
        if rows:
            logger.info(f"✅ {group}: loaded {len(rows)}/{len(runs)} seeds")
            all_results.extend(rows)
        else:
            logger.info(f"⚠️  {group}: no usable runs")

    return pd.DataFrame(all_results)


# ============================================================================
# learning-curve full-data points (mode="logs", --lr-logs)
# ============================================================================
def discover_lr_frac1_runs(lr_log: Path):
    """Return the Fraction_1.00 (full-data) repeats of one lr-curve log.

    An lr-curve log bundles every fraction × repeat in a single file, so the
    generic {group}_seed{S}_run{R} discovery can't see them. Here we walk the
    file top-to-bottom, remembering the most recent 'Saved test metrics to ...'
    path, and whenever a 'View run Fraction_1.00_repeat_N' line appears we pair
    that repeat with the remembered dir (see _VIEW_FRAC1_RE for why order, not
    timestamps, is the reliable join).

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


def run_lr_logs(lr_specs) -> pd.DataFrame:
    """Build rows from the Fraction_1.00 point of each explicitly listed lr-curve log.

    Each spec is 'path' or 'label=path'. The label becomes the group/model_type
    (the table row); when omitted it defaults to the filename stem with any
    '_lr_curves...' suffix stripped (cnn, resnet, omnirad, radiodino). These logs
    are user-listed, so they bypass the --include/--exclude filters.
    """
    all_rows = []
    for spec in lr_specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = re.sub(r"_lr_curves.*$", "", Path(path).stem)

        lr_log = Path(path)
        if not lr_log.exists():
            logger.info(f"⚠️  lr-curve log not found: {lr_log}")
            continue

        pairs = discover_lr_frac1_runs(lr_log)
        if not pairs:
            logger.info(f"⚠️  No Fraction_1.00 runs found in {lr_log.name}")
            continue

        rows = [
            row
            for repeat, run_dir in pairs
            if (row := build_row_from_run_dir(
                label, run_dir, repeat, lr_log.name, seed_fallback=repeat
            )) is not None
        ]
        if rows:
            logger.info(
                f"✅ {label}: loaded {len(rows)}/{len(pairs)} Fraction_1.00 runs "
                f"from {lr_log.name}"
            )
            all_rows.extend(rows)
        else:
            logger.info(f"⚠️  {label}: no usable Fraction_1.00 runs in {lr_log.name}")

    return pd.DataFrame(all_rows)


# ============================================================================
# dir mode (legacy)
# ============================================================================
def extract_model_type(model_id: str) -> str:
    """
    Extract human-readable model type from model_id.

    Expected format: {model_name}_{loss}_{mode}_{version}
    e.g. OmniRad-base_bce_transfer_v0
         OmniRad-large_focal_finetuned_v3
    """
    parts = model_id.split("_")

    # model_name may contain hyphens but not underscores, so it's always parts[0]
    model_name = parts[0] if len(parts) > 0 else model_id

    # mode: transfer or finetuned
    mode = None
    for p in parts[1:5]:
        if p.lower() in ("transfer", "finetuned"):
            mode = p.capitalize()
            break

    # version: anything matching v\d+
    finetune_type = None
    for p in parts[1:5]:
        if p.lower().startswith("v") and p[1:].isdigit():
            finetune_type = p.lower()
            break

    finetune_type_map = {
        "v0": "Transfer",
        "v2": "Partial Finetune",
        "v5": "Full Finetune",
    }

    if finetune_type is not None:
        finetune_type = finetune_type_map.get(finetune_type)

    # Optional suffixes: _clinical and/or _att
    remaining = parts[4:] if len(parts) > 4 else []
    clinical = "Clinical" if "clinical" in [p.lower() for p in remaining] else None
    attention = "Attention" if "att" in [p.lower() for p in remaining] else None
    cls_token = "cls" if "cls" in [p.lower() for p in remaining] else None

    components = [model_name]
    if finetune_type:
        components.append(finetune_type)
    if clinical:
        components.append(clinical)
    if attention:
        components.append(attention)
    if cls_token:
        components.append(cls_token)

    return " | ".join(components)


def load_test_metrics_from_dir(meta_dir: Path, dataset_name: str = ""):
    """
    Load all test metrics and training params JSONs from a given meta directory.
    Each model is expected to live in its own subdirectory containing:
      - training_params.json
      - test_metrics.json
    """
    results = []

    if not meta_dir.exists():
        logger.info(f"⚠️  Directory not found: {meta_dir}")
        return results

    for model_dir in sorted(meta_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        params_path = model_dir / "training_params.json"
        metrics_path = model_dir / "test_metrics.json"

        if not params_path.exists():
            logger.info(f"⚠️  No training_params.json found in {model_dir.name}")
            continue
        if not metrics_path.exists():
            logger.info(f"⚠️  No test_metrics.json found in {model_dir.name}")
            continue

        with params_path.open("r") as f:
            params = json.load(f)

        with metrics_path.open("r") as f:
            data = json.load(f)

        model_cfg = params.get("model", {})
        training_cfg = params.get("training", {})

        model_id = model_cfg.get("model_id", model_dir.name)
        if "Debug" in model_id:
            logger.info(f"⏭️  Skipping debug model: {model_id}")
            continue

        seed = data.get("seed", training_cfg.get("seed", 0))
        metrics = data.get("metrics", {})
        dataset_used = f" | {dataset_name}" if dataset_name != "" else ""

        results.append(
            {
                "model_id": model_id,
                "model_type": extract_model_type(model_id),
                "model_run": f"{extract_model_type(model_id)}{dataset_used}",
                "seed": seed,
                "loss_fn": training_cfg.get("loss_fn", None),
                "finetune_t": training_cfg.get("finetune_t", None),
                "learning_rate": training_cfg.get("learning_rate", None),
                "num_epochs": training_cfg.get("num_epochs", None),
                "threshold": data.get("threshold", None),
                "auroc": metrics.get("roc_auc", None),
                "pr_auc": metrics.get("pr_auc", None),
                "test_loss": metrics.get("loss", None),
                "accuracy": metrics.get("accuracy", None),
                "f1": metrics.get("f1", None),
                "precision": metrics.get("precision", None),
                "recall": metrics.get("recall", None),
                "checkpoint": data.get("checkpoint", ""),
            }
        )

    return results


def run_dir_mode(args) -> pd.DataFrame:
    """Scan a legacy {ModelType}{dataset}/ metrics tree and build the run table."""
    dataset_map = {
        "picai": "-p",
        "goc": "-g",
        "picai+goc": "-pg",
        "promis": "-pr",
        "all": "-all",
    }
    model_type_map = {
        "omnirad": "OmniRad",
        "resnet": "ResNet3D",
        "cnn": "3DCNN",
        "radiodino": "RadioDino",
    }

    if not all(m in model_type_map for m in args.model_types):
        raise ValueError(f"All model types must be one of {list(model_type_map)}")
    if not all(m in dataset_map for m in args.datasets):
        raise ValueError(f"All dataset types must be one of {list(dataset_map)}")

    training_outs = Path(args.metrics_base_dir)
    meta_dirs = {
        training_outs / f"{model_type_map[t]}{dataset_map[d]}": d.upper()
        for t, d in product(args.model_types, args.datasets)
    }

    all_results = []
    for meta_dir, dataset_name in meta_dirs.items():
        results = load_test_metrics_from_dir(meta_dir, dataset_name)
        if results:
            logger.info(f"✅ Loaded {len(results)} runs from {meta_dir.name}")
            all_results.extend(results)

    return pd.DataFrame(all_results)


# ============================================================================
# shared aggregation / output
# ============================================================================
def _flatten_grouped_columns(df_grouped: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by groupby.agg."""
    df_flat = df_grouped.copy()
    df_flat.columns = ["_".join(col).strip() for col in df_flat.columns.values]
    return df_flat


def plot_ranked_metric_panels(df_grouped: pd.DataFrame, out_path: Path):
    """4x1 panel plot of AUROC, F1, Precision/Recall, PR-AUC (mean ± std).

    Groups are sorted by AUROC mean for a consistent worst -> best ordering.
    """
    df_flat = _flatten_grouped_columns(df_grouped)
    df_sorted = df_flat.sort_values("auroc_mean")
    labels = df_sorted.index.tolist()
    x = np.arange(len(labels))

    def series(col):
        return (
            df_sorted[f"{col}_mean"].values,
            df_sorted[f"{col}_std"].fillna(0).values,
        )

    auroc_mean, auroc_std = series("auroc")
    f1_mean, f1_std = series("f1")
    prec_mean, prec_std = series("precision")
    rec_mean, rec_std = series("recall")
    pr_auc_mean, pr_auc_std = series("pr_auc")

    fig, axes = plt.subplots(
        4, 1, figsize=(max(10, len(labels) * 0.5), 13), sharex=True
    )

    axes[0].errorbar(x, auroc_mean, yerr=auroc_std, fmt="o-", ecolor="gray",
                     capsize=4, linewidth=2, markersize=6, color="tab:blue", label="AUROC")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Model Performance by Group (mean ± std across seeds)")
    axes[0].legend(loc="lower right")

    axes[1].errorbar(x, f1_mean, yerr=f1_std, fmt="o-", ecolor="gray",
                     capsize=4, linewidth=2, markersize=6, color="tab:green", label="F1")
    axes[1].set_ylabel("F1")
    axes[1].legend(loc="lower right")

    axes[2].errorbar(x, prec_mean, yerr=prec_std, fmt="o-", ecolor="gray",
                     capsize=4, linewidth=2, markersize=6, color="tab:orange", label="Precision")
    axes[2].errorbar(x, rec_mean, yerr=rec_std, fmt="s-", ecolor="lightgray",
                     capsize=4, linewidth=2, markersize=6, color="tab:red", label="Recall")
    axes[2].set_ylabel("Precision / Recall")
    axes[2].legend(loc="lower right")

    axes[3].errorbar(x, pr_auc_mean, yerr=pr_auc_std, fmt="o-", ecolor="gray",
                     capsize=4, linewidth=2, markersize=6, color="tab:purple", label="PR AUC")
    axes[3].set_ylabel("PR AUC")
    axes[3].set_xlabel("Model group (sorted by AUROC)")
    axes[3].legend(loc="lower right")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Saved AUROC/F1/Precision/Recall/PR_AUC panel plot to: {out_path}")


def aggregate_and_report(df: pd.DataFrame, args):
    """Per-run table, mean±std by group, CSV/JSON dumps, and the panel plot."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n✅ Loaded {len(df)} model runs total")
    logger.info("=" * 80)
    logger.info("AGGREGATING TEST METRICS FROM ALL RUNS")
    logger.info("=" * 80)

    df = df.sort_values(["model_type", "seed"])

    csv_path = output_dir / "all_models_test_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✅ Saved full comparison to: {csv_path}")

    json_path = output_dir / "all_models_test_comparison.json"
    with json_path.open("w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    logger.info(f"✅ Saved full comparison to: {json_path}")

    # -------- Per-run summary --------
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET PERFORMANCE (per run, at validation-best threshold)")
    logger.info("=" * 80)

    summary_cols = ["model_type", "model_id", "seed", *METRIC_COLS]
    summary_df = df[summary_cols].copy().round(3)
    with pd.option_context("display.max_colwidth", None, "display.width", None,
                           "display.max_rows", None):
        logger.info("\n" + summary_df.to_string(index=False))

    # -------- Aggregate by group/model_type (mean ± std across seeds) --------
    logger.info("\n" + "=" * 80)
    logger.info("MEAN ± STD BY GROUP (at validation-best threshold)")
    logger.info("=" * 80)

    agg_spec = {"auroc": ["mean", "std", "count"]}
    agg_spec.update({c: ["mean", "std"] for c in METRIC_COLS if c != "auroc"})
    grouped = df.groupby("model_type").agg(agg_spec).round(3)
    with pd.option_context("display.max_colwidth", None, "display.width", None,
                           "display.max_rows", None):
        logger.info("\n" + grouped.to_string())

    agg_csv = output_dir / "all_models_aggregated.csv"
    _flatten_grouped_columns(grouped).to_csv(agg_csv)
    logger.info(f"\n✅ Saved aggregated comparison to: {agg_csv}")

    if not args.no_plot:
        plot_ranked_metric_panels(grouped, output_dir / "all_models_ranked_panels.png")

    # -------- Top 5 runs by AUROC --------
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 RUNS BY AUROC (individual seeds)")
    logger.info("=" * 80)
    top5 = df.nlargest(5, "auroc")[["model_type", "model_id", "seed", *METRIC_COLS]].round(3)
    logger.info("\n" + top5.to_string(index=False))

    logger.info("\n" + "=" * 80)
    logger.info("✅ Analysis complete!")
    logger.info("=" * 80)


def main():
    args = parse_args()

    if args.mode == "logs":
        df = run_logs_mode(args)
        if args.lr_logs:
            lr_df = run_lr_logs(args.lr_logs)
            frames = [f for f in (df, lr_df) if not f.empty]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = run_dir_mode(args)

    if df.empty:
        logger.info("No test metrics found!")
        return

    aggregate_and_report(df, args)


if __name__ == "__main__":
    main()
