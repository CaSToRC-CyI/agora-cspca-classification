"""Test whether binary csPCa models' continuous scores carry ordinal Gleason info.

Each model is trained with a single binary cut (csPCa vs not, at the GG1|GG2
boundary). This script asks: do the sigmoid scores nonetheless rank samples by
ISUP Grade Group (GG1 -> GG5)? We run inference on each run's own test
split, tag every sample with its GG, and measure rank agreement (Spearman /
Kendall), adjacent-grade concordance (pairwise AUC), and per-GG score trends.

Models analysed are the best-performing *clinical* configs (flatten + att
fusion), seeds 7-11 -- exactly the CURVE_SETS["thr7"] mapping reused from
generate_roc_auc_pn_ratios.py.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import kendalltau, rankdata, spearmanr  # noqa: E402

from agora.scripts.generate_roc_auc_pn_ratios import (  # noqa: E402
    CURVE_SETS,
    RUN_OF_SEED,
    SEEDS,
    extract_model_id,
    make_dataloaders,
    metrics_dir_from_log,
)
from agora.scripts.train_v2 import get_config_class, get_trainer  # noqa: E402

# Best clinical configs (flatten + att), seeds 7-11. Reused verbatim.
MODELS = CURVE_SETS["thr7"]

# ISUP grade-group labels for plotting/tables.
GG_LABELS = {
    1: "GG1 (<=6)",
    2: "GG2 (3+4)",
    3: "GG3 (4+3)",
    4: "GG4 (=8)",
    5: "GG5 (9-10)",
}


def gleason_to_gg(gs: str):
    """Map a stored true_gs string (max lesion score) to an ISUP grade group.

    GG1 is Gleason <=6, so benign/no-tumour (0+0, nan, N/A) folds into GG1 (the
    low-risk tier); None only for unparseable values. 3+4 and 4+3 both sum to 7
    but are GG2 vs GG3, so the primary/secondary order separates them.
    """
    if gs is None:
        return 1
    gs = gs.decode() if isinstance(gs, bytes) else str(gs)
    gs = gs.strip().strip('"')
    if gs in ("", "nan", "None", "N/A", "0+0"):
        return 1
    try:
        primary, secondary = (int(x) for x in gs.split("+"))
    except (ValueError, AttributeError):
        return None
    total = primary + secondary
    if total <= 6:
        return 1
    if total == 7:
        return 2 if primary <= 3 else 3  # 3+4 -> GG2, 4+3 -> GG3
    if total == 8:
        return 4
    return 5  # 9-10


def load_model(run_dir: Path, model_type: str, device):
    """Rebuild a trained model from its metrics run dir + best checkpoint.

    Mirrors get_tpr_fpr() in generate_roc_auc_pn_ratios.py: metrics and
    checkpoints live in sibling trees, and old checkpoints prefixed their
    aggregator "spatial_aggregator".
    """
    run_dir = Path(run_dir)
    ckpt_dir = (
        run_dir.parent.parent.parent
        / "checkpoints"
        / run_dir.parent.stem
        / run_dir.name
    )
    ckpt_path = ckpt_dir / "best_model.pt"

    cfg = get_config_class(model_type).from_json(run_dir / "training_params.json")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    trainer = get_trainer(model_type).build_model_only(cfg)
    model = trainer.model

    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("spatial_aggregator") for k in state_dict):
        state_dict = {
            k.replace("spatial_aggregator", "aggregator"): v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


@torch.no_grad()
def score_test_split(model, cfg, device):
    """Return per-sample (score, label, true_gs) over this run's test split."""
    _, _, test_loader = make_dataloaders(cfg)
    use_clinical = cfg.model.use_clinical

    scores, labels, true_gs = [], [], []
    for batch in test_loader:
        images = batch["image"].to(device)
        if use_clinical:
            outputs = model(images, batch["clinical"].to(device)).squeeze(-1)
        else:
            outputs = model(images).squeeze(-1)
        scores.extend(torch.sigmoid(outputs).flatten().cpu().numpy().tolist())
        labels.extend(batch["label"].flatten().numpy().tolist())
        true_gs.extend(batch["true_gs"])
    return scores, labels, true_gs


def collect_scores(logs_dir: Path, device) -> pd.DataFrame:
    """Run inference for every model x seed and return a long-form score table."""
    rows = []
    for model_name, (model_type, pattern) in MODELS.items():
        for seed in SEEDS:
            log_name = pattern.format(s=seed, r=RUN_OF_SEED[seed])
            try:
                run_dir = metrics_dir_from_log(logs_dir, log_name)
            except (FileNotFoundError, ValueError) as e:
                print(f"  {model_name} seed {seed}: SKIP ({e})")
                continue

            print(f"  {model_name} seed {seed}: {run_dir}")
            model, cfg = load_model(run_dir, model_type, device)
            scores, labels, gs = score_test_split(model, cfg, device)

            model_id = extract_model_id(run_dir)
            for sc, lb, g in zip(scores, labels, gs):
                gg = gleason_to_gg(g)
                if gg is None:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "model_type": model_type,
                        "model_id": model_id,
                        "seed": seed,
                        "score": sc,
                        "label": int(lb),
                        "true_gs": (g.decode() if isinstance(g, bytes) else str(g)),
                        "gg": gg,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ordinal statistics
# ---------------------------------------------------------------------------
def _pairwise_auc(low_scores, high_scores) -> float:
    """P(score(high grade) > score(low grade)); ties count 0.5. NaN if a side empty.

    Tie-aware Mann-Whitney AUC via average ranks over the pooled scores.
    """
    n_low, n_high = len(low_scores), len(high_scores)
    if n_low == 0 or n_high == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([low_scores, high_scores]))
    rank_high = ranks[n_low:].sum()
    return (rank_high - n_high * (n_high + 1) / 2) / (n_low * n_high)


def ordinal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per model x seed: Spearman/Kendall over GG1-5 + ordinal C-index.

    Keeps the correlation p-values -- seeds are overlapping splits of one dataset,
    so we report them per seed rather than pooling (pooling would double-count).
    """
    rows = []
    for (model, seed), grp in df.groupby(["model", "seed"]):
        rho, p_rho = spearmanr(grp.score, grp.gg) if len(grp) > 2 else (np.nan, np.nan)
        tau, p_tau = kendalltau(grp.score, grp.gg) if len(grp) > 2 else (np.nan, np.nan)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "n": len(grp),
                "spearman_gg1_5": rho,
                "spearman_gg1_5_p": p_rho,
                "kendall_gg1_5": tau,
                "kendall_gg1_5_p": p_tau,
                "c_index_gg1_5": _ordinal_c_index(grp.score.values, grp.gg.values),
            }
        )
    return pd.DataFrame(rows)


def pvalue_summary(rank: pd.DataFrame, alpha: float = 0.05) -> list:
    """Per model: min/max p across seeds + how many seeds are significant."""
    p_cols = ["spearman_gg1_5_p", "kendall_gg1_5_p"]
    out = []
    for model, grp in rank.groupby("model"):
        rec = {"model": model, "n_seeds": int(len(grp))}
        for col in p_cols:
            rec[f"{col}_min"] = float(grp[col].min())
            rec[f"{col}_max"] = float(grp[col].max())
            rec[f"{col}_n_sig"] = int((grp[col] < alpha).sum())
        out.append(rec)
    return out


def _ordinal_c_index(scores, gg) -> float:
    """Fraction of concordant pairs among all pairs with different GG (ties=0.5)."""
    scores, gg = np.asarray(scores), np.asarray(gg)
    conc = disc = ties = 0
    for i in range(len(scores)):
        dg = gg[i] - gg
        ds = scores[i] - scores
        mask = dg > 0  # i has strictly higher grade
        conc += np.sum((ds[mask] > 0))
        disc += np.sum((ds[mask] < 0))
        ties += np.sum((ds[mask] == 0))
    total = conc + disc + ties
    return (conc + 0.5 * ties) / total if total else float("nan")


def adjacent_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """Per model x seed: pairwise AUC between each adjacent GG pair (k vs k+1)."""
    rows = []
    for (model, seed), grp in df.groupby(["model", "seed"]):
        by_gg = {g: grp[grp.gg == g].score.values for g in range(1, 6)}
        rec = {"model": model, "seed": seed}
        for k in range(1, 5):
            rec[f"auc_{k}v{k + 1}"] = _pairwise_auc(by_gg[k], by_gg[k + 1])
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(per_seed: pd.DataFrame, value_cols) -> pd.DataFrame:
    """Mean +/- std across seeds, per model."""
    agg = per_seed.groupby("model")[value_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.round(4)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_score_by_gg(df: pd.DataFrame, out_path: Path):
    """One box/strip panel per model: score distribution across GG1-5."""
    models = list(MODELS)
    ggs = list(range(1, 6))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for ax, model in zip(axes.flat, models):
        sub = df[df.model == model]
        groups = [sub[sub.gg == g].score.values for g in ggs]
        ax.boxplot(groups, positions=ggs, widths=0.6, showfliers=False)
        for g, y in zip(ggs, groups):
            if len(y):
                ax.scatter(
                    np.full(len(y), g) + np.random.uniform(-0.15, 0.15, len(y)),
                    y, s=6, alpha=0.25, color="tab:blue",
                )
        ax.set_title(model)
        ax.set_xticks(ggs)
        ax.set_xticklabels([GG_LABELS[g].split(" ")[0] for g in ggs])
        ax.set_ylabel("csPCa score")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Continuous score distribution by ISUP Grade Group (pooled seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mean_trend(df: pd.DataFrame, out_path: Path):
    """Mean score vs GG: faint per-seed lines + bold across-seed mean per model."""
    models = list(MODELS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for ax, model in zip(axes.flat, models):
        sub = df[df.model == model]
        for seed in sorted(sub.seed.unique()):
            s = sub[sub.seed == seed].groupby("gg").score.mean()
            ax.plot(s.index, s.values, color="gray", alpha=0.35, linewidth=1)
        m = sub.groupby(["seed", "gg"]).score.mean().groupby("gg").agg(["mean", "std"])
        ax.errorbar(m.index, m["mean"], yerr=m["std"].fillna(0), fmt="o-",
                    color="tab:red", capsize=4, linewidth=2, label="mean +/- std")
        ax.set_title(model)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("ISUP Grade Group")
        ax.set_ylabel("mean csPCa score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle("Mean score vs Grade Group (per-seed grey, across-seed mean red)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs-dir", type=str,
                   default="results/logs")
    p.add_argument("--out-dir", "-o", type=str,
                   default="results/ordinal_analysis")
    p.add_argument("--scores-csv", type=str, default=None,
                   help="Skip inference and load an existing long-form scores CSV.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.scores_csv:
        df = pd.read_csv(args.scores_csv)
    else:
        df = collect_scores(Path(args.logs_dir), device)
        df.to_csv(out_dir / "ordinal_scores.csv", index=False)
        print(f"Saved {out_dir / 'ordinal_scores.csv'} ({len(df)} rows)")

    if df.empty:
        print("No scores collected -- nothing to analyse.")
        return

    # recompute GG from true_gs so mapping changes apply on the --scores-csv path too
    df["gg"] = df["true_gs"].map(gleason_to_gg)
    df = df[df.gg.notna()].copy()
    df["gg"] = df["gg"].astype(int)

    # per-sample counts by GG (sanity)
    print("\nSample counts by model x GG (pooled seeds):")
    print(df.pivot_table(index="model", columns="gg", values="score",
                         aggfunc="count", fill_value=0))

    # rank / concordance stats
    rank = ordinal_stats(df)
    adj = adjacent_concordance(df)
    rank.to_csv(out_dir / "ordinal_rank_per_seed.csv", index=False)
    adj.to_csv(out_dir / "adjacent_concordance_per_seed.csv", index=False)

    rank_summary = summarize(
        rank, ["spearman_gg1_5", "kendall_gg1_5", "c_index_gg1_5"]
    )
    adj_summary = summarize(adj, [f"auc_{k}v{k + 1}" for k in range(1, 5)])
    rank_summary.to_csv(out_dir / "ordinal_rank_summary.csv")
    adj_summary.to_csv(out_dir / "adjacent_concordance_summary.csv")

    # per-GG mean score table
    per_gg = (
        df.groupby(["model", "gg"]).score.agg(["mean", "median", "std", "count"])
        .round(4)
    )
    per_gg.to_csv(out_dir / "score_by_gg.csv")

    # per-seed p-value ranges (recovered from the same correlations)
    p_summary = pvalue_summary(rank)

    print("\n=== Rank agreement (mean +/- std across seeds) ===")
    print(rank_summary.to_string())
    print("\n=== Adjacent-grade concordance / pairwise AUC ===")
    print(adj_summary.to_string())
    print("\n=== p-value ranges across seeds (per model) ===")
    for rec in p_summary:
        print(
            f"  {rec['model']}: spearman_gg1_5 p in "
            f"[{rec['spearman_gg1_5_p_min']:.1e}, {rec['spearman_gg1_5_p_max']:.1e}] "
            f"({rec['spearman_gg1_5_p_n_sig']}/{rec['n_seeds']} seeds sig)"
        )

    # plots
    plot_score_by_gg(df, out_dir / "score_by_gg.png")
    plot_mean_trend(df, out_dir / "mean_score_trend.png")

    # machine-readable summary
    with (out_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "rank_summary": rank_summary.reset_index().to_dict(orient="records"),
                "adjacent_concordance": adj_summary.reset_index().to_dict(orient="records"),
                "p_summary": p_summary,
                "n_samples": int(len(df)),
                "seeds": SEEDS,
                "models": {k: v[1] for k, v in MODELS.items()},
            },
            f,
            indent=2,
        )
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
