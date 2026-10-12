#!/usr/bin/env python3
"""Step 0 for an external cohort: resample images + scaffold the label CSV.

This is a *base* adapter, not a universal one. External datasets differ in their
on-disk layout and in where labels come from, so the two dataset-specific pieces
are made config/CLI-driven; everything after them (resampling parameters, the
canonical output naming, and the downstream loader) stays fixed so the rest of
the pipeline is dataset-agnostic. For a genuinely odd cohort, copy this file and
override `discover_studies` / `resolve_labels`.

What is generic here:
    - input discovery via per-modality glob patterns with a ``{study}`` slot,
    - resampling via ``MRIStudy`` with the SAME params as training,
    - output written with canonical modality tokens (``_t2w``/``_adc``/``_hbv``)
      regardless of the input naming, so the loader can read them.

Stages:
    resample : discover {T2,ADC,DWI} per study from --input-dir using
               ``input_layout``, resample each (T2->global spacing, ADC/DWI->T2
               ref, center crop/pad), and write canonical-named volumes to
               <data_dir>/<image_subdir>.
    csv      : discover resampled studies and write csv_file with an optional
               binary ``label`` column (per ``label_mode``) plus clinical columns.
    all      : resample -> csv.

Labels: this base script only knows how to (a) leave the cohort unlabeled
(``label_mode: none``) or (b) join a ready-made binary 0/1 column from a source
CSV (``label_mode: csv`` + ``--label-col``). Cohorts whose labels must be derived
(e.g. from the presence of a tumor mask) need a small dataset-specific script to
produce that source CSV first.

Example (per-case folders, labels joined from a metadata CSV):
    uv run python src/agora/scripts/prepare_external_data.py \\
        --external-config configs/prostate158_eval.yaml \\
        --input-dir /path/to/prostate158_raw --stage all \\
        --source-csv /path/to/labels.csv --source-id-col case --label-col csPCa
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
import yaml
from tqdm import tqdm

from agora.data.dataloader import CLINICAL_COLS, CachedMRIDataset
from agora.data.resampling import MRIStudy

# Canonical output tokens the downstream loader (_find_study_groups, goc mode)
# recognizes. nii.gz files keep a trailing ".nii" after Path.stem, so the loader
# matches these by startswith() on the nnUNet channel suffixes -- exact "t2w"/
# "adc"/"hbv" would NOT match (that spelling only works for picai .mha). This is
# also the naming the real goc resampled cohort uses. vols come out of MRIStudy
# in [T2, ADC, DWI] order.
CANONICAL_TOKENS = ["0000", "0001", "0003"]  # nnUNet channels: T2 / ADC / DWI

# Sensible default layout: the flat picai/goc filename convention. Override in
# the config's `input_layout` block for per-case-folder datasets.
DEFAULT_LAYOUT = {
    "t2": "{study}_t2w.nii.gz",
    "adc": "{study}_adc.nii.gz",
    "dwi": "{study}_hbv.nii.gz",
}


def load_external_block(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("external_eval", cfg)


def _pattern_to_regex(pattern: str) -> re.Pattern:
    """Turn a ``{study}``-templated glob into a regex that captures the id."""
    parts = pattern.split("{study}")
    body = "(?P<study>.+?)".join(re.escape(p) for p in parts)
    return re.compile(body + r"$")


def discover_studies(input_dir: Path, layout: dict) -> dict:
    """Group raw input files into {study_id: {"T2"|"ADC"|"DWI": path}}.

    ``layout`` maps modality keys (t2/adc/dwi) to glob patterns relative to
    ``input_dir`` containing a single ``{study}`` placeholder, e.g.
    ``"{study}/t2.nii.gz"`` or ``"{study}_t2w.nii.gz"``.
    """
    input_dir = Path(input_dir)
    mod_map = {"t2": "T2", "adc": "ADC", "dwi": "DWI"}
    studies: dict = defaultdict(dict)
    for key, dst in mod_map.items():
        pattern = layout.get(key)
        if not pattern:
            raise ValueError(f"input_layout is missing a pattern for modality '{key}'")
        regex = _pattern_to_regex(pattern)
        glob_pat = pattern.replace("{study}", "*")
        for path in sorted(input_dir.glob(glob_pat)):
            rel = path.relative_to(input_dir).as_posix()
            m = regex.match(rel)
            if m:
                studies[m.group("study")][dst] = path
    return studies


def discover_resampled(image_dir: Path) -> list:
    """Study ids that have all three canonical-named resampled volumes."""
    groups = CachedMRIDataset._find_study_groups(image_dir, dataset="goc")
    return [sid for sid, mods in groups.items() if {"T2", "ADC", "DWI"} <= set(mods)]


def stage_resample(block: dict, input_dir: Path, physical_size, global_spacing):
    out_dir = Path(block["data_dir"]) / block["image_subdir"]
    layout = block.get("input_layout", DEFAULT_LAYOUT)
    study_dict = discover_studies(input_dir, layout)
    if not study_dict:
        raise RuntimeError(
            f"No studies matched input_layout under {input_dir}. Check the "
            f"glob patterns: {layout}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_skip = 0, 0
    for study_id, paths in tqdm(study_dict.items(), total=len(study_dict)):
        if not {"T2", "ADC", "DWI"} <= set(paths):
            print(f"  skip {study_id}: missing a modality {set(paths)}")
            n_skip += 1
            continue
        study = MRIStudy(
            study_id,
            [str(paths["T2"]), str(paths["ADC"]), str(paths["DWI"])],
            physical_size=tuple(physical_size),
            global_spacing=tuple(global_spacing),
        )
        study.preprocess()
        # Write with canonical tokens ourselves so output naming is independent
        # of the (arbitrary) input naming. vols order is [T2, ADC, DWI].
        for vol, token in zip(study.vols, CANONICAL_TOKENS):
            sitk.WriteImage(vol, out_dir / f"{study_id}_{token}.nii.gz")
        n_ok += 1
    print(f"Resampled {n_ok} studies into {out_dir} ({n_skip} skipped)")
    if n_ok == 0:
        raise RuntimeError(
            f"No complete (T2+ADC+DWI) studies found under {input_dir}."
        )


def resolve_labels(study_ids, block: dict, args):
    """Return {study_id: 0/1/""} per label_mode, or None for unlabeled cohorts."""
    mode = (
        args.label_mode
        or block.get("label_mode")
        or ("csv" if args.source_csv and args.label_col else "none")
    ).lower()

    if mode == "none":
        return None

    if mode == "csv":
        if not args.source_csv or not args.label_col:
            raise ValueError("label_mode=csv needs --source-csv and --label-col")
        # Read the id column as str so zero-padded ids (e.g. "024") survive:
        # resampled study ids are always strings, and pandas would otherwise
        # coerce a numeric id column to int and drop the padding.
        src = pd.read_csv(
            args.source_csv, dtype={args.source_id_col: str}
        ).set_index(args.source_id_col)
        labels = {}
        for sid in study_ids:
            if sid not in src.index:
                labels[sid] = ""
                continue
            try:
                labels[sid] = int(float(src.loc[sid, args.label_col]))
            except (TypeError, ValueError):
                labels[sid] = ""
        matched = sum(v != "" for v in labels.values())
        print(f"csv labels: matched {matched}/{len(labels)} studies")
        return labels

    raise ValueError(f"Unknown label_mode: {mode!r}")


def stage_csv(block: dict, args):
    image_dir = Path(block["data_dir"]) / block["image_subdir"]
    csv_path = Path(block["csv_file"])
    study_ids = sorted(discover_resampled(image_dir))
    if not study_ids:
        raise RuntimeError(
            f"No resampled studies under {image_dir}; run --stage resample first."
        )

    labels = resolve_labels(study_ids, block, args)

    columns = ["study_id"]
    if labels is not None:
        columns.append("label")
    columns += list(CLINICAL_COLS)

    df = pd.DataFrame({c: "" for c in columns}, index=range(len(study_ids)))
    df["study_id"] = study_ids
    if labels is not None:
        df["label"] = [labels.get(s, "") for s in study_ids]

    # Optional clinical join from the source CSV (independent of label_mode).
    clin_map = {
        "patient_age": args.age_col,
        "psa": args.psa_col,
        "psad": args.psad_col,
        "prostate_volume": args.volume_col,
    }
    if args.source_csv and any(clin_map.values()):
        src = pd.read_csv(args.source_csv).set_index(args.source_id_col)
        for i, sid in enumerate(study_ids):
            if sid not in src.index:
                continue
            for dst, src_col in clin_map.items():
                if src_col and src_col in src.columns:
                    df.at[i, dst] = src.loc[sid, src_col]
        print(f"Joined clinical columns for studies present in {args.source_csv}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    note = "with label column" if labels is not None else "unlabeled (no label column)"
    print(f"Wrote {csv_path} ({len(df)} rows, {note})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external-config", required=True, help="YAML with external_eval block.")
    p.add_argument("--input-dir", default=None, help="Raw image dir (required for resample).")
    p.add_argument("--stage", choices=["resample", "csv", "all"], default="all")
    p.add_argument("--physical-size", type=float, nargs=3, default=[81.0, 192.0, 192.0],
                   help="Center crop/pad size in mm, ZYX (default matches training).")
    p.add_argument("--global-spacing", type=float, nargs=3, default=[0.5, 0.5, 3.0],
                   help="Resample spacing in mm, XYZ (default matches training).")
    # Label resolution (overrides config's label_mode if given).
    p.add_argument("--label-mode", choices=["none", "csv"], default=None,
                   help="How to derive the binary label. Defaults to the config value.")
    p.add_argument("--label-col", default=None, help="csv: binary 0/1 column in --source-csv.")
    # Optional clinical (and csv-label) source join:
    p.add_argument("--source-csv", default=None, help="Source metadata CSV to join from.")
    p.add_argument("--source-id-col", default="study_id", help="ID column in --source-csv.")
    p.add_argument("--age-col", default=None, help="Source column -> patient_age.")
    p.add_argument("--psa-col", default=None, help="Source column -> psa.")
    p.add_argument("--psad-col", default=None, help="Source column -> psad.")
    p.add_argument("--volume-col", default=None, help="Source column -> prostate_volume.")
    return p.parse_args()


def main():
    args = parse_args()
    block = load_external_block(args.external_config)

    if args.stage in ("resample", "all"):
        if not args.input_dir:
            raise ValueError("--input-dir is required for the resample stage.")
        stage_resample(block, Path(args.input_dir), args.physical_size, args.global_spacing)
    if args.stage in ("csv", "all"):
        stage_csv(block, args)


if __name__ == "__main__":
    main()
