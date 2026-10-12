#!/usr/bin/env python3
"""Generate prostate segmentations for an external cohort (nnUNet v1).

A reusable, config-driven script so external data is segmented with the SAME model
(Task005_Prostate, 3d_fullres) used for the training cohorts.

Stages:
    prepare  : stage T2 (_0000) + ADC (_0001) as nii.gz into an nnUNet input dir,
               using the same study grouping as the dataloader so output mask
               names ({study_id}.nii.gz) match what evaluation looks up.
    predict  : run `nnUNet_predict` (only with --run-predict; otherwise printed,
               since nnUNet typically lives in a separate venv).
    transfer : copy the produced {study_id}.nii.gz masks into the external seg_dir.
    all      : prepare -> predict -> transfer.

Example:
    uv run python src/agora/scripts/prepare_external_segmentations.py \\
        --external-config configs/external_eval_example.yaml --stage prepare
    # then, in your nnUNet venv:
    nnUNet_predict -i <input_dir> -o <output_dir> -t 5 -m 3d_fullres
    uv run python src/agora/scripts/prepare_external_segmentations.py \\
        --external-config configs/external_eval_example.yaml --stage transfer
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

from agora.data.dataloader import CachedMRIDataset


def load_external_block(yaml_path: str) -> dict:
    """Read the external_eval block raw (avoids ExternalEvalConfig's seg_dir check)."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("external_eval", cfg)


def clean_dir(d: Path):
    files = [f for f in d.iterdir() if f.is_file()] if d.exists() else []
    if files:
        print(f"Cleaning {len(files)} files from {d}")
        for f in files:
            f.unlink()


def stage_prepare(block: dict, input_dir: Path, clean: bool):
    data_dir = Path(block["data_dir"])
    image_dir = data_dir / block["image_subdir"]
    kind = "picai" if str(block.get("file_format", "nii.gz")).lstrip(".") == "mha" else "goc"

    input_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        clean_dir(input_dir)

    study_dict = CachedMRIDataset._find_study_groups(image_dir, dataset=kind)
    n = 0
    for study_id, paths in study_dict.items():
        if "T2" not in paths or "ADC" not in paths:
            print(f"  skip {study_id}: missing T2 or ADC")
            continue
        # External data preparation uses nnUNet's channel convention.
        for chan, key in (("0000", "T2"), ("0001", "ADC")):
            shutil.copy(paths[key], input_dir / f"{study_id}_{chan}.nii.gz")
        n += 1
    print(f"Prepared {n} cases into {input_dir}")
    if n == 0:
        raise RuntimeError(
            f"No T2+ADC pairs found under {image_dir}. Check image_subdir/file_format."
        )


def stage_predict(input_dir: Path, output_dir: Path, task: str, model: str, run: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNet_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-t", task,
        "-m", model,
    ]
    printable = " ".join(cmd)
    if run:
        print(f"Running: {printable}")
        subprocess.run(cmd, check=True)
    else:
        print(
            "\nRun the following in your nnUNet venv, then re-run with --stage transfer:\n"
            f"    {printable}\n"
        )


def stage_transfer(block: dict, output_dir: Path):
    seg_dir = Path(block["data_dir"]) / block["seg_dir"]
    seg_dir.mkdir(parents=True, exist_ok=True)
    masks = list(output_dir.glob("*.nii.gz"))
    if not masks:
        raise RuntimeError(
            f"No masks found in {output_dir}. Did nnUNet_predict run? "
            "(Use --stage predict --run-predict, or run it manually.)"
        )
    for m in masks:
        shutil.copy(m, seg_dir / m.name)
    print(f"Transferred {len(masks)} masks into {seg_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external-config", required=True, help="YAML with external_eval block.")
    p.add_argument("--stage", choices=["prepare", "predict", "transfer", "all"], default="all")
    p.add_argument("--input-dir", default=None, help="nnUNet input dir (default: data_dir/<name>_nnunet_in).")
    p.add_argument("--output-dir", default=None, help="nnUNet output dir (default: data_dir/<name>_nnunet_out).")
    p.add_argument("--task", default="5", help="nnUNet task id (default: 5 = Task005_Prostate).")
    p.add_argument("--model", default="3d_fullres", help="nnUNet configuration (default: 3d_fullres).")
    p.add_argument("--run-predict", action="store_true", help="Actually invoke nnUNet_predict (needs it on PATH).")
    p.add_argument("--clean", action="store_true", help="Clean the input dir before staging.")
    return p.parse_args()


def main():
    args = parse_args()
    block = load_external_block(args.external_config)
    data_dir = Path(block["data_dir"])
    name = block.get("name", "external")
    input_dir = Path(args.input_dir) if args.input_dir else data_dir / f"{name}_nnunet_in"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / f"{name}_nnunet_out"

    if args.stage in ("prepare", "all"):
        stage_prepare(block, input_dir, args.clean)
    if args.stage in ("predict", "all"):
        stage_predict(input_dir, output_dir, args.task, args.model, args.run_predict)
    if args.stage == "transfer" or (args.stage == "all" and args.run_predict):
        stage_transfer(block, output_dir)


if __name__ == "__main__":
    main()
