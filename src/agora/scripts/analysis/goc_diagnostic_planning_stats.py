"""Count GOC patients with Diagnostic / Planning / both MRI studies in the resampled set.

The resampled filenames are `{PatientID}-{StudyID}_000{modality}.nii.gz` and carry no record
of whether the series came from `Diagnostic_MRI` or `Planning_MRI`. That link is recovered by
re-walking the original DICOM tree, reading the header of every file, and rebuilding the same
study key used when the NIfTIs were written.

A study key is attributed to the folder that supplied every modality present in its filenames
(T2 `_0000`, ADC `_0001`, DWI>b1400 `_0003`). Keys that both folders could have supplied are
reported separately as ambiguous rather than guessed at.

Usage:
    uv run python src/agora/scripts/analysis/goc_diagnostic_planning_stats.py \
        --cache goc_key_index.json --out-csv goc_dx_plan.csv --workers 16
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pydicom
from tqdm import tqdm

DEFAULT_GOC_ROOT = "data/GOC_Data_v2"
DEFAULT_RESAMPLED_DIR = "data/GOC_Data_pre/all_resampled_v4"
DEFAULT_NIFTI_DIR = "data/GOC_Data_nif_v3"

# Directory name under a patient -> source label reported in the stats
MRI_SOURCES = {"Diagnostic_MRI": "diagnostic", "Planning_MRI": "planning"}
SOURCE_LABELS = ("diagnostic", "planning", "unlabelled")

# Series description -> modality code used for the NIfTI conversion
MODALITIES_MAP = {
    # T2 weighted -> 0
    "T2 weighted": 0,
    "Ax T2 FSE": 0,
    "Ax T2 FSE 3mm": 0,
    "t2_tse_tra": 0,
    "t2_tse_tra_p2": 0,
    "t2_tse_tra_p2_352": 0,
    "t2_tse_tra_p3_352": 0,
    # DWI for b > 1400 -> 3
    "DWI for b > 1400": 3,
    "Ax DWI b1400": 3,
    "Ax DWI b1500": 3,
    "Ax FOCUS B1500": 3,
    "ep2d_diff_zoomit_b1500_tra_TRACEW": 3,
    "resolve_diff_b1500_tra_p2_TRACEW": 3,
    "resolve_diff_b1500_tra_p2_new_TRACEW": 3,
    # DWI for b < 1400 -> 2
    "DWI for b < 1400": 2,
    "Ax DWI b0,b800": 2,
    "Ax DWI b50,b1000": 2,
    "Ax FOCUS B-50-800 Synthetic B-1500": 2,
    "ep2d_diff_b50_800_tra_p2_TRACEW": 2,
    "ep2d_diff_zoomit _b0_50_200_400_700_1000_TRACEW": 2,
    "ep2d_diff_zoomit_b50_200_800_TRACEW": 2,
    "resolve_diff_b50_200_800_tra_p2_new_TRACEW": 2,
    "resolve_diff_b50_200_800_tra_p2_ TRACEW": 2,
    # ADC -> 1
    "ADC": 1,
    "ADC (10^-6 mm2/s)": 1,
    "Apparent Diffusion Coefficient (mm2/s)": 1,
    "ep2d_diff_b50_800_tra_p2_ADC": 1,
    "ep2d_diff_zoomit_b0_50_200_400_700_1000_ADC": 1,
    "ep2d_diff_zoomit_b50_200_800_ADC": 1,
    "resolve_diff_b50_200_800_tra_p2_ADC": 1,
    "resolve_diff_b50_200_800_tra_p2_new_ADC": 1,
}
MODALITIES_MAP_NORM = {k.lower().strip(): v for k, v in MODALITIES_MAP.items()}

# Header tags only: PatientID, StudyID, SeriesDescription
_TAGS = [(0x0010, 0x0020), (0x0020, 0x0010), (0x0008, 0x103E)]


def find_series_dirs(patient_dir):
    """Yield (source, dir) for every directory holding DICOM files under a patient.

    Matches the original DICOM conversion: DICOM/NUM or DICOM/NUM/NUM under
    each MRI folder, plus a bare DICOM folder directly under the patient and loose .dcm files
    sitting directly in an MRI folder.
    """
    patient_dir = Path(patient_dir)

    def walk_dicom_root(dicom_root):
        dirs = []
        for num_dir in dicom_root.iterdir():
            if not num_dir.is_dir():
                continue
            sub_dirs = [d for d in num_dir.iterdir() if d.is_dir()]
            dirs.extend(sub_dirs if sub_dirs else [num_dir])
        return dirs

    direct_dicom = patient_dir / "DICOM"
    if direct_dicom.is_dir():
        for d in walk_dicom_root(direct_dicom):
            yield "unlabelled", d

    for dir_name, source in MRI_SOURCES.items():
        mri_dir = patient_dir / dir_name
        if not mri_dir.is_dir():
            continue
        mri_dicom = mri_dir / "DICOM"
        if mri_dicom.is_dir():
            for d in walk_dicom_root(mri_dicom):
                yield source, d
        if any(f.suffix == ".dcm" for f in mri_dir.iterdir() if f.is_file()):
            yield source, mri_dir


def scan_patient(patient_dir):
    """Map (study key, source) -> modality codes found, by reading every DICOM header.

    Every file is read rather than one per directory: some patients keep loose .dcm files from
    two different studies in a single MRI folder, so one header does not describe the folder.
    """
    patient_dir = Path(patient_dir)
    found = defaultdict(lambda: {"modalities": set(), "n_files": 0, "descriptions": set()})
    n_read, n_failed = 0, 0

    try:
        for source, series_dir in find_series_dirs(patient_dir):
            files = [f for f in series_dir.iterdir()
                     if f.is_file() and (not f.suffix or f.suffix == ".dcm")]
            for f in files:
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=_TAGS)
                except Exception:
                    n_failed += 1
                    continue
                n_read += 1
                patient_id = str(ds.get("PatientID", "") or "").strip()
                if not patient_id:
                    continue  # DICOMDIR and similar non-image files
                study_id = str(ds.get("StudyID", "") or "").strip()
                description = str(ds.get("SeriesDescription", "") or "").strip().lower()

                entry = found[(f"{patient_id}-{study_id}", source)]
                entry["n_files"] += 1
                if description in MODALITIES_MAP_NORM:
                    entry["modalities"].add(MODALITIES_MAP_NORM[description])
                    entry["descriptions"].add(description)
    except Exception as exc:
        return {"anonymized_id": patient_dir.name, "records": [], "error": str(exc),
                "n_read": n_read, "n_failed": n_failed}

    records = [
        {"anonymized_id": patient_dir.name, "group": patient_dir.parent.name,
         "study_key": key, "source": source,
         "modalities": sorted(entry["modalities"]), "n_files": entry["n_files"]}
        for (key, source), entry in found.items()
    ]
    return {"anonymized_id": patient_dir.name, "records": records, "error": None,
            "n_read": n_read, "n_failed": n_failed}


def build_key_index(goc_root, workers, cache_path=None):
    """Map study key -> source -> {patients, modalities, n_files}, scanning headers in parallel."""
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached key index from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    patient_dirs = sorted(
        p for group in sorted(Path(goc_root).glob("GROUP_*"))
        for p in group.iterdir() if p.is_dir()
    )
    print(f"Scanning {len(patient_dirs)} patient directories under {goc_root}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    records, errors = [], []
    n_read = n_failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scan_patient, p): p for p in patient_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading DICOM headers"):
            result = future.result()
            n_read += result["n_read"]
            n_failed += result["n_failed"]
            if result["error"]:
                errors.append((result["anonymized_id"], result["error"]))
            records.extend(result["records"])

    print(f"Read {n_read} DICOM headers ({n_failed} unreadable)")
    if errors:
        print(f"{len(errors)} patient(s) failed to scan:")
        for pid, err in errors[:10]:
            print(f"  {pid}: {err}")

    index = defaultdict(dict)
    for rec in records:
        index[rec["study_key"]][rec["source"]] = {
            "patient": rec["anonymized_id"],
            "group": rec["group"],
            "modalities": rec["modalities"],
            "n_files": rec["n_files"],
        }
    index = dict(index)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Wrote key index cache to {cache_path}")

    return index


def collect_study_keys(directory):
    """Study key -> modality codes present, from `{key}_000{code}.nii.gz` filenames."""
    keys = defaultdict(set)
    for f in Path(directory).glob("*.nii.gz"):
        stem = f.name[: -len(".nii.gz")]
        key, _, suffix = stem.rpartition("_")
        if not key or not suffix.isdigit():
            continue
        keys[key].add(int(suffix))
    return dict(keys)


def resolve_source(entry, required_modalities, tiebreak="none"):
    """Pick the folder that supplied this study, or flag it as ambiguous/unresolved."""
    if entry is None:
        return "unresolved", []
    # Only a folder holding every modality in the filenames could have produced them.
    candidates = [src for src, info in entry.items()
                  if required_modalities <= set(info["modalities"])]
    if len(candidates) == 1:
        return candidates[0], candidates
    if len(candidates) > 1:
        if tiebreak in candidates:
            return tiebreak, sorted(candidates)
        return "ambiguous", sorted(candidates)
    # No folder has the full set (e.g. a description not in the modality map); fall back to
    # whichever folder contributed the most eligible modalities.
    ranked = sorted(entry.items(), key=lambda kv: -len(kv[1]["modalities"]))
    if ranked and ranked[0][1]["modalities"]:
        return ranked[0][0], [ranked[0][0]]
    return "unresolved", []


def summarise(study_keys, index, label, out_csv=None, tiebreak="none"):
    """Resolve every study key to a source and aggregate per patient."""
    buckets = SOURCE_LABELS + ("ambiguous",)
    per_patient = defaultdict(lambda: {b: [] for b in buckets})
    unresolved = []
    key_rows = []

    for key in sorted(study_keys):
        entry = index.get(key)
        source, candidates = resolve_source(entry, study_keys[key], tiebreak)
        if source == "unresolved":
            unresolved.append(key)
            continue
        patient = next(iter(entry.values()))["patient"]
        per_patient[patient][source].append(key)
        key_rows.append({"study_key": key, "anonymized_id": patient, "source": source,
                         "candidates": ";".join(candidates)})

    counts = defaultdict(int)
    rows = []
    for patient in sorted(per_patient):
        rec = per_patient[patient]
        has_diag, has_plan = bool(rec["diagnostic"]), bool(rec["planning"])
        if has_diag and has_plan:
            category = "both"
        elif has_diag:
            category = "diagnostic_only"
        elif has_plan:
            category = "planning_only"
        else:
            category = "undetermined"
        counts[category] += 1
        row = {"anonymized_id": patient, "category": category}
        row.update({f"n_{b}": len(rec[b]) for b in buckets})
        row.update({f"{b}_keys": ";".join(rec[b]) for b in buckets})
        rows.append(row)

    n_studies = {b: sum(row[f"n_{b}"] for row in rows) for b in buckets}

    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")
    print(f"Study keys (patient-study pairs): {len(study_keys)}")
    print(f"  attributed to a folder:         {len(study_keys) - len(unresolved)}")
    print(f"  unresolved:                     {len(unresolved)}")
    print("\nStudies by source folder:")
    for b in buckets:
        print(f"  {b:<12} {n_studies[b]}")

    print(f"\nPatients with at least one study: {len(rows)}")
    for category in ("diagnostic_only", "planning_only", "both", "undetermined"):
        if counts[category] or category != "undetermined":
            print(f"  {category:<16} {counts[category]}")
    print(f"  {'total':<16} {len(rows)}")
    print(f"\nPatients with any diagnostic: {counts['diagnostic_only'] + counts['both']}")
    print(f"Patients with any planning:   {counts['planning_only'] + counts['both']}")

    if unresolved:
        print(f"\nUnresolved keys ({len(unresolved)}):")
        for key in unresolved:
            print(f"  {key}  modalities_in_files={sorted(study_keys[key])}  "
                  f"dicom_entry={'missing' if key not in index else index[key]}")

    ambiguous = [r for r in key_rows if r["source"] == "ambiguous"]
    if ambiguous:
        print(f"\nAmbiguous keys ({len(ambiguous)}) — both folders hold the full modality set:")
        for r in ambiguous:
            print(f"  {r['study_key']} ({r['anonymized_id']}): {r['candidates']}")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote per-patient breakdown to {out_csv}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goc-root", default=DEFAULT_GOC_ROOT,
                        help="Root of the original DICOM tree (GROUP_A/B/C)")
    parser.add_argument("--resampled-dir", default=DEFAULT_RESAMPLED_DIR,
                        help="Directory of resampled NIfTI files to analyse")
    parser.add_argument("--nifti-dir", default=DEFAULT_NIFTI_DIR,
                        help="Pre-resampling NIfTI directory, for --compare-stages")
    parser.add_argument("--compare-stages", action="store_true",
                        help="Also report the same stats for the pre-resampling NIfTI set")
    parser.add_argument("--cache", default=None,
                        help="JSON path to cache the study-key index (reused if it exists)")
    parser.add_argument("--out-csv", default=None, help="Per-patient breakdown CSV")
    parser.add_argument("--ambiguous-tiebreak", choices=["none", "planning", "diagnostic"],
                        default="none",
                        help="Assign keys both folders could have produced to this folder. "
                             "`planning` reproduces the notebook, which sorts Diagnostic_MRI "
                             "before Planning_MRI and lets the later write win.")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    index = build_key_index(args.goc_root, args.workers, args.cache)
    print(f"Indexed {len(index)} distinct study keys from the DICOM tree")

    resampled = collect_study_keys(args.resampled_dir)
    summarise(resampled, index, f"RESAMPLED SET: {args.resampled_dir}",
              args.out_csv, args.ambiguous_tiebreak)

    if args.compare_stages:
        nifti = collect_study_keys(args.nifti_dir)
        summarise(nifti, index, f"NIFTI SET (pre-resampling): {args.nifti_dir}",
                  tiebreak=args.ambiguous_tiebreak)


if __name__ == "__main__":
    main()
