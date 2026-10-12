# agoracls: csPCa classification from biparametric MRI

Code for the paper *[title to be added]*. It trains and evaluates models that classify
clinically significant prostate cancer (csPCa) from biparametric MRI (T2W, ADC, high-b DWI),
optionally fused with clinical variables (age, PSA, PSA density, prostate volume).

Four architectures are compared:

| Key | Model | Image encoder |
|---|---|---|
| `cnn3d` | 3D CNN | Per-modality shallow 3D conv blocks, then a shared 3D CNN |
| `resnet3d` | 3D ResNet | Per-modality 3D ResNet blocks, then a shared 3D ResNet |
| `omnirad` | OmniRad | 2D ViT ([OmniRad-base](https://huggingface.co/Snarcy/OmniRad-base)) applied slice by slice, aggregated with a 3D conv stack |
| `radiodino` | RadioDino | Same as `omnirad` with a [RadioDino-b16](https://huggingface.co/Snarcy/RadioDino-b16) backbone |

Clinical variables pass through a small MLP and are fused with the image features by
cross-attention (`att_fusion: true`) or concatenation.

## Repository layout

```
configs/      Training, tuning and evaluation YAML configs
notebooks/    TPR/FPR table, combined ROC curves, ablation statistics (with outputs)
scripts/      Shell drivers (multi-seed training)
src/agora/
  data/       Dataset, caching, preprocessing (resampling)
  models/     3D CNN, 3D ResNet, OmniRad/RadioDino models; fine-tuning helpers
  training/   Configs, trainer, learning-curve trainer, Ray Tune search
  scripts/    Entry points for training, evaluation and analysis
tests/        Ordinal-analysis checks
```

All paths in the configs and scripts are relative to the repository root: inputs live
under `data/` and everything produced under `results/`. Both are git-ignored.

## Installation

Requirements: Linux, an NVIDIA GPU with CUDA 12 (the data pipeline uses CuPy), and
[uv](https://docs.astral.sh/uv/). Tested with Python 3.10.

```bash
git clone https://github.com/CaSToRC-CyI/agora-cspca-classification.git
cd agora-cspca-classification
uv sync --python 3.10 --locked
```

`uv.lock` pins every package. PyTorch is installed from the CUDA 12.8 wheel index.
OmniRad and RadioDino weights are downloaded from the Hugging Face Hub on first use.

Runs are logged to MLflow. Without further setup they go to a local `./mlruns`
directory; set `MLFLOW_TRACKING_URI` to use a tracking server.

## Data

| Cohort | Role | Access |
|---|---|---|
| [PI-CAI](https://pi-cai.grand-challenge.org/) | Training / validation / test | Public |
| In-house cohort (`GMI` in the paper, `goc` in the code) | Training / validation / test | Not public; see [Data availability](#data-availability) |

### Expected layout

```
data/
  picai_pre/input_resampled_v2/          PI-CAI volumes: <study>_t2w.mha, <study>_adc.mha, <study>_hbv.mha
  GOC_Data_pre/all_resampled_v4/         In-house volumes: <study>_0000.nii.gz (T2), _0001 (ADC), _0003 (DWI)
  combined_segmentations_v4/             Prostate masks: <study>.nii.gz
  combined_lesion_allds_gs.csv           Labels and clinical variables for all studies
  GOC_Data/anonymid_studyid_map_v4.json  In-house filename study ID -> CSV study_id
  preprocessed_cache/                    HDF5 caches, created on first run
```

- **Volumes** are resampled to a common spacing and centre-cropped/padded with
  `agora.data.resampling.MRIStudy`. `src/agora/scripts/prepare_external_data.py` runs
  the same resampling for a new cohort and can be used as a template.
- **Prostate masks** come from the nnU-Net v1 `Task005_Prostate` model (`3d_fullres`).
  `src/agora/scripts/prepare_external_segmentations.py` stages inputs and collects masks.
- **Label CSV** is indexed by `study_id` and needs `Lesion_gs` (comma-separated Gleason
  scores per lesion, e.g. `3+4,4+3`; the highest is used) and the clinical columns
  `patient_age`, `psa`, `psad`, `prostate_volume`.
- **Labels**: a study is csPCa when its highest Gleason sum reaches `cspca_threshold`.
  7.0 (GS7, ISUP ≥ 2) is the main setting; 6.0 (GS6) is the secondary one. 4+3 counts as 7.5.

On first run the dataset builds an HDF5 cache in `preprocessed_cache/` (Z-normalisation,
crop/pad to `target_shape`); later runs read from it.

## Training

```bash
# One run
uv run python src/agora/scripts/train_v2.py \
    --config configs/train_config_cnn3d.yaml --model-type cnn3d --seed 7

# One run group over the paper's seeds; logs go to results/logs/<group>_seed<s>_run<i>.log
scripts/train_seeds.sh cnn3d cnn_v0 configs/train_config_cnn3d.yaml 7 8 9 10 11
```

`--model-type` is one of `cnn3d`, `resnet3d`, `omnirad`, `radiodino`. Checkpoints and
per-run metrics go to `results/training_outputs/{checkpoints,metrics}/`. The analysis
scripts find runs through the log files, so use `scripts/train_seeds.sh` (or the same
log naming) for anything you want to aggregate.

### Run groups in the paper

Every group starts from `configs/train_config_<model>.yaml`. Copy the config and apply
the overrides below.

| Group(s) | Overrides | Seeds | Used for |
|---|---|---|---|
| `cnn_v0`, `resnet_v0`, `omnirad_v5_flat`, `radiodino_v5_flat` | none | 7–11 | Main results (GS7) |
| `cnn_v0_thr6`, `resnet_v0_thr6`, `omnirad_v5_thr6`, `radiodino_v5_thr6` | `cspca_threshold: 6.0` | 7–11 | Main results (GS6) |
| `cnn_v0_gap`, `resnet_v0_gap`, `omnirad_v5`, `radiodino_v5` | `img_pooling: gap`, `warmup_frac: 0.1` | 42–46 | Flatten vs GAP ablation |
| `omnirad_v5_cls`, `radiodino_v5_cls` | `use_dense_features: false` | 42–46 | CLS token vs dense features |
| `omnirad_v0_flat`, `radiodino_v0_flat` | `finetune_t: v0` | 42–46 | Transfer-learning strategy |
| `omnirad_v2_flat`, `radiodino_v2_flat` | `finetune_t: v2` | 42–46 | Transfer-learning strategy |

`finetune_t` selects which encoder layers are trained: `v0` frozen, `v2` last blocks plus
embeddings, `v5` all (see `FINETUNE_MAP` in `src/agora/models/utils.py`).

### Hyperparameter search

```bash
uv run python src/agora/training/param_tuning.py \
    --config configs/tune_config_omnirad.yaml --trainer omnirad --num-samples 50
```

Ray Tune with the ASHA scheduler; start a Ray cluster first (`ray start --head`).

## Reproducing the paper

Run these from the repository root after training the groups above.

| Result | Command |
|---|---|
| Main performance table | `uv run python src/agora/scripts/compare_models.py --mode logs --logs-dir results/logs --include '^((cnn\|resnet)_v0\|(omnirad\|radiodino)_v5_flat\|(cnn\|resnet)_v0_thr6\|(omnirad\|radiodino)_v5_thr6)$' --seeds 7 8 9 10 11 -o results/main_table` |
| TPR/FPR table | `notebooks/tpr_fpr_table.ipynb` |
| ROC curves | `uv run python src/agora/scripts/generate_roc_auc_pn_ratios.py --mode logs --logs-dir results/logs --curve-set all -o results/roc_jsons`, then `notebooks/vis_roc_combined.ipynb` |
| Ablation statistics (flatten/GAP, CLS/dense, fine-tuning) | `notebooks/stat_analysis.ipynb` |
| Learning curves | `uv run python src/agora/scripts/run_learning_curve.py --config configs/train_config_<model>.yaml --model-type <model> --seed 7` |
| Ordinal (ISUP grade) analysis | `uv run python src/agora/scripts/ordinal_analysis.py` |

The notebooks read from `../results/` and run from the `notebooks/` directory.

### ROI shape analysis

Prostate ROI shape statistics per cohort are computed with the scripts in
`src/agora/scripts/analysis/`, e.g.

```bash
uv run python src/agora/scripts/analysis/roi_shape_analysis.py \
    --input_dir data/combined_segmentations_v4 --output_dir results/roi_shape --total_tasks 1
```

The scripts split the work into chunks for SLURM job arrays (chunk index from
`SLURM_ARRAY_TASK_ID`); `--total_tasks 1` processes everything in one go.

## Data availability

PI-CAI is publicly available from the link above. The in-house cohort is not publicly
available; it can be requested from the corresponding author with reasonable
justification, subject to ethics approval.

## Citation

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

## License

BSD 3-Clause; see [LICENSE](LICENSE). The license covers the code only, not the data.
