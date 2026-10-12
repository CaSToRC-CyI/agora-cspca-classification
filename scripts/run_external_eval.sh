#!/usr/bin/env bash
# Drive external evaluation for all model types for a single seed.
#
# Usage:
#   scripts/run_external_eval.sh <seed> \
#       <cnn3d_run_dir> <resnet3d_run_dir> <radiodino_run_dir> <omnirad_run_dir>
#
# out-dir is built automatically as:  $OUT_BASE/<model_dir>/no_clin_<seed>/
set -euo pipefail

USAGE="usage: $0 <seed> <cnn3d_run_dir> <resnet3d_run_dir> <radiodino_run_dir> <omnirad_run_dir>"
SEED="${1:?$USAGE}"

OUT_BASE="results/external_eval"
EXTERNAL_CONFIG="configs/external_eval_example.yaml"
CLINICAL_MODE="impute"

# model-type  ->  out-dir subfolder name
declare -A OUT_SUBDIR=(
    [cnn3d]=cnn
    [resnet3d]=resnet
    [radiodino]=radiodino
    [omnirad]=omnirad
)

# Per-model training run dirs (metrics/<group>/<run_id>)
declare -A RUN_DIRS=(
    [cnn3d]="${2:?$USAGE}"
    [resnet3d]="${3:?$USAGE}"
    [radiodino]="${4:?$USAGE}"
    [omnirad]="${5:?$USAGE}"
)

for MODEL in cnn3d resnet3d radiodino omnirad; do
    OUT_DIR="$OUT_BASE/${OUT_SUBDIR[$MODEL]}/no_clin_${SEED}"
    echo "=== $MODEL  (seed $SEED) -> $OUT_DIR ==="
    uv run python src/agora/scripts/evaluate_external.py \
        --run-dir "${RUN_DIRS[$MODEL]}" \
        --model-type "$MODEL" \
        --external-config "$EXTERNAL_CONFIG" \
        --out-dir "$OUT_DIR" \
        --clinical-mode "$CLINICAL_MODE"
done
