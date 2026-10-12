#!/usr/bin/env bash
# Train one run group over several seeds, one train_v2.py run per seed.
#
# Usage:
#   scripts/train_seeds.sh <model-type> <group> <config> <seed> [<seed> ...]
#
# Example (headline CNN3D, GS7):
#   scripts/train_seeds.sh cnn3d cnn_v0 configs/train_config_cnn3d.yaml 7 8 9 10 11
#
# Each run's stdout is saved as results/logs/<group>_seed<seed>_run<i>.log, where i
# is the seed's 1-based position in the list. The analysis scripts and notebooks
# discover runs by this name, so keep it when reproducing the paper's tables.
set -euo pipefail

USAGE="usage: $0 <model-type> <group> <config> <seed> [<seed> ...]"
MODEL_TYPE="${1:?$USAGE}"
GROUP="${2:?$USAGE}"
CONFIG="${3:?$USAGE}"
shift 3
[ "$#" -ge 1 ] || { echo "$USAGE" >&2; exit 1; }

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

RUN=0
for SEED in "$@"; do
    RUN=$((RUN + 1))
    LOG_FILE="$LOG_DIR/${GROUP}_seed${SEED}_run${RUN}.log"
    echo "=== $GROUP  seed $SEED (run $RUN/$#) -> $LOG_FILE"
    uv run python src/agora/scripts/train_v2.py \
        --config "$CONFIG" \
        --model-type "$MODEL_TYPE" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_FILE"
done
