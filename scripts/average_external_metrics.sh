#!/usr/bin/env bash
# Aggregate metrics.json across no_clin_<seed> dirs, per model.
#
# For every model folder under BASE_DIR that contains no_clin_<seed> subdirs,
# writes <model>/no_clin_avg.json with mean + std (ddof=1) of the metrics.
#
# Usage:
#   scripts/average_external_metrics.sh [BASE_DIR] [GLOB]
#     BASE_DIR  default: results/external_eval
#     GLOB      default: no_clin_*   (subdir pattern to aggregate)
set -euo pipefail

BASE_DIR="${1:-results/external_eval}"
GLOB="${2:-no_clin_*}"
OUT_NAME="$(echo "$GLOB" | tr -d '*')avg.json"   # e.g. no_clin_ -> no_clin_avg.json

# Find every parent dir that holds matching metrics.json files, uniquely.
mapfile -t MODEL_DIRS < <(
    find "$BASE_DIR" -type d -name "$GLOB" -exec test -f '{}/metrics.json' ';' -print \
        | xargs -r -n1 dirname | sort -u
)

if [ "${#MODEL_DIRS[@]}" -eq 0 ]; then
    echo "No '$GLOB/metrics.json' found under $BASE_DIR" >&2
    exit 1
fi

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    OUT_FILE="$MODEL_DIR/$OUT_NAME"
    uv run python - "$MODEL_DIR" "$GLOB" "$OUT_FILE" <<'PY'
import glob, json, os, sys
import numpy as np

model_dir, pattern, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
shared_keys = ["n", "n_positive", "dataset"]

files = sorted(glob.glob(os.path.join(model_dir, pattern, "metrics.json")))
data = [json.load(open(f)) for f in files]

out = {"n_seeds": len(data), "seed_dirs": [os.path.basename(os.path.dirname(f)) for f in files]}
for k in shared_keys:
    vals = {d.get(k) for d in data}
    out[k] = vals.pop() if len(vals) == 1 else sorted(vals)  # flag if they differ

for k in metric_keys:
    arr = np.array([d[k] for d in data], dtype=float)
    out[f"{k}_mean"] = float(arr.mean())
    out[f"{k}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

with open(out_file, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"{os.path.relpath(model_dir)}: {out['n_seeds']} seeds -> {out_file}")
PY
done
