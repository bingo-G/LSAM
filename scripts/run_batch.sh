#!/usr/bin/env bash
# =============================================================================
#  run_batch.sh — batch-mode driver for LSAM inference (manifest mode)
# -----------------------------------------------------------------------------
#  Runs `infer.py` with sensible batch defaults over a manifest CSV.
#
#  Usage:
#      bash scripts/run_batch.sh [CKPT] [MANIFEST] [OUTPUT_CSV]
#
#  Positional args (all optional — defaults shown in brackets):
#      CKPT        [weights/lsam.pth]          path to the LSAM checkpoint
#      MANIFEST    [pairs.csv]                 manifest CSV to score
#                                              (see README §2.3 for the schema)
#      OUTPUT_CSV  [outputs/scores.csv]        per-video prediction output
#
#  Environment-variable overrides (optional):
#      GPU=0            which GPU index to use (default 0)
#      BATCH_SIZE=4     DataLoader batch size (default 4)
#      NUM_WORKERS=4    DataLoader worker count (default 4)
#
#  Example:
#      GPU=1 BATCH_SIZE=8 bash scripts/run_batch.sh \
#          weights/lsam.pth my_pairs.csv outputs/my_scores.csv
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-weights/lsam.pth}"
MANIFEST="${2:-pairs.csv}"
OUTPUT="${3:-outputs/scores.csv}"

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: manifest not found: ${MANIFEST}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

echo "=============================================================="
echo "  LSAM batch inference"
echo "    Checkpoint : ${CKPT}"
echo "    Manifest   : ${MANIFEST}"
echo "    Output     : ${OUTPUT}"
echo "    GPU        : ${GPU}"
echo "    Batch size : ${BATCH_SIZE}"
echo "    Workers    : ${NUM_WORKERS}"
echo "=============================================================="

CUDA_VISIBLE_DEVICES="${GPU}" python -B -u infer.py \
    --eval_ckpt   "${CKPT}" \
    --manifest    "${MANIFEST}" \
    --output      "${OUTPUT}" \
    --batch_size  "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
