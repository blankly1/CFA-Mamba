#!/usr/bin/env bash
set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${SOURCE_MODEL_ROOT:?Set SOURCE_MODEL_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

DOMAINS=(Art Clipart Product RealWorld)

for SEED in 0 1 2; do
  for SOURCE_ENV in 0 1 2 3; do
    for TARGET_ENV in 0 1 2 3; do
      [ "$SOURCE_ENV" -ne "$TARGET_ENV" ] || continue

      export DATASET=office-home SOURCE_ENV TARGET_ENV SEED
      export TARGET_NAME=${DOMAINS[$TARGET_ENV]}
      export ADAPTER_TYPE=scan EPOCHS=15 LR=3e-5
      export ADAPTER_LR_FACTOR=1.0
      export CLIP_CONF_THRESHOLD=0.78
      export LAMBDA_PSEUDO=0.5 LAMBDA_CONSISTENCY=0.3
      export LAMBDA_ENTROPY=0.1 LAMBDA_DIVERSITY=1.0
      export LAMBDA_ANCHOR=0.1
      export CLIP_WARMUP_EPOCHS=0 CLIP_MIN_WEIGHT=1.0

      echo "[START] seed=$SEED ${DOMAINS[$SOURCE_ENV]}->${DOMAINS[$TARGET_ENV]}"
      bash "$REPO_ROOT/scripts/train_target.sh"
      echo "[COMPLETE] seed=$SEED ${DOMAINS[$SOURCE_ENV]}->${DOMAINS[$TARGET_ENV]}"
    done
  done
done
