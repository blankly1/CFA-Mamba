#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${SOURCE_MODEL_ROOT:?Set SOURCE_MODEL_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

DATASET=${DATASET:-office}
SOURCE_ENV=${SOURCE_ENV:-0}
TARGET_ENV=${TARGET_ENV:-1}
TARGET_NAME=${TARGET_NAME:-dslr}
ADAPTER_TYPE=${ADAPTER_TYPE:-scan}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-15}
LR=${LR:-3e-5}
LR_FACTOR1=${LR_FACTOR1:-0.1}
LR_FACTOR2=${LR_FACTOR2:-1.0}
ADAPTER_LR_FACTOR=${ADAPTER_LR_FACTOR:-1.0}

CLIP_PRETRAINED=${CLIP_PRETRAINED:-laion2b_s34b_b88k}
CLIP_CONF_THRESHOLD=${CLIP_CONF_THRESHOLD:-0.78}
CLIP_TEMPERATURE=${CLIP_TEMPERATURE:-0.01}
CLIP_WARMUP_EPOCHS=${CLIP_WARMUP_EPOCHS:-1}
CLIP_MIN_WEIGHT=${CLIP_MIN_WEIGHT:-0.05}
CLIP_NUM_WORKERS=${CLIP_NUM_WORKERS:-0}

LAMBDA_PSEUDO=${LAMBDA_PSEUDO:-0.5}
LAMBDA_CONSISTENCY=${LAMBDA_CONSISTENCY:-0.3}
LAMBDA_ENTROPY=${LAMBDA_ENTROPY:-0.1}
LAMBDA_DIVERSITY=${LAMBDA_DIVERSITY:-1.0}
LAMBDA_ANCHOR=${LAMBDA_ANCHOR:-0.1}

NUM_WORKERS=${NUM_WORKERS:-1}

export PYTHONPATH="$REPO_ROOT/repository:$REPO_ROOT/third_party/selective_scan"
RUN_LABEL=${RUN_LABEL:-$ADAPTER_TYPE}
mkdir -p "$OUTPUT_ROOT/cache" "$OUTPUT_ROOT/results"

python "$REPO_ROOT/repository/cfa_mamba/${TRAIN_ENTRY:-train.py}" \
  --cfg "$REPO_ROOT/repository/configs/vmambav2_small_224.yaml" \
  --data_path "$DATA_ROOT/" \
  --output "$OUTPUT_ROOT/results/$RUN_LABEL/seed$SEED" \
  --dataset "$DATASET" \
  --source_env "$SOURCE_ENV" \
  --target_env "$TARGET_ENV" \
  --adapter_type "$ADAPTER_TYPE" \
  --lr "$LR" \
  --lr_factor1 "$LR_FACTOR1" \
  --lr_factor2 "$LR_FACTOR2" \
  --batch_size 32 \
  --test_batch_size 16 \
  --dg_aug True \
  --neck bn_relu \
  --fusion_module ssm \
  --clip_backend open_clip \
  --clip_model ViT-B-16 \
  --clip_pretrained "$CLIP_PRETRAINED" \
  --clip_temperature "$CLIP_TEMPERATURE" \
  --clip_cache "$OUTPUT_ROOT/cache/${DATASET}_${TARGET_NAME}_ViT-B-16_${CLIP_CACHE_ID:-laion2b_s34b_b88k}_t${CLIP_TEMPERATURE}.pt" \
  --clip_num_workers "$CLIP_NUM_WORKERS" \
  --gate_hidden_dim 0 \
  --gate_dropout 0.0 \
  --gate_lr_factor "$ADAPTER_LR_FACTOR" \
  --clip_conf_threshold "$CLIP_CONF_THRESHOLD" \
  --lambda_pseudo "$LAMBDA_PSEUDO" \
  --lambda_consistency "$LAMBDA_CONSISTENCY" \
  --lambda_entropy "$LAMBDA_ENTROPY" \
  --lambda_diversity "$LAMBDA_DIVERSITY" \
  --lambda_anchor "$LAMBDA_ANCHOR" \
  --clip_warmup_epochs "$CLIP_WARMUP_EPOCHS" \
  --clip_min_weight "$CLIP_MIN_WEIGHT" \
  MODEL.SOURCE_DIR "$SOURCE_MODEL_ROOT" \
  DATA.NUM_WORKERS "$NUM_WORKERS" \
  TRAIN.EPOCHS "$EPOCHS" \
  TRAIN.WARMUP_EPOCHS 1 \
  SEED "$SEED"
