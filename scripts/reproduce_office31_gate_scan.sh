#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

SOURCES=(amazon amazon dslr dslr webcam webcam)
SOURCE_IDS=(0 0 1 1 2 2)
TARGETS=(dslr webcam amazon webcam amazon dslr)
TARGET_IDS=(1 2 0 2 0 1)

PROTOCOL=$OUTPUT_ROOT/protocol.tsv
mkdir -p "$OUTPUT_ROOT"
printf "order\tadapter\tseed\tsource\ttarget\tepochs\tselection\n" > "$PROTOCOL"
ORDER=0

for SEED in 0 1 2; do
  for INDEX in 0 1 2 3 4 5; do
    if [ $(((SEED + INDEX) % 2)) -eq 0 ]; then
      ADAPTERS=(scan gate)
    else
      ADAPTERS=(gate scan)
    fi

    for ADAPTER_TYPE in "${ADAPTERS[@]}"; do
      ORDER=$((ORDER + 1))
      export DATASET=office
      export SOURCE_ENV=${SOURCE_IDS[$INDEX]}
      export TARGET_ENV=${TARGET_IDS[$INDEX]}
      export SOURCE_NAME=${SOURCES[$INDEX]}
      export TARGET_NAME=${TARGETS[$INDEX]}
      export ADAPTER_TYPE SEED
      export EPOCHS=15

      printf "%s\t%s\t%s\t%s\t%s\t15\tfixed_endpoint\n" \
        "$ORDER" "$ADAPTER_TYPE" "$SEED" "$SOURCE_NAME" "$TARGET_NAME" \
        >> "$PROTOCOL"

      echo "[START] $ADAPTER_TYPE seed=$SEED $SOURCE_NAME->$TARGET_NAME"
      bash "$REPO_ROOT/scripts/train_target.sh"
      echo "[COMPLETE] $ADAPTER_TYPE seed=$SEED $SOURCE_NAME->$TARGET_NAME"
    done
  done
done

echo "[ALL COMPLETE] 36/36"
