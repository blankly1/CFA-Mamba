#!/usr/bin/env bash
set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT}"
export TRAIN_ENTRY=train_cg_mamba_no_adapter.py
export RUN_LABEL=output_only
bash "$REPO_ROOT/scripts/reproduce_officehome.sh"
