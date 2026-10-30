#!/usr/bin/env bash
set -Eeuo pipefail

MODE="$1"
EXP="$2"
LOG="$3"

ROOT="$HOME/robotics_thesis/forcevla_lab"
PY="$HOME/miniforge3/envs/forcevla_lab_tf_test/bin/python"
SCRIPT="$ROOT/scripts/forcevla_lora_first_run_v1.py"

export PYTHONPATH="$ROOT/ForceVLA_thesis/src:$ROOT/ForceVLA_thesis/lerobot:$ROOT/ForceVLA_thesis/dlimp"

export HF_LEROBOT_HOME="/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset"
export HF_DATASETS_CACHE="$HF_LEROBOT_HOME/.forcevla_hf_datasets_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

unset JAX_PLATFORMS

cd "$ROOT/ForceVLA_thesis"

echo "===== FORCEVLA LIVE TRAINING ====="
echo "MODE=$MODE"
echo "EXPERIMENT=$EXP"
echo "LOG=$LOG"

if "$PY" -u "$SCRIPT" "$MODE" "$EXP" 2>&1 \
    | tee "$LOG" \
    | grep --line-buffered -E \
      '^(TRAINING_MODE=|TRAINING=|TRAINING_COMPLETED=|Step [0-9]+:|RESUME_FROM_SAVED_STEP=|ALREADY_FINISHED_)|Restoring checkpoint|Saving checkpoint at step|Finished saving checkpoint|Traceback|Error|RESOURCE_EXHAUSTED|out of memory'
then
    echo "RUN_EXIT_STATUS=PASS"
else
    echo "RUN_EXIT_STATUS=FAILED"
    tail -35 "$LOG"
    exit 1
fi

echo "FULL_LOG=$LOG"
echo "########################################################################################################"
