#!/bin/bash
# Heterogeneous MIMO LLaVA training — LLM on ranks 0-3, CLIP on ranks 4-7.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/test_mimo_training_llava.py"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"
TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.tmp}"
mkdir -p "$TMPDIR"
export TMPDIR
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

python -m torch.distributed.run \
    --nproc_per_node "${NUM_GPUS}" \
    --nnodes 1 \
    --master_port "$MASTER_PORT" \
    "${TEST_FILE}" \
    --micro-batch-size 2 \
    --global-batch-size 32 \
    --train-iters 500 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --lr 1e-4 \
    --lr-warmup-iters 20 \
    --min-lr 2.0e-5 \
    --weight-decay 0.01 \
    --wandb-project "Megatron-Bridge-MIMO" \
    --wandb-exp-name "mimo-llava-e2e-test" \
    --wandb-save-dir "/tmp/wandb" \
    --dataset-root /lustre/fsw/portfolios/coreai/users/kjafarisadeg/nemo_workspace/workspace/datasets/LLaVA-Pretrain/
