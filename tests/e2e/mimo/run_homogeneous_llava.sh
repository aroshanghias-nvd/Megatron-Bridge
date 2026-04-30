#!/bin/bash
# Homogeneous MIMO LLaVA training — all modules on every rank (TP=4, DP=1).
# 4 GPUs: every rank runs both LLM and CLIP ViT encoder together.

GPUS_PER_NODE=4 # Deliberately set to 4 for homogeneous test, even if the machine has 8 GPUs to get DP=1 and TP=4.
NUM_NODES=1

# Set DETERMINISTIC=1 to export deterministic NCCL/CUBLAS/cuDNN/TE env vars
# AND pass --deterministic to the training script (FP32, unfused attention, etc.).
# Also disables gradient clipping (clip-grad=0.0), which is non-associative under
# distributed reductions and introduces run-to-run variance.
DETERMINISTIC=${DETERMINISTIC:-0}
DETERMINISTIC_FLAG=""
EXP_SUFFIX=""
CLIP_GRAD=1.0
if [[ "${DETERMINISTIC}" == "1" ]]; then
    DETERMINISTIC_FLAG="--deterministic"
    EXP_SUFFIX="-fp32"
    CLIP_GRAD=0.0
    # Pin Ring algorithm for deterministic reduction order.
    # Tree is faster for some message sizes but NCCL 2.28 Tree doesn't support
    # AllGather with Int8 (used by torch.distributed.all_gather_object), and
    # letting NCCL choose per-operation (^NVLS) still leaves Tree/Ring selection
    # non-deterministic.  Ring supports all collective ops.
    export NCCL_ALGO=Ring
    export NCCL_PROTO=Simple
    # Disable NCCL's topology-aware optimizations that can change paths between runs
    export NCCL_TUNER_PLUGIN=""
    # For full CUDA-level determinism
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    # Force deterministic cuDNN attention (disable non-deterministic workspace)
    export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
    # Required by Transformer Engine when deterministic_mode=True
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
fi

uv run torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NUM_NODES" \
    tests/e2e/mimo/test_mimo_training_llava_homogeneous.py \
    --micro-batch-size 4 \
    --global-batch-size 96 \
    --train-iters 100 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad ${CLIP_GRAD} \
    --log-interval 1 \
    --lr 1e-3 \
    --lr-warmup-iters 60 \
    --min-lr 2.0e-5 \
    --weight-decay 0.0 \
    --wandb-project "Megatron-Bridge-MIMO" \
    --wandb-exp-name "mimo-llava-homo-e2e-test${EXP_SUFFIX}" \
    --wandb-save-dir "/tmp/wandb" \
    --vision-encoder-checkpoint /path/to/clip_checkpoint \
    --language-model-checkpoint /path/to/llm_checkpoint \
    --dataset-root /path/to/LLaVA-Pretrain/
    ${DETERMINISTIC_FLAG} \
