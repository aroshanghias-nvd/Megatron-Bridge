#!/bin/bash
# Homogeneous MIMO LLaVA training — all modules on every rank (TP=4, DP=1).
# 4 GPUs: every rank runs both LLM and CLIP ViT encoder together.

GPUS_PER_NODE=4 # Deliberately set to 4 for homogeneous test, even if the machine has 8 GPUs to get DP=1 and TP=4.
NUM_NODES=1

uv run torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NUM_NODES" \
    tests/e2e/mimo/test_mimo_training_llava_homogeneous.py \
    --micro-batch-size 4 \
    --global-batch-size 128 \
    --train-iters 1000 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --lr 1e-3 \
    --lr-warmup-iters 60 \
    --min-lr 2.0e-5 \
    --weight-decay 0.0 \
    --wandb-project "Megatron-Bridge-MIMO" \
    --wandb-exp-name "mimo-llava-homo-e2e-test" \
    --wandb-save-dir "/tmp/wandb" \
    --vision-encoder-checkpoint /path/to/clip_checkpoint \
    --language-model-checkpoint /path/to/llm_checkpoint \
    --dataset-root /path/to/LLaVA-Pretrain/