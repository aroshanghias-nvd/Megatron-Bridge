#!/bin/bash
# =============================================================================
# Submit MiMo PP2 LLM-only e2e validation for PP fix.
#
# Runs tests/e2e/mimo/test_mimo_training_e2e.py with:
#   LLM:    TP=1, PP=2, DP=2, offset=0
#   Vision: TP=1, PP=1, DP=4, offset=4
#
# Logs are written to ${MBRIDGE}/job_logs/slurm/.
#
# Usage:
#   MBRIDGE=/path/to/Megatron-Bridge bash submit_mimo_pp2_fix_e2e.sh
#   MBRIDGE=/path/to/Megatron-Bridge PARTITION=batch_short bash submit_mimo_pp2_fix_e2e.sh
# =============================================================================

set -euo pipefail

MBRIDGE=${MBRIDGE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SCRIPT_NAME="$(basename "$0" .sh)"
JOB_NAME="${SCRIPT_NAME#submit_}"
LOG_ROOT=${LOG_ROOT:-${MBRIDGE}/job_logs}
SLURM_LOG_DIR="${LOG_ROOT}/slurm"

mkdir -p "${SLURM_LOG_DIR}"

NUM_GPUS=${NUM_GPUS:-8}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_genai}
PARTITION=${PARTITION:-batch}
TIME_LIMIT=${TIME_LIMIT:-00:20:00}

if [ -f "${MBRIDGE}/container-name.txt" ]; then
    export CONTAINER
    CONTAINER=$(tr -d '[:space:]' < "${MBRIDGE}/container-name.txt")
else
    export CONTAINER
    CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/containers/mcore_ci_dev_39951663.sqsh"
fi

export MOUNTS="/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1"

export COMMAND="\
export PYTHONPATH=${MBRIDGE}/src:${MBRIDGE}/3rdparty/Megatron-LM && \
cd ${MBRIDGE} && \
echo RUN_SHA=\$(git rev-parse --short HEAD) && \
bash tests/e2e/mimo/run_mimo_parallelism_tests.sh --gpus ${NUM_GPUS} --config pp2_llm_only"

cd "${MBRIDGE}"

JOB_ID=$(sbatch \
    --nodes=1 \
    --account="${ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --gres="gpu:${NUM_GPUS}" \
    --output="${SLURM_LOG_DIR}/%x-%j.out" \
    --error="${SLURM_LOG_DIR}/%x-%j.err" \
    --parsable \
    mimo_e2e.sub)

RESOLVED_SLURM_OUT="$(readlink -f "${SLURM_LOG_DIR}")/${JOB_NAME}-${JOB_ID}.out"
RESOLVED_SLURM_ERR="$(readlink -f "${SLURM_LOG_DIR}")/${JOB_NAME}-${JOB_ID}.err"

echo "Job submitted: ${JOB_NAME}  (job ${JOB_ID})"
echo "  Account:   ${ACCOUNT}"
echo "  Partition: ${PARTITION}"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Container: ${CONTAINER}"
echo "  Time:      ${TIME_LIMIT}"
echo "  Config:    pp2_llm_only (env-driven)"
echo ""
echo "  Stdout: ${RESOLVED_SLURM_OUT}"
echo "  Stderr: ${RESOLVED_SLURM_ERR}"
echo ""
echo "  tail -f ${RESOLVED_SLURM_OUT}"
echo ""
echo "Check status: squeue -j ${JOB_ID}"
