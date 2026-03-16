#!/bin/bash
# =============================================================================
# Submit MIMO checkpoint save→resume round-trip e2e test via sbatch.
#
# Runs 4 parallelism configs (DP, TP, TP+DP, PP+DP) x 2 phases (save, resume)
# on 8 GPUs. Logs are written to ${MBRIDGE}/job_logs/slurm/.
#
# Usage (from Mac via SSH):
#   ssh dfw bash -l <<'EOF'
#     cd /lustre/fs1/portfolios/coreai/users/aroshanghias/Megatron-Bridge
#     bash tests/e2e/mimo/submit_mimo_checkpoint_resume.sh
#   EOF
#
# Or directly on the cluster:
#   cd /path/to/Megatron-Bridge
#   bash tests/e2e/mimo/submit_mimo_checkpoint_resume.sh
# =============================================================================

set -euo pipefail

MBRIDGE=${MBRIDGE:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)}
SCRIPT_NAME="$(basename "$0" .sh)"
JOB_NAME="${SCRIPT_NAME#submit_}"
LOG_ROOT=${LOG_ROOT:-${MBRIDGE}/job_logs}
SLURM_LOG_DIR="${LOG_ROOT}/slurm"

mkdir -p "${SLURM_LOG_DIR}"

ACCOUNT=${ACCOUNT:-coreai_dlalgo_genai}
PARTITION=${PARTITION:-batch}
NUM_GPUS=${NUM_GPUS:-8}
TIME_LIMIT=${TIME_LIMIT:-00:30:00}

if [ -f "${MBRIDGE}/container-name.txt" ]; then
    CONTAINER=$(tr -d '[:space:]' < "${MBRIDGE}/container-name.txt")
else
    CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/containers/mcore_ci_dev_39951663.sqsh"
fi

JOB_ID=$(sbatch \
    --nodes=1 \
    --account="${ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --gres="gpu:${NUM_GPUS}" \
    --no-container-mount-home \
    --container-image="${CONTAINER}" \
    --container-mounts="/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1" \
    --output="${SLURM_LOG_DIR}/%x-%j.out" \
    --error="${SLURM_LOG_DIR}/%x-%j.err" \
    --parsable \
    --wrap "
set -euo pipefail
export PYTHONPATH=\"${MBRIDGE}/src:${MBRIDGE}/3rdparty/Megatron-LM\"
cd \"${MBRIDGE}\"
echo \"RUN_SHA=\$(git rev-parse --short HEAD)\"
bash tests/e2e/mimo/run_mimo_checkpoint_resume.sh --gpus ${NUM_GPUS}
")

RESOLVED_SLURM_OUT="$(readlink -f "${SLURM_LOG_DIR}")/${JOB_NAME}-${JOB_ID}.out"
RESOLVED_SLURM_ERR="$(readlink -f "${SLURM_LOG_DIR}")/${JOB_NAME}-${JOB_ID}.err"

echo "Job submitted: ${JOB_NAME}  (job ${JOB_ID})"
echo "  Account:   ${ACCOUNT}"
echo "  Partition: ${PARTITION}"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Container: ${CONTAINER}"
echo "  Time:      ${TIME_LIMIT}"
echo ""
echo "  Stdout: ${RESOLVED_SLURM_OUT}"
echo "  Stderr: ${RESOLVED_SLURM_ERR}"
echo ""
echo "  tail -f ${RESOLVED_SLURM_OUT}"
echo ""
echo "Check status: squeue -j ${JOB_ID}"
