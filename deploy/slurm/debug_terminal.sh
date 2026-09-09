#!/bin/bash
echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
 
REPO_NAME=unidepthlss-dev
IMAGE_NAME=unidepthlss
PATH_TO_SOURCE_CODE=/raid/${USER}/Workspace/${REPO_NAME}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${IMAGE_NAME}_v1.sqsh

# Same name as the Makefile + README. This repo needs exactly one dataset. Unlike
# train_slurm.sh/pull_and_install.sh (submitted via sbatch, so their whole script runs on the
# allocated compute node), this script runs its own preamble on the *login* node and only hands
# off to a compute node at the final srun --pty -- so a mount source must stay a compute-node
# path even though this preamble can only see the login-node view. There is nothing optional to
# probe here, so the mount is unconditional and no login-node check is needed. Override by
# exporting NUSCENES_DATA_ROOT before running this script.
NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-/raid/${USER}/Datasets/nuscenes}"

MOUNTS="${PATH_TO_SOURCE_CODE}:/workspace"
MOUNTS="${MOUNTS},${NUSCENES_DATA_ROOT}:/data/nuscenes"
MOUNTS="${MOUNTS},/dev/dri:/dev/dri"

srun \
    --partition=H200 \
    --gres=gpu:h200:1 \
    --mem=256G \
    --cpus-per-task=16 \
    --time=23:59:59 \
    --job-name="${IMAGE_NAME}-terminal" \
    --container-image="$OUTPUT_SQSH" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --pty bash -c "source /workspace/deploy/docker/entrypoint.sh && bash"