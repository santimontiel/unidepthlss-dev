#!/bin/bash
# job-name is a static #SBATCH directive (sbatch parses it before any shell code runs, so it
# can't reference $IMAGE_NAME below) -- keep it in sync by hand if IMAGE_NAME ever changes.
#SBATCH --job-name=unidepthlss-train
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:2
#SBATCH --time=2-23:59:59
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_NAME=unidepthlss-dev
IMAGE_NAME=unidepthlss
PATH_TO_SOURCE_CODE=/raid/${USER}/Workspace/${REPO_NAME}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${IMAGE_NAME}_v1.sqsh

# Same name as the Makefile. This repo needs exactly one dataset. Override by exporting it
# before submitting (sbatch forwards the submitting shell's environment by default).
NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-/raid/${USER}/Datasets/nuscenes}"

if [ ! -f "$OUTPUT_SQSH" ]; then
  echo "Error: $OUTPUT_SQSH not found. Please create the container image first."
  exit 1
else
  echo "Found squashfile at $OUTPUT_SQSH."
fi

if [ ! -d "$PATH_TO_SOURCE_CODE/.venv" ]; then
  echo "Error: Virtual environment not found at $PATH_TO_SOURCE_CODE/.venv. Please run the installation script first."
  exit 1
else
  echo "Found virtual environment in source code directory."
fi

MOUNTS="${PATH_TO_SOURCE_CODE}:/workspace,${NUSCENES_DATA_ROOT}:/data/nuscenes"

srun \
  --gpus=2 \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$MOUNTS" \
  --container-env=WANDB_API_KEY \
  bash -c '
    cd /workspace
    uv run tools/train.py
    echo "Training completed"
  '