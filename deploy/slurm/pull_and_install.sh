#!/bin/bash
# job-name is a static #SBATCH directive (sbatch parses it before any shell code runs, so it
# can't reference $IMAGE_NAME below) -- keep it in sync by hand if IMAGE_NAME ever changes.
#SBATCH --job-name=unidepthlss-install
#SBATCH --partition=H200-debug
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:29:59
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16

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
  enroot import \
    -o "$OUTPUT_SQSH" \
    docker://santimontiel/${IMAGE_NAME}:v1
else
  echo "Found $OUTPUT_SQSH, skipping import."
fi

echo "SBATCH cpus-per-task: $SLURM_CPUS_PER_TASK"

MOUNTS="${PATH_TO_SOURCE_CODE}:/workspace,${NUSCENES_DATA_ROOT}:/data/nuscenes"

srun \
  --gpus=1 \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$MOUNTS" \
  bash -c '
    ls -l /workspace
    cd /workspace
    echo "🔄 Checking virtual environment..."
    if [ ! -d ".venv" ]; then
      echo " .venv not found — updating virtual environment..."

      # Run uv sync and capture output
      if ! uv sync --link-mode=copy; then
        echo "❌ uv sync failed, cleaning up..."
        rm -rf .venv
        exit 1
      fi

      echo "✅ .venv updated"
    else
      echo "✅ .venv found"
    fi
    echo "Installation completed!"
  '
