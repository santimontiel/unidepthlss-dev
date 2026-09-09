#!/bin/bash
#SBATCH --job-name=unidepthlss-sidecar
#SBATCH --partition=H200-debug
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:29:59   # H200-debug rejects anything at or above 30 min
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16

# Build the calibration sidecar on the cluster, once, before the first training job.
#
#   sbatch deploy/slurm/build_sidecar_slurm.sh
#   FORCE=1 sbatch deploy/slurm/build_sidecar_slurm.sh     # rebuild in place
#   SIDECAR=/some/writable/path.npz sbatch ...             # explicit destination
#
# `train_slurm.sh` runs the same check-and-build inline, so this job is not strictly required --
# it exists so the (few-minute, CPU-bound, devkit-loading) build can happen on the debug
# partition instead of burning the front of a multi-day training allocation, and so a failure
# here is diagnosed on its own rather than as a dead training job.
#
# ## What this builds and why
#
# `data.source=store` reads the sibling repos' nuScenes store for labels, intrinsics, image paths
# and split lists -- all verified byte-identical to the devkit -- but NOT for extrinsics: the
# store holds `cam_from_ego_flat` at the LIDAR pose while this model consumes `calibrated_sensor`
# at the camera's own timestamp, and the two differ by up to 0.54 m. This sidecar (0.9 MB
# compressed, because calibration is per-log and deduplicates almost entirely) supplies the
# released convention, and is what lets training drop the devkit -- worth ~7.8 GB of RAM per
# dataloader worker.
#
# It requires the devkit and the dataset, takes a few minutes, and needs redoing only if the
# dataset version changes. `data.source=devkit` needs no sidecar at all (but caps num_workers).

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_NAME=unidepthlss-dev
IMAGE_NAME=unidepthlss
PATH_TO_SOURCE_CODE=/raid/${USER}/Workspace/${REPO_NAME}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${IMAGE_NAME}_v1.sqsh
NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-/raid/${USER}/Datasets/nuscenes}"

if [ ! -f "$OUTPUT_SQSH" ]; then
  echo "Error: $OUTPUT_SQSH not found. Run pull_and_install.sh first."
  exit 1
fi
if [ ! -d "$PATH_TO_SOURCE_CODE/.venv" ]; then
  echo "Error: no .venv at $PATH_TO_SOURCE_CODE. Run pull_and_install.sh first."
  exit 1
fi

srun \
  --gpus=1 \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$PATH_TO_SOURCE_CODE:/workspace,${NUSCENES_DATA_ROOT}:/data/nuscenes" \
  bash -c "
    set -e
    cd /workspace

    # The dataset mount is frequently read-only on a cluster, and the sidecar's natural home is
    # beside the dataset. Fall back to the repo mount rather than failing -- training accepts an
    # explicit path, so the only cost is having to pass it.
    SIDECAR=\"\${SIDECAR:-}\"
    if [ -z \"\$SIDECAR\" ]; then
      if touch /data/nuscenes/.unidepthlss_write_test 2>/dev/null; then
        rm -f /data/nuscenes/.unidepthlss_write_test
        SIDECAR=/data/nuscenes/unidepthlss_calibration.npz
      else
        SIDECAR=/workspace/.cache/unidepthlss_calibration.npz
        echo '⚠️  /data/nuscenes is not writable -- writing the sidecar to the repo mount instead.'
        echo '   Pass this to training:  data.calibration_sidecar='\$SIDECAR
      fi
    fi
    echo \"Sidecar destination: \$SIDECAR\"

    FORCE_FLAG=''
    if [ -n \"\${FORCE}\" ]; then FORCE_FLAG='--force'; fi

    uv run --no-sync tools/build_calibration_sidecar.py \
      --dataroot /data/nuscenes --out \"\$SIDECAR\" \$FORCE_FLAG

    uv run --no-sync tools/build_calibration_sidecar.py \
      --dataroot /data/nuscenes --out \"\$SIDECAR\" --check

    echo 'Sidecar ready ✅'
  "
