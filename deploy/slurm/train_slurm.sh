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

# Trains BOTH BEV windows, sequentially, in one allocation:
#
#   sbatch deploy/slurm/train_slurm.sh                      # standard, then paper
#   BEV_GRID=standard sbatch deploy/slurm/train_slurm.sh    # one arm only
#   BEV_GRIDS="paper standard" sbatch train_slurm.sh        # any list, in that order
#   SMOKE=1 sbatch deploy/slurm/train_slurm.sh              # both, a few batches each
#   MAX_EPOCHS=10 sbatch deploy/slurm/train_slurm.sh        # shorter schedule (see the caveat)
#   RESUME=/workspace/outputs/train/<run>/checkpoints/last.ckpt BEV_GRID=standard sbatch ...
#
# **Why two arms is the default.** The whole point of this repo is that the two windows are not
# interchangeable: `paper` (128x128 over 64x64 m) reproduces the published 49.4 IoU, and
# `standard` (200x200 over 100x100 m) is the only setting comparable to the baselines that
# number is quoted against. A checkpoint trained on one cannot be fairly evaluated on the other
# -- see docs/baseline.md -- so the pair has to be trained, not just evaluated. `standard` runs
# first because it is the one with an open question attached.

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_NAME=unidepthlss-dev
IMAGE_NAME=unidepthlss
PATH_TO_SOURCE_CODE=/raid/${USER}/Workspace/${REPO_NAME}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${IMAGE_NAME}_v1.sqsh
DEVICES=2

# Same name as the Makefile. This repo needs exactly one dataset. Override by exporting it
# before submitting (sbatch forwards the submitting shell's environment by default).
NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-/raid/${USER}/Datasets/nuscenes}"

# 30 epochs is the paper's schedule (Sec. 3.6). It is NOT just a stopping point: `max_epochs`
# sets CosineAnnealingLR's `T_max`, so lowering it changes the entire learning-rate trajectory
# rather than truncating this one. The released checkpoint is from epoch 10 of a T_max=30
# schedule -- its saved optimizer LR is 0.00022525, which matches cosine(t=10, T_max=30) to
# 1e-10 and not cosine(t=10, T_max=10), which would already be annealed to 1e-6. So training
# "for 10 epochs" is a different experiment, not a shorter version of this one.
MAX_EPOCHS=${MAX_EPOCHS:-30}

# Which windows this job trains, and in what order. `BEV_GRID` names exactly one arm and is the
# escape hatch from the two-arm default.
BEV_GRID_EXPLICIT=${BEV_GRID:+yes}
BEV_GRID=${BEV_GRID:-standard}
DEFAULT_ARMS="standard paper"
if [ -n "${BEV_GRID_EXPLICIT}" ]; then
  BEV_GRIDS=${BEV_GRIDS:-$BEV_GRID}
else
  BEV_GRIDS=${BEV_GRIDS:-$DEFAULT_ARMS}
fi

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

# `%L` is the allocation's remaining wall time, which is what the guard below needs -- elapsed
# time would have to assume this job started when it was submitted, and a queued job does not.
remaining_hours() {
  local left
  left=$(squeue -h -j "${SLURM_JOB_ID}" -o "%L" 2>/dev/null | tr -d ' ')
  case "$left" in
    ""|UNLIMITED|INVALID) echo "9999"; return ;;
  esac
  echo "$left" | awk -F'[-:]' '{
    if (NF == 4)      h = $1 * 24 + $2 + $3 / 60
    else if (NF == 3) h = $1 + $2 / 60
    else if (NF == 2) h = $1 / 60
    else              h = 9999
    printf "%.1f", h
  }'
}

# Refuse to start an arm that cannot finish. Slurm kills the job at the wall clock regardless of
# which epoch it is on, and a truncated arm is worse than a missing one: it lands in W&B looking
# like a complete run, at a checkpoint the cosine schedule never annealed, and invites comparison
# against the arm that did finish. Left at 0 until a real 30-epoch arm has been timed on this
# partition -- set it once that number exists rather than guessing now.
MIN_HOURS_PER_ARM=${MIN_HOURS_PER_ARM:-0}

# One stamp for the whole job, so the two arms are identifiable as a pair in W&B and on disk.
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
STATUS_LINES=""
ARM_INDEX=0
ARM_COUNT=$(echo ${BEV_GRIDS} | wc -w)
EXIT_CODE=0

echo ""
echo "=================================================================="
echo "UniDepth-LSS training sweep"
echo "  Arms       : ${BEV_GRIDS}  (${ARM_COUNT})"
echo "  Epochs     : ${MAX_EPOCHS} per arm"
echo "  Devices    : ${DEVICES}"
echo "  Stamp      : ${STAMP}"
echo "=================================================================="

for GRID in ${BEV_GRIDS}; do
  ARM_INDEX=$((ARM_INDEX + 1))
  LEFT=$(remaining_hours)
  ARM_RUN_ID="${GRID}_${MAX_EPOCHS}ep_${STAMP}"

  if awk "BEGIN{exit !(${LEFT} < ${MIN_HOURS_PER_ARM})}"; then
    echo "⏭️  Skipping ${GRID}: ${LEFT} h left, needs ${MIN_HOURS_PER_ARM} h"
    STATUS_LINES="${STATUS_LINES}\n  ${ARM_INDEX}. ${GRID}: SKIPPED (out of wall clock)"
    EXIT_CODE=1
    continue
  fi

  OVERRIDES="data/bev=${GRID} trainer.max_epochs=${MAX_EPOCHS} trainer.devices=${DEVICES}"
  OVERRIDES="${OVERRIDES} run_id=${ARM_RUN_ID}"

  if [ -n "${SMOKE}" ]; then
    OVERRIDES="${OVERRIDES} trainer.limit_train_batches=20 trainer.limit_val_batches=10"
    OVERRIDES="${OVERRIDES} logger=csv task_name=smoke"
  fi
  if [ -n "${RESUME}" ]; then
    echo "↩️  Resuming from ${RESUME}"
    OVERRIDES="${OVERRIDES} resume_checkpoint=${RESUME}"
  fi

  echo ""
  echo "=================================================================="
  echo "Arm ${ARM_INDEX}/${ARM_COUNT}: bev=${GRID}   (${LEFT} h of wall clock left)"
  echo "  Run ID          : ${ARM_RUN_ID}"
  echo "  Hydra overrides : ${OVERRIDES}"
  echo "=================================================================="

  srun \
    --gpus=${DEVICES} \
    --container-image="$OUTPUT_SQSH" \
    --container-mounts="$MOUNTS" \
    --container-env=WANDB_API_KEY,HF_TOKEN \
    bash -c "
      set -e
      cd /workspace

      # Seconds, and it catches the class of bug a dry run cannot: --cfg job composes config but
      # never instantiates it, so a constructor that rejects one of the config's keys still
      # passes a dry run and then dies at launch, after the queue wait.
      uv run --no-sync dev/check_instantiate.py

      uv run --no-sync tools/train.py ${OVERRIDES}
    "
  ARM_STATUS=$?

  # Deliberately *not* fatal. The two arms answer different questions, so an arm that crashes
  # must not cancel the one that would still have been informative. The job's own exit code
  # still reports the failure, and the summary below says which arm it was.
  if [ $ARM_STATUS -eq 0 ]; then
    echo "✅ ${GRID} finished"
    STATUS_LINES="${STATUS_LINES}\n  ${ARM_INDEX}. ${GRID}: OK        ${ARM_RUN_ID}"
  else
    echo "❌ ${GRID} exited ${ARM_STATUS} -- continuing to the next arm"
    STATUS_LINES="${STATUS_LINES}\n  ${ARM_INDEX}. ${GRID}: FAILED(${ARM_STATUS})  ${ARM_RUN_ID}"
    EXIT_CODE=1
  fi
done

echo ""
echo "=================================================================="
echo "Sweep summary (stamp ${STAMP}, $(remaining_hours) h wall clock left)"
echo "=================================================================="
printf "%b\n" "${STATUS_LINES}"
echo "=================================================================="
echo "Artifacts : /workspace/outputs/train/<run_id>/ (config snapshot, train.log,"
echo "            checkpoints/, csv/ or wandb/)"
echo "W&B       : project unidepthlss_nuscenes_vehicle, runs matching *_${STAMP}"
echo ""
echo "Then score each checkpoint under BOTH windows to compare like for like:"
echo "  uv run tools/eval.py checkpoint_path=outputs/train/<run_id>/checkpoints/best.ckpt"
exit $EXIT_CODE
