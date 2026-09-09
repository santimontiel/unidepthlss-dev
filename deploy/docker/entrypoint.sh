#!/bin/bash
clear
export PYTHONPATH=$PYTHONPATH:/workspace
ulimit -n 65536

echo -e "\n------------------------------------------------------------------------------------\n"
figlet -c "UniDepthLSS"
echo -e "\n------------------------------------ System info -----------------------------------\n"

# Update uv project installation
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

# Get current username and store in a local variable.
CURRENT_USER=$(whoami)
# The Makefile mounts $NUSCENES_DATA_ROOT's *value* here; the variable itself is not forwarded
# into the container, so everything inside resolves this fixed path.
NUSCENES_DATA_ROOT="/data/nuscenes"

# Check if the nuScenes dataset is available.
echo -e "\n\U0001F50D Checking datasets availability..."
if [ ! -d "$NUSCENES_DATA_ROOT/samples" ] || [ ! -d "$NUSCENES_DATA_ROOT/sweeps" ]; then
    echo -e "\u274C \033[91m\033[1mnuScenes dataset not found or incomplete...\033[0m"
    echo -e "   Please download the dataset and share it in a volume at $NUSCENES_DATA_ROOT."
else
    echo -e "\u2705 \033[92m\033[1mnuScenes dataset at\033[0m $NUSCENES_DATA_ROOT\033[92m\033[1m found!\033[0m"
fi

# Check if CUDA is available
echo -e "\n🔍 Checking GPU and CUDA availability..."
if ! uv run --preview-features extra-build-dependencies python -c "import torch" 2>/dev/null; then
    echo -e "❌ \033[91m\033[1mFailed to import torch\033[0m"
    echo -e "   Please check your PyTorch installation!"
else

    CUDA_AVAILABLE=$(uv run --preview-features extra-build-dependencies python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" == "True" ]; then
        echo -e "✅ \033[92m\033[1mPyTorch is working properly with the GPU.\033[0m"
        echo -e "📍 GPU Information:"
        uv run --preview-features extra-build-dependencies python -c "import torch; print(f'   - CUDA version:     {torch.version.cuda}')"
        uv run --preview-features extra-build-dependencies python -c "import torch; print(f'   - Device name:      {torch.cuda.get_device_name(0)}')"
        uv run --preview-features extra-build-dependencies python -c "import torch; print(f'   - Number of GPUs:   {torch.cuda.device_count()}')"
    else
        echo -e "❌ \033[91m\033[1mCUDA is not available!\033[0m"
        echo -e "   Check your PyTorch installation"
    fi
fi

echo -e "\n------------------------------------------------------------------------------------\n"