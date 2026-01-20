#!/bin/bash
set -e

echo "============================================"
echo "FoundationPose Install Script"
echo "Target: Python 3.12, CUDA 12.8 compatibility"
echo "============================================"

# 1. Create a virtual environment with python 3.12
echo "[Step 1] Creating Python 3.12 virtual environment..."

if ! command -v python3.12 &> /dev/null; then
    echo "Error: python3.12 not found. Please install Python 3.12."
    exit 1
fi

python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
pip install wheel setuptools ninja

# 2. Install PyTorch with CUDA support
# CUDA 12.8 is backward compatible with CUDA 12.x binaries.
# Installing PyTorch compatible with CUDA 12.4 (current stable standard for 12.x)
echo "[Step 2] Installing PyTorch (CUDA 12.8 build to match CUDA 12.8 driver)..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install requirements
echo "[Step 3] Installing dependencies from requirements.txt..."
# Note: requirements.txt includes nvdiffrast and warp from git.
# Installing them after PyTorch ensures they can locate the torch CUDA extensions.
pip install -r requirements.txt

pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# 4. Install nvdiffrast and warp (Verify/Reinstall for CUDA 12.8 context)
# The requirements.txt install should handle this, but being explicit about
# expectations. The git+ installation builds from source, which is ideal
# for ensuring compatibility with the specific CUDA Toolkit version on the system.

# 5, Install FoundationPose package
echo "[Step 4] Installing FoundationPose package..."
pip install -e .