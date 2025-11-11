#!/bin/bash

# FBGEMM Build Script with CUTLASS Blackwell FMHA Support
# This script sets up a conda environment and builds FBGEMM with genai target

set -e  # Exit on error

# Configuration
CONDA_ENV_NAME="fbgemm"
PYTHON_VERSION="3.12"
CUDA_VERSION="12.9"
PYTORCH_VERSION="nightly"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FBGEMM GenAI Build Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Initialize git submodules (including CUTLASS)
echo -e "\n${YELLOW}[Step 1/6] Initializing git submodules (CUTLASS, etc.)${NC}"
git submodule sync
git submodule update --init --recursive

# Verify CUTLASS is available
if [ -d "external/cutlass/include" ]; then
    echo -e "${GREEN}✓ CUTLASS submodule initialized successfully${NC}"
else
    echo -e "${RED}✗ CUTLASS submodule not found!${NC}"
    exit 1
fi

# Step 2: Create conda environment
echo -e "\n${YELLOW}[Step 2/6] Creating conda environment: ${CONDA_ENV_NAME}${NC}"
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Environment ${CONDA_ENV_NAME} already exists. Using existing environment..."
    echo -e "${GREEN}✓ Using existing conda environment${NC}"
else
    echo "Creating new conda environment ${CONDA_ENV_NAME}..."
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
    echo -e "${GREEN}✓ Conda environment created${NC}"
fi

# Activate the environment
echo -e "\n${YELLOW}Activating conda environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}

# Step 3: Install PyTorch nightly with CUDA 12.9
echo -e "\n${YELLOW}[Step 3/6] Installing PyTorch nightly with CUDA ${CUDA_VERSION}${NC}"
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Step 4: Install build dependencies
echo -e "\n${YELLOW}[Step 4/6] Installing build dependencies${NC}"
pip install -r fbgemm_gpu/requirements_genai.txt

# Install NCCL for distributed communication support
echo -e "\n${YELLOW}Installing NCCL...${NC}"
conda install -c conda-forge nccl -y

# Step 5: Set environment variables
echo -e "\n${YELLOW}[Step 5/6] Setting environment variables${NC}"

# CUDA paths (using system CUDA 12.9)
export CUDA_HOME="/usr/local/cuda-12.9"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# PyTorch CUDA architecture list - Blackwell (100a)
export TORCH_CUDA_ARCH_LIST="10.0a"

# Suppress warnings
export CXXFLAGS="-w"
export NVCCFLAGS="-w"

# CMake and build settings
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export MAX_JOBS=$(nproc)

# Set library paths for NCCL and NVML
export NCCL_LIB_PATH="${CONDA_PREFIX}/lib/libnccl.so"
export NVML_LIB_PATH="${CUDA_HOME}/lib64/stubs/libnvidia-ml.so"

echo "CUDA_HOME: ${CUDA_HOME}"
echo "CUDACXX: ${CUDACXX}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "CMAKE_BUILD_PARALLEL_LEVEL: ${CMAKE_BUILD_PARALLEL_LEVEL}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CXXFLAGS: ${CXXFLAGS}"
echo "NVCCFLAGS: ${NVCCFLAGS}"
echo "NCCL_LIB_PATH: ${NCCL_LIB_PATH}"
echo "NVML_LIB_PATH: ${NVML_LIB_PATH}"

# Step 6: Build FBGEMM with genai target
echo -e "\n${YELLOW}[Step 6/6] Building FBGEMM with genai target${NC}"

cd fbgemm_gpu

# Clean previous builds
echo "Cleaning previous build artifacts..."
rm -rf build dist *.egg-info _skbuild
rm -rf experimental/gen_ai/build experimental/gen_ai/dist experimental/gen_ai/*.egg-info

# Install the package
echo -e "\n${GREEN}Installing the built package...${NC}"
python setup.py install \
    --build-target=genai \
    --build-variant=cuda \
    --nvml_lib_path="${NVML_LIB_PATH}" \
    --nccl_lib_path="${NCCL_LIB_PATH}" \
    -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

cd ..

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
python -c "
import torch
import fbgemm_gpu
import fbgemm_gpu.experimental.gen_ai

print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ FBGEMM_GPU imported successfully')
print(f'✓ FBGEMM_GPU GenAI module imported successfully')
print('\nAvailable FBGEMM GenAI operators:')
ops = [op for op in dir(torch.ops.fbgemm) if not op.startswith('_')]
for op in sorted(ops)[:10]:  # Show first 10 operators
    print(f'  - {op}')
"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Instructions${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "To use this environment:"
echo -e "  ${YELLOW}conda activate ${CONDA_ENV_NAME}${NC}"
echo -e ""
echo -e "To run the Blackwell FMHA test:"
echo -e "  ${YELLOW}CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 python fmha.py"
echo -e ""
echo -e "To deactivate:"
echo -e "  ${YELLOW}conda deactivate${NC}"
echo -e ""
echo -e "${GREEN}Build artifacts location:${NC}"
echo -e "  ${SCRIPT_DIR}/fbgemm_gpu/build"
echo -e ""
