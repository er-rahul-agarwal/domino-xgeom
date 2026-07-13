#!/bin/bash
# ============================================================
# Prove the container can see the GPUs. Two minutes, zero risk.
#
# ** WHY THIS DESERVES ITS OWN JOB **
#   The --nv flag is what exposes the host's GPUs to the container. WITHOUT IT,
#   torch does not crash -- it silently reports no CUDA and falls back to CPU.
#   Training then "works": the loss decreases, checkpoints are written, and it is
#   roughly a hundred times slower. On a 24-hour wall clock you would discover
#   this by the job timing out, having learned nothing.
#
#   That is the same species of failure as every other trap in this project: a
#   plausible outcome, no error message. So we assert it BEFORE spending a real
#   training run on it.
#
# SUBMIT:  sbatch scripts/gpu_check.slurm
# ============================================================

#SBATCH --job-name=domino-gpucheck
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpucheck_%j.out
#SBATCH --error=logs/gpucheck_%j.err

set -euo pipefail

echo "=========================================="
echo "  node:  $(hostname)"
echo "  gpus:  $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

echo ">>> what the HOST sees:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

echo ">>> what the CONTAINER sees (with --nv):"
apptainer exec --nv "$HOME/containers/physicsnemo.sif" python -c "
import torch
print(f'    torch            {torch.__version__}')
print(f'    cuda available   {torch.cuda.is_available()}')
print(f'    device count     {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'    [{i}] {p.name}, {p.total_memory / 1e9:.0f} GB')

# The assertion. If this fails, --nv is not doing its job and every training run
# would silently run on CPU.
assert torch.cuda.is_available(), 'NO CUDA IN CONTAINER -- check the --nv flag'

# Prove it can actually compute, not merely enumerate.
x = torch.randn(4096, 4096, device='cuda')
y = x @ x
torch.cuda.synchronize()
print(f'    matmul on GPU    OK  (result {y.shape}, {y.device})')
"
echo ""

echo ">>> can the container import PhysicsNeMo and DoMINO?"
apptainer exec --nv \
    --bind "$HOME/domino-xgeom/external/physicsnemo:/physicsnemo" \
    "$HOME/containers/physicsnemo.sif" python -c "
import sys
sys.path.insert(0, '/physicsnemo')
import physicsnemo
print(f'    physicsnemo      {physicsnemo.__version__}')
from physicsnemo.models.domino.model import DoMINO
print(f'    DoMINO           imported OK')
from physicsnemo.datapipes.cae.domino_datapipe import CachedDoMINODataset
print(f'    CachedDoMINODataset imported OK')
"
echo ""

echo "=========================================="
echo "  ALL CHECKS PASSED"
echo "=========================================="