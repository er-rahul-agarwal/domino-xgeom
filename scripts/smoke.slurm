#!/bin/bash
# ============================================================
# THE SMOKE TEST. The first time train.py has ever been run in this project.
#
# ** WHAT THIS IS TESTING **
#   NOT accuracy. NOT the model. Whether DoMINO will consume our .npy files AT ALL.
#
#   The .npy contract was reverse-engineered by reading __getitem__ in
#   CachedDoMINODataset. It could be wrong about a key, a dtype, a shape, an axis
#   order. Everything downstream -- 160 preprocessed cases, the whole of Phase 3 --
#   rests on that guess being right.
#
#   SUCCESS = the data loads, the loss is finite, a checkpoint appears.
#   Ten epochs on four cases proves nothing about the model and everything about
#   the plumbing.
#
# ** THE RUNTIME IS ASSEMBLED, NOT SHIPPED **
#   NVIDIA's containers lag their GitHub main by a MAJOR version -- 25.11 ships
#   PhysicsNeMo 1.3.0; our pin is 2.2.0a0. There is no container that matches.
#   So: the container supplies torch 2.9 / CUDA 12.9 / cuML (the hard parts), and
#   PYTHONPATH supplies our pinned source plus the handful of pure-Python packages
#   the container lacks. See scripts/setup_env.sh for why installing the deps
#   naively silently replaces the container's PyTorch.
#
# ** --nv IS LOAD-BEARING **
#   Without it torch does not crash -- it reports no CUDA and runs on the CPU,
#   ~100x slower, with a decreasing loss and real checkpoints. You would find out
#   when the job timed out.
#
# SUBMIT:  sbatch scripts/smoke.slurm
# ============================================================

#SBATCH --job-name=domino-smoke
#SBATCH --partition=gpu-l40,gpu-v100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

REPO="$HOME/domino-xgeom"
SIF="$HOME/containers/physicsnemo-2511.sif"
DEPS="$HOME/pnemo-deps"
DOMINO="$REPO/external/physicsnemo/examples/cfd/external_aerodynamics/domino/src"

echo "=========================================="
echo "  job:   $SLURM_JOB_ID"
echo "  node:  $(hostname)"
echo "  gpu:   ${CUDA_VISIBLE_DEVICES:-none}"
echo "  start: $(date)"
echo "=========================================="

echo ""
echo ">>> host GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Our config lives in the repo, but Hydra resolves config_path relative to
# train.py. Rather than fight that, copy ours into the example's conf/ under a
# distinct name. The example's own config.yaml is left untouched -- we never edit
# their code (the two-repository principle).
echo ""
echo ">>> installing our config into the example's conf/"
cp "$REPO/conf/smoke.yaml" "$DOMINO/conf/smoke.yaml"

echo ""
echo ">>> running train.py"
echo "    10 epochs, 4 train cases, 2 val, surface-only"
echo ""

cd "$DOMINO"

apptainer exec --nv \
    --env PYTHONPATH="$DEPS:$REPO/external/physicsnemo" \
    --bind "$HOME/smoke:/data" \
    "$SIF" \
    python train.py --config-name=smoke \
        ++data.input_dir=/data/train \
        ++data.input_dir_val=/data/val \
        ++output="$REPO/outputs/smoke" \
        ++project_dir="$REPO/outputs/smoke"

echo ""
echo ">>> did a checkpoint appear?"
find "$REPO/outputs/smoke" -name "*.pt" -o -name "*.pth" | head

echo ""
echo "=========================================="
echo "  done:  $(date)"
echo "=========================================="