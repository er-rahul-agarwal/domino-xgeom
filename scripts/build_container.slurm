#!/bin/bash
# ============================================================
# Build the PhysicsNeMo Apptainer container.
#
# ** WHY THIS IS A SLURM JOB AND NOT A LOGIN-NODE COMMAND **
#
#   `apptainer pull` does two things:
#     1. DOWNLOAD the Docker layers  -- network, cheap
#     2. ASSEMBLE them into a SIF    -- mksquashfs: CPU + I/O, VERY expensive
#
#   Run on the login node, step 2 ground for 41 minutes at 12% CPU and produced
#   NOTHING -- throttled and I/O-starved on a machine shared with 80 other users.
#   ARC's guidelines cap login-node CPU work at ~15 minutes and this is precisely
#   the workload they have in mind. It should not have been started there.
#
#   The 15 GB of layers are already in ~/.apptainer/cache from that attempt, so
#   this run skips the download and goes straight to assembly.
#
# WHY cpu2025 AND NOT A GPU PARTITION:
#   mksquashfs is pure CPU compression. A GPU would sit idle while occupying one
#   of only 4 H100 nodes on the whole cluster.
#
#   cpu2025 at time of writing: 1832 of 2432 cores IDLE, 1 TB RAM per node. The
#   emptiest partition available, and the newest hardware.
#
# WHY A CONTAINER AT ALL:
#   PhysicsNeMo needs torch, CUDA, NVIDIA Warp and cuml, all version-matched.
#   Building that natively against an HPC module system is days of work and
#   usually fails. NVIDIA ships a container so that nobody has to.
#
# SUBMIT:  sbatch scripts/build_container.slurm
# WATCH:   tail -f logs/container_<jobid>.out
# ============================================================

#SBATCH --job-name=domino-container
#SBATCH --partition=cpu2025
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/container_%j.out
#SBATCH --error=logs/container_%j.err

set -euo pipefail

echo "=========================================="
echo "  job:   $SLURM_JOB_ID"
echo "  node:  $(hostname)"
echo "  cores: $SLURM_CPUS_PER_TASK"
echo "  start: $(date)"
echo "=========================================="

mkdir -p "$HOME/containers"

# ** BUILD IN NODE-LOCAL /tmp, NOT ON THE SHARED FILESYSTEM. **
# mksquashfs writes millions of small blocks. Doing that over a network mount is
# a large part of why the login-node attempt crawled. /tmp is a local disk.
export APPTAINER_TMPDIR="/tmp/apptainer_${SLURM_JOB_ID}"
mkdir -p "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

echo ""
echo ">>> assembling nvcr.io/nvidia/physicsnemo/physicsnemo:25.06"
echo "    layers are cached; this is the mksquashfs step"
echo "    tmpdir: $APPTAINER_TMPDIR"
echo ""

cd "$HOME/containers"
apptainer pull --force physicsnemo.sif \
    docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.06

echo ""
echo ">>> built:"
ls -lh "$HOME/containers/physicsnemo.sif"

echo ""
echo ">>> sanity check: can the container see torch?"
apptainer exec "$HOME/containers/physicsnemo.sif" \
    python -c "import torch; print(f'    torch {torch.__version__}, CUDA built: {torch.version.cuda}')"

echo ""
echo "=========================================="
echo "  done:  $(date)"
echo "=========================================="