#!/bin/bash
# ============================================================
# Preprocess all three datasets: raw VTP/VTU/STL -> DoMINO .npy
#
# WHY SLURM AND NOT THE LOGIN NODE:
#   ARC's guidelines cap login-node CPU work at ~15 minutes -- it is a shared
#   resource and heavy work there blocks other users from even logging in. 160
#   cases at ~30-65s each is well past that.
#
#   It is also simply faster: a cpu2023 node has 64 cores to the login node's 16,
#   and they are not shared.
#
# WHY CPU AND NOT GPU:
#   Nothing here touches a GPU. This is VTK mesh cleaning and numpy. Requesting a
#   GPU partition would waste a scarce resource and sit in a longer queue.
#
# THE BOTTLENECK:
#   pyvista's clean() -- single-threaded per mesh, and DrivAer's are 8.8M facets.
#   Parallelism therefore comes from running CASES concurrently, not from
#   speeding up any one case.
#
# SUBMIT:  sbatch scripts/preprocess.slurm
# WATCH:   squeue -u $USER
#          tail -f logs/preprocess_<jobid>.out
# ============================================================

#SBATCH --job-name=domino-preprocess
#SBATCH --partition=cpu2023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

set -euo pipefail

echo "=========================================="
echo "  job:    $SLURM_JOB_ID"
echo "  node:   $(hostname)"
echo "  cores:  $SLURM_CPUS_PER_TASK"
echo "  start:  $(date)"
echo "=========================================="

module load python/3.12.5
cd "$HOME/domino-xgeom"
source .venv/bin/activate

RAW="$HOME/data"
OUT="$HOME/processed"

# Note: preprocess.py skips cases whose .npy already exists, so this is safe to
# re-run after a timeout or a crash. It picks up where it left off.
for ds in ahmed windsor drivaer; do
    case "$ds" in
        ahmed)   raw="$RAW/ahmedml"   ;;
        windsor) raw="$RAW/windsorml" ;;
        drivaer) raw="$RAW/drivaerml" ;;
    esac

    echo ""
    echo ">>> $ds"
    python src/preprocess.py \
        --dataset "$ds" \
        --raw "$raw" \
        --out "$OUT/$ds" \
        --workers "$SLURM_CPUS_PER_TASK"
done

echo ""
echo "=========================================="
echo "  done:   $(date)"
echo "=========================================="