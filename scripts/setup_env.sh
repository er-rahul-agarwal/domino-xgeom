#!/bin/bash
# ============================================================
# Assemble the runtime: NVIDIA's container + OUR pinned PhysicsNeMo.
#
# ** THE PROBLEM THIS SOLVES **
#
#   NVIDIA's containers LAG their GitHub main by a MAJOR VERSION.
#
#       our pin (59aaf59)            physicsnemo 2.2.0a0
#       newest container (25.11)     physicsnemo 1.3.0
#       previous container (25.06)   physicsnemo 1.1.0
#
#   There is NO container that matches the source. And this is not cosmetic:
#   2.2.0a0 requires warp-lang >= 1.14, while 25.11 ships 1.10. Importing our
#   pinned source against the container's Warp fails immediately:
#
#       AttributeError: module 'warp' has no attribute 'LOG_WARNING'
#
# ** WHY WE KEEP THE PIN RATHER THAN DOWNGRADING TO MATCH THE CONTAINER **
#
#   Every load-bearing finding in this study is commit-specific:
#     - loss.py:423   channel 0 = pressure, channel 1 = x-shear (the ordering trap)
#     - train.py:646  load_checkpoint restores the EPOCH COUNTER (the fine-tuning trap)
#     - the .npy contract, reverse-engineered from domino_datapipe.py
#
#   Diffing 59aaf59 against v1.3.0 shows loss.py changed by 106 lines and train.py
#   by 145. Downgrading means re-verifying both from scratch -- and they are what
#   the entire study rests on.
#
# ** THE TRAP IN THE OBVIOUS FIX **
#
#   The natural move is `pip install` our source's dependencies. DO NOT.
#
#   `tensordict` depends on torch. pip cheerfully installed:
#
#       torch-2.13.0                 <- a WHOLE NEW PYTORCH
#       nvidia-cublas-13.1.1.3       <- a CUDA 13 stack
#       nvidia-cudnn-cu13-9.20.0.48
#       nvidia-nccl-cu13-2.29.7
#       numpy-2.5.1
#
#   The container ships torch 2.9 built against CUDA 12.9, matched to the driver.
#   Because PYTHONPATH puts our directory FIRST, Python would import the new torch
#   2.13 -- built for CUDA 13, against a driver that does not have it.
#
#   That is EXACTLY the dependency hell the container exists to prevent, and we
#   reintroduced it in one command. pip even warned us, in the middle of a wall of
#   success messages:
#
#       torchvision 0.24.0a0 requires torch==2.9.0a0+...nv25.10,
#         but you have torch 2.13.0 which is incompatible
#
# ** THE FIX **
#
#   --no-deps, into --target, with EVERY transitive dependency named explicitly.
#   Add only the pure-Python packages the container lacks. NEVER shadow torch,
#   numpy, or the CUDA libraries -- those come from the container, matched to the
#   driver, and they are the entire reason for using a container.
#
# ============================================================
# USAGE (login node -- this is a download, not compute):
#     bash scripts/setup_env.sh
#
# THEN, in any job:
#     apptainer exec --nv \
#       --env PYTHONPATH="$HOME/pnemo-deps:$HOME/domino-xgeom/external/physicsnemo" \
#       "$HOME/containers/physicsnemo-2511.sif" python ...
# ============================================================

set -euo pipefail

# ** 25.06, NOT 25.11. THE DRIVER IS THE CONSTRAINT. **
#
# ARC's NVIDIA driver supports CUDA 12.6. The 25.11 container's PyTorch is built
# against a newer CUDA and refuses to initialize:
#
#     CUDA initialization: The NVIDIA driver on your system is too old
#     (found version 12060)
#
# It does NOT crash. It falls back to CPU -- and training then "works", roughly a
# hundred times slower, with a decreasing loss and real checkpoints. You would
# find out when the 24-hour job timed out.
#
# 25.06 ships torch 2.7 built against a CUDA the driver DOES support.
#
# We cannot use a newer container than the driver allows, however much we might
# want its PhysicsNeMo.
SIF="$HOME/containers/physicsnemo.sif"
DEPS="$HOME/pnemo-deps"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: $SIF not found. Build it first:"
    echo "    sbatch scripts/build_container.slurm"
    exit 1
fi

echo ">>> wiping $DEPS (a partial install is worse than none)"
rm -rf "$DEPS"
mkdir -p "$DEPS"

echo ""
echo ">>> installing ONLY what the container lacks"
echo "    --no-deps is DELIBERATE. See the header."
echo ""

apptainer exec "$SIF" pip install --target="$DEPS" --no-deps \
    "warp-lang>=1.14" \
    jaxtyping \
    tensordict \
    einops \
    treescope \
    pyvers \
    cloudpickle \
    orjson \
    wadler-lindig \
    timm \
    torchinfo

echo ""
echo ">>> verifying: container's torch, OUR physicsnemo"
echo ""

apptainer exec "$SIF" env \
    PYTHONPATH="$DEPS:$HOME/domino-xgeom/external/physicsnemo" \
    python -c "
import torch, warp, physicsnemo

print(f'    torch        {torch.__version__}')
print(f'    warp         {warp.__version__}')
print(f'    physicsnemo  {physicsnemo.__version__}')
print(f'    from         {physicsnemo.__file__}')

# ** THE ASSERTION THAT MATTERS. **
# If torch does NOT come from the container, we have shadowed it with a pip build
# compiled against a different CUDA than the driver provides -- and every GPU run
# will fail in ways that look like a model problem rather than an install problem.
assert '/usr/local/lib' in torch.__file__, (
    f'torch is coming from {torch.__file__}, NOT the container. '
    f'Something pulled in its own PyTorch. Wipe \$HOME/pnemo-deps and reinstall '
    f'with --no-deps.'
)

# And physicsnemo MUST come from our pinned clone, not the container's 1.3.0.
assert 'domino-xgeom' in physicsnemo.__file__, (
    f'physicsnemo is coming from {physicsnemo.__file__}, not our pinned clone. '
    f'Check PYTHONPATH ordering.'
)
assert physicsnemo.__version__.startswith('2.'), (
    f'physicsnemo {physicsnemo.__version__} -- expected 2.2.0a0 from our pin. '
    f'The container ships 1.3.0 and all our findings are against 2.2.0a0.'
)

from physicsnemo.models.domino.model import DoMINO
from physicsnemo.datapipes.cae.domino_datapipe import CachedDoMINODataset
print('    DoMINO       imported')
print('    datapipe     imported')
"

echo ""
echo ">>> environment ready"
echo ""