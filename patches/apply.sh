#!/bin/bash
# ============================================================
# Apply our changes to the pinned PhysicsNeMo clone.
#
# ** IDEMPOTENT. ** You will re-clone, and this must not fail the second time.
#
# THE TWO-REPOSITORY PRINCIPLE:
#   external/ is a pinned, gitignored clone of NVIDIA's code. We NEVER edit it in
#   place. Every change lives here, and is:
#     - re-appliable after a re-clone
#     - legible as a small, honest delta, not lost in a 100k-line fork
#     - reproducible: "clone at 59aaf59, run patches/apply.sh"
#
# WHY sed AND NOT A .patch FILE:
#   A hand-written unified diff must get its @@ line counts exactly right or git
#   rejects it as corrupt -- which is what happened. These edits are small and
#   surgical; a targeted substitution is more robust and, frankly, more readable.
#
# USAGE:  bash patches/apply.sh
# ============================================================

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PNEMO="$REPO/external/physicsnemo"
DOMINO="$PNEMO/examples/cfd/external_aerodynamics/domino/src"

EXPECTED="59aaf59b48901bd8df2c9bad83f6d05ea47d8c04"

if [[ ! -d "$PNEMO" ]]; then
    echo "ERROR: $PNEMO not found. Clone it first:"
    echo "    cd external && git clone https://github.com/NVIDIA/physicsnemo.git"
    echo "    cd physicsnemo && git checkout $EXPECTED"
    exit 1
fi

cd "$PNEMO"
ACTUAL="$(git rev-parse HEAD)"

echo ">>> pinned commit: $ACTUAL"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo ""
    echo "WARNING: expected $EXPECTED"
    echo ""
    echo "Every finding in the plan is verified against that commit --"
    echo "loss.py:423 (the ordering trap), train.py:646 (the epoch trap)."
    echo "On another commit the line numbers, and possibly the behaviour, differ."
    echo ""
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" == "y" ]] || exit 1
fi

# ============================================================
# PATCH #1 -- compute_statistics.py: respect model_type
#
# ** THE BUG **
#   compute_statistics.py hardcodes target_keys to include the VOLUME fields,
#   regardless of cfg.model.model_type. On a surface-only dataset -- where
#   volume_fields is None, because there IS no volume data and the volume meshes
#   are 61x larger on disk -- CAEDataset then calls
#
#       torch.from_numpy(data["volume_fields"])
#
#   on None, and dies:
#
#       TypeError: expected np.ndarray (got NoneType)
#
#   model_type is a documented config value with three legal settings
#   (surface / volume / combined). This script honoured none of them.
#
# ** WHY NOT SIMPLY OMIT THE VOLUME KEYS FROM OUR .npy **
#   Because then data["volume_fields"] raises KeyError instead of TypeError. The
#   script is still asking for a key that cannot exist. The fix belongs where the
#   assumption is made.
# ============================================================

echo ""
echo ">>> patch 1: compute_statistics.py -- respect model_type"

TARGET="$DOMINO/compute_statistics.py"

if grep -q "PATCHED: respect model_type" "$TARGET"; then
    echo "    already applied, skipping"
else
    python3 - "$TARGET" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()

old = '''        target_keys = [
            "volume_fields",
            "surface_fields",
            "stl_centers",
            "volume_mesh_centers",
            "surface_mesh_centers",
        ]'''

new = '''        # ** PATCHED: respect model_type. **
        #
        # The original hardcoded all five keys, including the volume ones. On a
        # surface-only dataset -- where volume_fields is None -- CAEDataset then
        # calls torch.from_numpy(None) and dies with
        #
        #     TypeError: expected np.ndarray (got NoneType)
        #
        # model_type is a documented config value with three legal settings
        # (surface / volume / combined). This script honoured none of them.
        model_type = cfg.model.model_type

        target_keys = ["stl_centers"]
        if model_type in ("surface", "combined"):
            target_keys += ["surface_fields", "surface_mesh_centers"]
        if model_type in ("volume", "combined"):
            target_keys += ["volume_fields", "volume_mesh_centers"]

        logger.info(f"model_type={model_type}, target_keys={target_keys}")'''

if old not in src:
    print("    ERROR: could not find the block to patch.")
    print("    The upstream file has changed. Re-derive the patch.")
    sys.exit(1)

path.write_text(src.replace(old, new, 1))
print("    ok")
PYEOF
fi

# ============================================================
# PATCH #2 -- utils.py: load_scaling_factors must respect model_type
#
# ** THE SAME BUG, IN A DIFFERENT FILE. **
#   load_scaling_factors() unconditionally reads scaling_factors[...]["volume_fields"],
#   regardless of model_type. On a surface-only run the statistics file legitimately
#   has no volume entry (patch #1 saw to that), so this dies:
#
#       KeyError: 'volume_fields'
#
# ** THE PATTERN, AND IT IS WORTH NAMING. **
#   `model_type: surface` is a DOCUMENTED, LEGAL setting in NVIDIA's own config --
#   one of three, alongside volume and combined. Their example code does not
#   actually support it. Only the volume/combined path has been exercised.
#
#   This is the second file we have had to patch for the same assumption, and it
#   will not be the last.
# ============================================================

echo ""
echo ">>> patch 2: utils.py -- load_scaling_factors respects model_type"

TARGET2="$DOMINO/utils.py"

if grep -q "PATCHED: respect model_type" "$TARGET2"; then
    echo "    already applied, skipping"
else
    python3 - "$TARGET2" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()

old = '''    if cfg.model.normalization == "min_max_scaling":
        vol_factors = np.asarray(
            [
                scaling_factors.max_val["volume_fields"],
                scaling_factors.min_val["volume_fields"],
            ]
        )'''

new = '''    # ** PATCHED: respect model_type. **
    #
    # The original read "volume_fields" unconditionally. On a surface-only run the
    # statistics file legitimately has no volume entry -- there IS no volume data --
    # and this raised KeyError: 'volume_fields'.
    #
    # model_type is a documented setting with three legal values. This is the SECOND
    # file that ignores it.
    _has_volume = cfg.model.model_type in ("volume", "combined")

    if cfg.model.normalization == "min_max_scaling":
        vol_factors = np.asarray(
            [
                scaling_factors.max_val["volume_fields"],
                scaling_factors.min_val["volume_fields"],
            ]
        ) if _has_volume else None'''

if old not in src:
    print("    ERROR: could not find the min_max block. Upstream changed.")
    sys.exit(1)

src = src.replace(old, new, 1)

old2 = '''    elif cfg.model.normalization == "mean_std_scaling":
        vol_factors = np.asarray(
            [
                scaling_factors.mean["volume_fields"],
                scaling_factors.std["volume_fields"],
            ]
        )'''

new2 = '''    elif cfg.model.normalization == "mean_std_scaling":
        vol_factors = np.asarray(
            [
                scaling_factors.mean["volume_fields"],
                scaling_factors.std["volume_fields"],
            ]
        ) if _has_volume else None'''

if old2 not in src:
    print("    ERROR: could not find the mean_std block. Upstream changed.")
    sys.exit(1)

src = src.replace(old2, new2, 1)
path.write_text(src)
print("    ok")
PYEOF
fi

echo ""
echo ">>> delta from upstream:"
git diff --stat
echo ""