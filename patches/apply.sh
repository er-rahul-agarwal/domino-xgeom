#!/bin/bash
# ============================================================
# Apply our patches to the pinned PhysicsNeMo clone.
#
# ** IDEMPOTENT. ** You will re-clone, and this must not fail the second time.
#
# THE TWO-REPOSITORY PRINCIPLE:
#   external/ is a pinned, gitignored clone of NVIDIA's code. We NEVER edit it in
#   place. Every change lives here, as a patch, and is:
#     - re-appliable after a re-clone
#     - legible as a small, honest delta rather than lost in a 100k-line fork
#     - reproducible by anyone: "clone at 59aaf59, run patches/apply.sh"
#
# USAGE:  bash patches/apply.sh
# ============================================================

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PNEMO="$REPO/external/physicsnemo"

if [[ ! -d "$PNEMO" ]]; then
    echo "ERROR: $PNEMO not found. Clone it first:"
    echo "    cd external && git clone https://github.com/NVIDIA/physicsnemo.git"
    echo "    cd physicsnemo && git checkout 59aaf59b48901bd8df2c9bad83f6d05ea47d8c04"
    exit 1
fi

cd "$PNEMO"

echo ">>> pinned commit:"
git rev-parse HEAD

EXPECTED="59aaf59b48901bd8df2c9bad83f6d05ea47d8c04"
ACTUAL="$(git rev-parse HEAD)"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo ""
    echo "WARNING: expected $EXPECTED"
    echo "         got      $ACTUAL"
    echo ""
    echo "Every finding in the plan is verified against $EXPECTED --"
    echo "loss.py:423 (the ordering trap), train.py:646 (the epoch trap)."
    echo "On another commit the line numbers, and possibly the behaviour, differ."
    echo ""
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" == "y" ]] || exit 1
fi

echo ""
for patch in "$REPO"/patches/*.patch; do
    name="$(basename "$patch")"

    # Already applied? (--reverse --check succeeds iff it IS applied)
    if git apply --reverse --check "$patch" 2>/dev/null; then
        echo "    $name  already applied, skipping"
        continue
    fi

    echo -n "    $name  applying... "
    git apply "$patch"
    echo "ok"
done

echo ""
echo ">>> patches applied. Delta from upstream:"
git diff --stat
echo ""