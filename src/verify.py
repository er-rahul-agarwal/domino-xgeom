# verify.py
# ============================================================
# PURPOSE:
#   Check that the preprocessed .npy files are what we think they are.
#
# ** WHY THIS EXISTS -- IT ALREADY CAUGHT A REAL CORRUPTION **
#
#   Two Slurm jobs were accidentally submitted at once: one started before the
#   old Windsor files were deleted, one after. They wrote to the SAME directory,
#   with DIFFERENT code -- one applying the y/z swap, one not. Whichever finished
#   first won, per file.
#
#   Result: 20 of 40 Windsor files were y-up and 20 were z-up.
#
#   `ls | wc -l` said 40. Every file loaded fine. Every array had the right
#   shape. The training loss would have converged smoothly and the model would
#   have been worthless -- and NOTHING downstream would have told us.
#
#   The only reason we know is that this asks a question about the CONTENT:
#   "is the geometry actually z-up?"
#
#   ** THE LESSON: when data is regenerated, verify the CONTENT, not the file
#   count. **
#
# USAGE:
#     python src/verify.py --dataset windsor --dir ~/processed/windsor
#     python src/verify.py --all --root ~/processed
#
# DEPENDENCIES: numpy. No pyvista, no GPU.
# ============================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from datasets import CANONICAL_VARIABLES, DATASETS


def check_case(path: Path, dataset: str) -> list[str]:
    """
    Return a list of problems with one .npy file. Empty means it is fine.

    WHY EACH CHECK EXISTS:
        Every one corresponds to a failure mode we have actually hit, or one that
        would be silent if we had not looked for it.
    """
    ds = DATASETS[dataset]
    problems: list[str] = []

    d = np.load(path, allow_pickle=True).item()

    # ---- The keys DoMINO's datapipe requires ---------------------------------
    required = [
        "stl_coordinates", "stl_centers", "stl_faces", "stl_areas",
        "surface_mesh_centers", "surface_normals", "surface_areas",
        "surface_fields", "global_params_values",
    ]
    for k in required:
        if k not in d:
            problems.append(f"missing key '{k}'")
    if problems:
        return problems      # nothing else is meaningful

    v = d["stl_coordinates"]
    sf = d["surface_fields"]
    n = d["surface_normals"]

    # ---- The y/z swap. THE ONE THAT CAUGHT THE CORRUPTION. -------------------
    # After the swap, Windsor's z (height, ~0.475) must exceed its y (width,
    # ~0.389). Before the swap it was the other way round. A file written by the
    # old code is indistinguishable from a correct one EXCEPT by this test.
    y_span = float(v[:, 1].max() - v[:, 1].min())
    z_span = float(v[:, 2].max() - v[:, 2].min())
    if ds.swap_yz and z_span < y_span:
        problems.append(
            f"NOT SWAPPED: y-span {y_span:.3f} > z-span {z_span:.3f}. This file "
            f"was written by pre-swap code and is y-up. It will poison training."
        )

    # ---- Streamwise axis -----------------------------------------------------
    # x must be the longest extent -- a vehicle is longest along the flow. DoMINO's
    # drag loss hardcodes normals[:, :, 0] as streamwise.
    x_span = float(v[:, 0].max() - v[:, 0].min())
    if not (x_span > y_span and x_span > z_span):
        problems.append(
            f"x is not the longest axis (x {x_span:.3f}, y {y_span:.3f}, "
            f"z {z_span:.3f}). DoMINO assumes x is streamwise."
        )

    # ---- Channel count and order --------------------------------------------
    if sf.shape[1] != len(CANONICAL_VARIABLES):
        problems.append(
            f"surface_fields has {sf.shape[1]} channels, expected "
            f"{len(CANONICAL_VARIABLES)} {CANONICAL_VARIABLES}"
        )
    else:
        # loss.py:423 requires channel 0 = Cp and channel 1 = x-shear. If they
        # were swapped, Cp would be tiny (O(0.01)) and Cf_x would be O(1). The
        # magnitudes are the only evidence -- the tensor carries no names.
        cp_range = float(sf[:, 0].max() - sf[:, 0].min())
        cfx_range = float(sf[:, 1].max() - sf[:, 1].min())
        if cp_range < 0.5:
            problems.append(
                f"channel 0 spans only {cp_range:.4f} -- too small for Cp. "
                f"Channels may be PERMUTED. loss.py indexes positionally."
            )
        if cfx_range > 0.5:
            problems.append(
                f"channel 1 spans {cfx_range:.4f} -- too large for Cf_x. "
                f"Channels may be PERMUTED."
            )
        # Skin friction opposes motion: streamwise shear is positive-dominant.
        if float(sf[:, 1].mean()) <= 0:
            problems.append(
                f"mean Cf_x = {sf[:, 1].mean():.5f} <= 0. Skin friction opposes "
                f"motion; this should be positive."
            )

    # ---- Normals -------------------------------------------------------------
    # DoMINO's drag loss assumes unit normals.
    norms = np.linalg.norm(n, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        problems.append(
            f"normals are not unit length (min {norms.min():.4f}, "
            f"max {norms.max():.4f})"
        )

    # ---- NaNs ----------------------------------------------------------------
    # A NaN in the targets poisons the loss silently.
    if not np.all(np.isfinite(sf)):
        problems.append("non-finite value in surface_fields")

    # ---- Lengths agree -------------------------------------------------------
    if not (len(sf) == len(n) == len(d["surface_areas"])
            == len(d["surface_mesh_centers"])):
        problems.append("surface array lengths disagree")

    return problems


def main() -> int:
    p = argparse.ArgumentParser(description="Verify preprocessed .npy files")
    p.add_argument("--dataset", choices=list(DATASETS))
    p.add_argument("--dir", type=Path)
    p.add_argument("--all", action="store_true", help="check all three datasets")
    p.add_argument("--root", type=Path, default=Path.home() / "processed")
    args = p.parse_args()

    if args.all:
        targets = [(ds, args.root / ds) for ds in DATASETS]
    elif args.dataset and args.dir:
        targets = [(args.dataset, args.dir)]
    else:
        p.error("give either --all, or both --dataset and --dir")

    total_bad = 0

    for dataset, directory in targets:
        files = sorted(directory.glob("*.npy"))
        if not files:
            print(f"\n  {dataset}: no .npy files in {directory}")
            total_bad += 1
            continue

        ds = DATASETS[dataset]
        print(f"\n  {dataset}  ({len(files)} files, "
              f"swap_yz={ds.swap_yz}, U_inf={ds.u_inf})")

        bad: dict[str, list[str]] = {}

        for i, f in enumerate(files, 1):
            # Progress on one line -- these files are large and unpickling is slow.
            pct = 100 * i // len(files)
            bar = "#" * (pct // 4) + "." * (25 - pct // 4)
            print(f"\r    [{bar}] {i}/{len(files)}  {f.stem:<12}",
                  end="", flush=True)

            problems = check_case(f, dataset)
            if problems:
                bad[f.stem] = problems

        print(f"\r    [{'#' * 25}] {len(files)}/{len(files)}"
              f"{' ' * 20}")

        if bad:
            total_bad += len(bad)
            print(f"\n    ** {len(bad)} of {len(files)} FAILED **\n")
            for name, problems in bad.items():
                print(f"      {name}")
                for prob in problems:
                    print(f"        - {prob}")
            print()
        else:
            print(f"    all {len(files)} OK")

    print()
    if total_bad:
        print(f"  ** {total_bad} PROBLEMS. Do not train on this data. **\n")
        return 1

    print("  all files verified\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())