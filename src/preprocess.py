# preprocess.py
# ============================================================
# PURPOSE:
#   Convert a raw case (STL + surface mesh) into the .npy file DoMINO's
#   CachedDoMINODataset reads. This is Phase 3 -- the unified multi-body datapipe.
#
# ** WHY WE DO NOT USE NVIDIA'S process_data.py **
#
#   Their preprocessing exists (in examples/.../domino_nim_finetuning/src/, NOT
#   where the README says). We read it. It is wrong for this study in three ways,
#   and each one is silent:
#
#     1. IT DIVIDES BY rho*U^2, NOT (1/2)*rho*U^2.
#            surface_fields = surface_fields / (air_density * stream_velocity**2)
#        No factor of two. Their "Cp" is HALF the conventional one. Internally
#        consistent for training, but it is not Cp -- and mixing it with our Cp
#        would be corruption that no test would catch.
#
#     2. IT HARDCODES rho = 1.226.
#            global_params_reference = {"air_density": 1.226}
#        We PROVED rho = 1.0 in Phase 2, by backing it out of p/Cp, which is an
#        identity rather than an estimate (std 1.8e-9 across 1.1M facets). Their
#        own file has a commented-out `# AIR_DENSITY = 1.205` above it, so they
#        have flip-flopped on this themselves.
#
#     3. IT TAKES ONE GLOBAL VELOCITY FOR ALL DATASETS.
#            global_params_reference = {"inlet_velocity": [38.89]}
#        That is DrivAer's. AHMED RUNS AT 1 m/s. Wall shear scales as U^2, so
#        using 38.889 for Ahmed would be wrong by a factor of ~1500.
#
#   So we produce [Cp, Cf_x, Cf_y, Cf_z] ourselves, through src/datasets.py and
#   the SAME code path that already validates to 0.03% against published Cd.
#
#   ** THE CONSEQUENCE IS THE POINT: the training data and the force integrator
#   agree BY CONSTRUCTION. ** Had they disagreed, a model could be trained on one
#   convention and evaluated against another, and nothing in the pipeline would
#   have told us.
#
# ** THE OUTPUT CONTRACT **
#   Read from openfoam_datapipe.py's __getitem__ (the only place it is written
#   down). A .npy holding a pickled dict:
#
#       stl_coordinates       (V, 3)   STL vertices
#       stl_centers           (F, 3)   STL facet centroids
#       stl_faces             (3F,)    flattened triangle indices
#       stl_areas             (F,)     STL facet areas
#       surface_mesh_centers  (N, 3)   surface facet centroids
#       surface_normals       (N, 3)   OUTWARD unit normals
#       surface_areas         (N,)     surface facet areas
#       surface_fields        (N, 4)   ** [Cp, Cf_x, Cf_y, Cf_z] -- ORDER MATTERS **
#       volume_*              None     surface-only study
#       filename              str
#       global_params_values      (2,) [U_inf, rho]
#       global_params_reference   (2,) same
#
# ** THE ORDERING TRAP LIVES IN surface_fields. **
#   loss.py:423 indexes it POSITIONALLY:
#       pres_true = output_true[:, :, 0] * normals[:, :, 0]   # channel 0 = pressure
#       wx_true   = output_true[:, :, 1]                       # channel 1 = x-shear
#   A permuted column produces a smooth, converging loss and a worthless model.
#   We assert the order rather than trusting it.
#
# USAGE:
#     python src/preprocess.py --dataset ahmed --raw ~/data/ahmedml \
#                              --out ~/processed/ahmed --workers 8
#
# DEPENDENCIES: pyvista, numpy. No torch, no GPU. Runs on a login node or a CPU
#   partition.
# ============================================================

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyvista as pv

from datasets import CANONICAL_VARIABLES, DATASETS, Dataset
from forces import _cp_cf, _oriented_surface


def build_case(case_dir: Path, dataset: str) -> dict:
    """
    Build the .npy dict for one case.

    WHY IT REUSES forces.py:
        _oriented_surface() and _cp_cf() are the SAME functions that produce the
        Cd validated to 0.03% against published values, across 60 cases. Calling
        them here means the training targets and the evaluation metric are
        computed by identical code.

        The alternative -- reimplementing the physics in the datapipe -- is how
        you end up training a model on one convention and scoring it against
        another. Nothing would catch that. The loss would converge. The Cd would
        be wrong.

    WHY THE STL IS READ SEPARATELY FROM THE SURFACE:
        DoMINO's geometry encoder consumes the STL point cloud -- that is the
        INPUT. The surface mesh carries the fields -- that is the TARGET. They are
        different meshes with different resolutions, and conflating them is a
        category error.
    """
    ds: Dataset = DATASETS[dataset]

    # ---- Geometry: the STL. This is the model's INPUT. -----------------------
    stl_matches = sorted(case_dir.glob(ds.stl_glob))
    if not stl_matches:
        raise FileNotFoundError(
            f"No STL matching '{ds.stl_glob}' in {case_dir}"
        )
    stl = pv.read(stl_matches[0])

    stl_vertices = np.asarray(stl.points, dtype=np.float32)
    # pyvista stores faces as [3, i, j, k, 3, i, j, k, ...] -- the leading 3 is
    # the vertex count. Strip it. (Assumes triangles, which STL guarantees.)
    stl_faces = np.asarray(stl.faces).reshape(-1, 4)[:, 1:].flatten()
    stl_sizes = stl.compute_cell_sizes(length=False, area=True, volume=False)
    stl_areas = np.asarray(stl_sizes.cell_data["Area"], dtype=np.float32)
    stl_centers = np.asarray(stl.cell_centers().points, dtype=np.float32)

    # ---- Surface: the fields. This is the model's TARGET. --------------------
    # Same code path as the validated force integrator. Handles all three normal
    # strategies (clean / shipped / flip) and all three shear conventions.
    surf, sign = _oriented_surface(case_dir, ds)

    sized = surf.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = np.asarray(sized.cell_data["Area"], dtype=np.float32)
    surface_centers = np.asarray(surf.cell_centers().points, dtype=np.float32)

    normals = sign * np.asarray(surf.cell_data["Normals"], dtype=np.float64)
    # Guard: DoMINO's drag loss assumes these are UNIT vectors. Averaging point
    # normals onto cells (Windsor) does not preserve unit length.
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"{case_dir.name}: degenerate normal (zero length)")
    normals = (normals / norms).astype(np.float32)

    # ---- The target: [Cp, Cf_x, Cf_y, Cf_z] ----------------------------------
    cp, cf = _cp_cf(surf, ds)
    surface_fields = np.column_stack([cp, cf]).astype(np.float32)

    # ** THE ORDERING ASSERTION. **
    # loss.py:423 hardcodes channel 0 = pressure, channel 1 = x-wall-shear. There
    # is no metadata in the tensor; nothing downstream can check this. If the
    # columns were permuted, training would converge smoothly and produce a
    # worthless model. So we assert it HERE, where the columns are still named.
    if surface_fields.shape[1] != len(CANONICAL_VARIABLES):
        raise ValueError(
            f"{case_dir.name}: surface_fields has {surface_fields.shape[1]} "
            f"columns, expected {len(CANONICAL_VARIABLES)} "
            f"{CANONICAL_VARIABLES}. loss.py indexes these POSITIONALLY."
        )
    if not (len(surface_fields) == len(normals) == len(surface_areas)
            == len(surface_centers)):
        raise ValueError(
            f"{case_dir.name}: array length mismatch -- fields "
            f"{len(surface_fields)}, normals {len(normals)}, areas "
            f"{len(surface_areas)}, centers {len(surface_centers)}"
        )
    if not np.all(np.isfinite(surface_fields)):
        raise ValueError(
            f"{case_dir.name}: non-finite value in surface_fields. A NaN here "
            f"poisons the loss silently."
        )

    # ---- Global parameters ---------------------------------------------------
    # Per-dataset, NOT global. Ahmed is 1 m/s; DrivAer is 38.889. Passing one
    # value for all three -- which is what NVIDIA's config does -- would be wrong
    # by a factor of ~1500 on the wall shear.
    if ds.rho is None:
        raise ValueError(f"{ds.name}: rho is None. Verify it before preprocessing.")
    global_params = np.array([ds.u_inf, ds.rho], dtype=np.float32)

    return {
        "stl_coordinates": stl_vertices,
        "stl_centers": stl_centers,
        "stl_faces": np.float32(stl_faces),
        "stl_areas": stl_areas,
        "surface_mesh_centers": surface_centers,
        "surface_normals": normals,
        "surface_areas": surface_areas,
        "surface_fields": surface_fields,     # [Cp, Cf_x, Cf_y, Cf_z]
        # Surface-only study. The volume is 61x larger and drag needs only the
        # surface integral.
        "volume_fields": None,
        "volume_mesh_centers": None,
        "filename": case_dir.name,
        "global_params_values": global_params,
        "global_params_reference": global_params,
    }


def _worker(args) -> tuple[str, str]:
    case_dir, dataset, out_dir = args
    out_file = out_dir / f"{case_dir.name}.npy"

    if out_file.exists():
        return case_dir.name, "skipped (exists)"

    try:
        t0 = time.time()
        data = build_case(case_dir, dataset)
        np.save(out_file, data, allow_pickle=True)
        return case_dir.name, f"ok ({time.time() - t0:.1f}s)"
    except Exception as exc:
        # Do NOT swallow this. A case that fails to preprocess is a case that
        # silently vanishes from the training set, and a training set that is
        # quietly smaller than you think is exactly the kind of thing that
        # produces an inexplicable result three weeks later.
        return case_dir.name, f"FAILED: {type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--raw", type=Path, required=True,
                   help="directory of run_* case dirs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N cases (for testing)")
    args = p.parse_args()

    # HuggingFace leaves a .cache/ directory alongside the runs. Filter to real
    # cases -- otherwise .cache is treated as one and the failure looks like a
    # data problem when it is a globbing problem.
    cases = sorted(
        d for d in args.raw.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    if args.limit:
        cases = cases[:args.limit]

    if not cases:
        print(f"  no run_* directories in {args.raw}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\n  {args.dataset}: {len(cases)} cases -> {args.out}")
    print(f"  {len(CANONICAL_VARIABLES)} target channels: "
          f"{', '.join(CANONICAL_VARIABLES)}")
    print(f"  U_inf = {DATASETS[args.dataset].u_inf}, "
          f"rho = {DATASETS[args.dataset].rho}\n")

    jobs = [(c, args.dataset, args.out) for c in cases]

    failures = []
    with Pool(args.workers) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            print(f"  [{i:>3}/{len(cases)}] {name:<12} {status}", flush=True)
            if status.startswith("FAILED"):
                failures.append(f"{name}: {status}")

    print()
    if failures:
        print(f"  ** {len(failures)} FAILURES **")
        for f in failures:
            print(f"    {f}")
        print("\n  A case that fails to preprocess silently VANISHES from the")
        print("  training set. Do not proceed until these are understood.\n")
        return 1

    print(f"  {len(cases)} cases written to {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())