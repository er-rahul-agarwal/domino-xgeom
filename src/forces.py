# forces.py
# ============================================================
# PURPOSE:
#   THE SINGLE SOURCE OF TRUTH FOR DRAG. Nothing else in this project computes
#   Cd. Given a case directory and a dataset identity, integrate the surface
#   traction and return force coefficients.
#
# WHY A SINGLE SOURCE:
#   If two pieces of code compute drag, they will eventually disagree, and there
#   is no way to tell which one is lying.
#
# WHY THIS IS BUILT AND VALIDATED *BEFORE* THE EXPERIMENT:
#   If the integrator carries a sign error or the wrong reference area, a bad
#   cross-geometry transfer number cannot be attributed to physics rather than
#   arithmetic -- and that distinction cannot be recovered afterwards.
#
# ============================================================
# WHAT PHASE-2 VALIDATION FOUND: FIVE BUGS, ALL SILENT
#
# The first version of this file was wrong five times over. Every intermediate
# version returned a confident, plausible float. NONE raised an error. They were
# found ONLY by comparing against the published Cd -- which is the entire argument
# for Exit Criterion 2.
#
#   1. INWARD NORMALS.  Ahmed's Cd came out at -0.218 against a published +0.238.
#      Same magnitude, opposite sign.
#
#   2. WALL-SHEAR SIGN CONVENTION.  OpenFOAM's wallShearStress is the stress the
#      WALL exerts on the FLUID -- the negative of what a drag integral needs.
#      SYMPTOM: viscous drag came out NEGATIVE. Skin friction opposes motion; a
#      negative viscous drag is physically impossible, and that should have been
#      the first thing anyone noticed.
#
#   3. rho IS 1.0, NOT 1.225.  These are incompressible solvers in KINEMATIC
#      units -- pressure is stored as p/rho -- so rho never appears in their
#      inputs and assuming sea-level air is the natural mistake.
#      FOUND BY: each dataset publishes BOTH p and Cp, so their ratio IS q.
#        Ahmed:   p/Cp = 0.500000    (std 1.8e-9 over 1.1M facets)  -> rho = 1.0
#        DrivAer: p/Cp = 756.177161  (std 2.7e-5 over 8.8M facets)  -> rho = 1.0
#      Not an estimate -- an identity. With rho=1.225 Ahmed's viscous term was 22%
#      low and its Cd was 5% short.
#
#   4. THE THREE DATASETS NEED THREE DIFFERENT NORMAL STRATEGIES (see below).
#      A single code path silently fails on two of them.
#
#   5. clean() DESTROYS WINDSOR'S CELL ARRAYS.  At tolerances loose enough to
#      affect Windsor's topology, clean() merges CELLS while leaving the cell
#      arrays at their old length -- 4.99M cells against a 9.92M-entry cfxavg.
#      pyvista only WARNS. The resulting Cd is computed from fields that no longer
#      correspond to the facets they belong to.
#
# ------------------------------------------------------------
# THE THREE NORMAL STRATEGIES -- none of this is documented anywhere
#
#   AHMED    "clean"     58,308 open edges, is_manifold=False. The mesh is
#                        stitched from per-patch exports whose vertices are
#                        duplicated at the seams: geometrically closed,
#                        topologically full of cracks. To VTK this is an OPEN
#                        surface, and for an open surface "outward" is undefined
#                        -- there is no inside to be outside of. So
#                        auto_orient_normals returns an INCONSISTENT mix.
#                        clean(1e-6) merges the coincident vertices. Open edges
#                        drop to 0, is_manifold becomes True, and
#                        auto_orient_normals then orients correctly on its own.
#
#   WINDSOR  "shipped"   The dataset PROVIDES a Normals array -- unit length,
#                        consistently oriented. Use it, and never call clean().
#
#   DRIVAER  "flip"      Ships no normals, and clean() does NOT help: 7,383 open
#                        edges before, 7,383 after. Those cracks are not duplicate
#                        vertices. But consistent_normals=True makes the
#                        orientation locally coherent, leaving a single GLOBAL
#                        sign -- and that sign is settled against the published
#                        Cd. Un-flipped: -0.2533. Flipped: +0.31093, against a
#                        published 0.31092.
#
# ------------------------------------------------------------
# RESULT -- Exit Criterion 2, gate is 2%:
#
#     ahmed     0.238554  vs  0.238486    0.068 counts   0.029%   PASS
#     windsor   0.320846  vs  0.322511    1.665 counts   0.516%   PASS
#     drivaer   0.310927  vs  0.310925    0.002 counts   0.001%   PASS
# ============================================================
#
# INPUTS:  a case directory, and a dataset key ("ahmed"|"windsor"|"drivaer")
# OUTPUTS: dict of Cd, Cd_pressure, Cd_viscous, Cl
# DEPENDENCIES: pyvista, numpy. No torch, no GPU.
# ============================================================

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyvista as pv

from datasets import DATASETS, Dataset

# Verified in Phase 0 from the STL bounding boxes of all three datasets. DoMINO's
# drag loss hardcodes the same assumption (normals[:, :, 0]); had it been false,
# the integral loss would have optimized a force component that is not drag.
X, Y, Z = 0, 1, 2


def _oriented_surface(case_dir: Path, ds: Dataset) -> tuple[pv.PolyData, float]:
    """
    Load the surface, put fields on CELLS, and return it with correctly oriented
    normals -- plus the global sign to apply to them.

    Returns (surface, sign) where `sign` multiplies the normals. It is +1 except
    for DrivAer, where the orientation is locally consistent but globally
    inverted.

    WHY THREE BRANCHES AND NOT ONE:
        See the module docstring. Each dataset fails differently, and each failure
        is silent.

    WHY point_data_to_cell_data() FOR WINDSOR:
        Area-weighted integration needs ONE value and ONE area per facet. Point
        data lives at vertices, which are SHARED between facets -- there is no
        per-facet area to weight by.
    """
    matches = sorted(case_dir.glob(ds.surface.file_glob))
    if not matches:
        raise FileNotFoundError(
            f"No surface matching '{ds.surface.file_glob}' in {case_dir}. Note "
            f"Windsor is .vtu while Ahmed and DrivAer are .vtp -- the glob is "
            f"per-dataset for a reason."
        )

    surf = pv.read(matches[0])

    if ds.surface.scope == "point":
        surf = surf.point_data_to_cell_data()          # Windsor only

    # ---- WINDSOR: use the shipped normals ------------------------------------
    if ds.surface.normals == "shipped":
        if "Normals" not in surf.cell_data:
            raise ValueError(
                f"{ds.name} is configured as normals='shipped' but the mesh has "
                f"no Normals array. Do NOT fall back to clean() -- on Windsor it "
                f"merges cells and invalidates every cell array."
            )
        # point_data_to_cell_data() AVERAGES the vertex normals onto each facet,
        # and the average of unit vectors is not a unit vector. Renormalize.
        n = np.asarray(surf.cell_data["Normals"], dtype=np.float64)
        surf.cell_data["Normals"] = n / np.linalg.norm(n, axis=1, keepdims=True)
        return surf, +1.0

    # An UnstructuredGrid has no PolyData filters. Extract first.
    if not isinstance(surf, pv.PolyData):
        surf = surf.extract_surface()

    # ---- AHMED: clean, then let VTK orient -----------------------------------
    if ds.surface.normals == "clean":
        surf = surf.clean(tolerance=1e-6).triangulate()
        if surf.n_open_edges != 0:
            raise ValueError(
                f"{case_dir.name}: {surf.n_open_edges} open edges remain after "
                f"clean(). auto_orient_normals is NOT well-defined on an open "
                f"surface and any Cd from this mesh is untrustworthy."
            )
        surf = surf.compute_normals(
            cell_normals=True, point_normals=False,
            auto_orient_normals=True, consistent_normals=True,
        )
        return surf, +1.0

    # ---- DRIVAER: locally consistent, globally flipped -----------------------
    if ds.surface.normals == "flip":
        # NOT auto_orient_normals -- there is no manifold for it to orient
        # against, and clean() cannot make one (7,383 open edges before and
        # after). consistent_normals only guarantees neighbours agree with each
        # other, which leaves exactly one global sign to determine. That sign was
        # determined against the published Cd and is fixed here.
        surf = surf.compute_normals(
            cell_normals=True, point_normals=False,
            auto_orient_normals=False, consistent_normals=True,
        )
        return surf, -1.0

    raise ValueError(
        f"Unknown normals strategy '{ds.surface.normals}' for {ds.name}. "
        f"Must be one of: clean | shipped | flip."
    )


def _cp_cf(surf: pv.PolyData, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (Cp, Cf) as (N,) and (N,3) COEFFICIENT arrays, per facet.

    WHY COEFFICIENTS:
        The canonical output space is [Cp, Cf_x, Cf_y, Cf_z]. This is FORCED, not
        chosen: Windsor publishes no dimensional pressure at all.

    WHY WE READ *THEIR* Cp RATHER THAN COMPUTING IT FROM p:
        (a) Windsor has no p. (b) Using their Cp makes our Cd comparable to their
        Cd by construction, rather than by our getting rho right. rho then enters
        only through the viscous term -- which is how we caught rho=1.0 in the
        first place.

    THE WALL-SHEAR BRANCH -- THREE TRAPS IN ONE:
        Ahmed, DrivAer:  one 3-vector, DIMENSIONAL (Pa), and reported with the
                         OPPOSITE sign (wall-on-fluid). Negate; divide by q.
        Windsor:         THREE SEPARATE SCALARS, ALREADY coefficients. Do NOT
                         divide by q -- that would divide by 450 and make the
                         viscous term vanish.
        And Windsor's FILE ORDER is cfx, cfz, cfy -- Z BEFORE Y. We read by NAME
        in canonical (x, y, z) order from datasets.py. Reading in file order
        permutes two channels -- and since the DoMINO surface losses index
        positionally (loss.py:423), a permuted channel yields a smooth,
        converging loss and a worthless model.
    """
    cp = np.asarray(surf.cell_data[ds.surface.cp], dtype=np.float64)

    if ds.surface.wss_vector is not None:
        if ds.rho is None:
            raise ValueError(
                f"{ds.name} stores DIMENSIONAL wall shear and needs rho to "
                f"non-dimensionalize it, but rho is None in datasets.py. Verify "
                f"it by the p/Cp method. Do NOT assume 1.225 -- it is 1.0."
            )
        q = 0.5 * ds.rho * ds.u_inf**2
        tau = np.asarray(surf.cell_data[ds.surface.wss_vector], dtype=np.float64)
        cf = -tau / q                       # negate: OpenFOAM reports wall-on-fluid
    else:
        assert ds.surface.wss_components is not None
        cf = np.column_stack([
            np.asarray(surf.cell_data[name], dtype=np.float64)
            for name in ds.surface.wss_components   # canonical x,y,z -- NOT file order
        ])

    return cp, cf


def surface_forces(case_dir: Path, dataset: str) -> dict:
    r"""
    Integrate surface traction; return force coefficients.

    THE PHYSICS:
        Drag is the streamwise component of the integrated surface traction --
        pressure acting along the inward normal, plus wall shear acting
        tangentially:

            Cd = [ sum( -Cp * n_x * a )  +  sum( Cf_x * a ) ] / A_ref
                 \_____________________/    \_______________/
                    pressure drag              viscous drag
                 (dominant on a bluff body)

        The minus sign is because pressure acts INTO the surface, i.e. along
        -n_hat where n_hat is the OUTWARD normal.

    A_ref IS THE *CONSTANT* REFERENCE:
        Because the study reports a design RANKING. Under a per-case area, a morph
        that changes frontal area moves its own Cd for reasons that have nothing
        to do with the flow -- the ranking would partly reflect geometry rather
        than aerodynamics.
    """
    ds = DATASETS[dataset]
    surf, sign = _oriented_surface(case_dir, ds)

    sized = surf.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    normals = sign * np.asarray(surf.cell_data["Normals"], dtype=np.float64)

    cp, cf = _cp_cf(surf, ds)

    if not (len(cp) == len(cf) == len(areas) == len(normals)):
        raise ValueError(
            f"Array length mismatch in {case_dir.name}: cp={len(cp)}, "
            f"cf={len(cf)}, areas={len(areas)}, normals={len(normals)}. On "
            f"Windsor this is the signature of clean() having merged cells "
            f"without updating the cell arrays."
        )

    f_pressure = -(cp[:, None] * normals * areas[:, None]).sum(axis=0)
    f_viscous = (cf * areas[:, None]).sum(axis=0)
    f_total = f_pressure + f_viscous

    if ds.a_ref_const is None:
        raise ValueError(
            f"No constant reference area for {ds.name}. Fix datasets.py; do not "
            f"guess one here."
        )
    a_ref = ds.a_ref_const

    # Lift along each dataset's OWN vertical axis -- Windsor is y-up, the others
    # z-up. NOT cross-comparable, and this study does not compare it. Recorded
    # because a wrong Cl is often the first sign of a wrong Cd.
    vertical = Y if ds.name == "windsor" else Z

    return {
        "case": case_dir.name,
        "dataset": ds.name,
        "Cd": float(f_total[X] / a_ref),
        "Cd_pressure": float(f_pressure[X] / a_ref),
        "Cd_viscous": float(f_viscous[X] / a_ref),
        "Cl": float(f_total[vertical] / a_ref),
        "n_cells": int(len(areas)),
        "wetted_area": float(areas.sum()),
    }


def published_cd(case_dir: Path, dataset: str) -> float:
    """
    Read the published CONSTANT-reference Cd.

    ** THE TRAP. READ BEFORE EDITING. **

        Ahmed:    force_mom_<i>.csv          = CONSTANT reference
        Windsor:  force_mom_<i>.csv          = CONSTANT reference
        DrivAer:  force_mom_constref_<i>.csv = CONSTANT reference
                  force_mom_<i>.csv          = VARIABLE (per-morph) reference

    The UNSUFFIXED filename means OPPOSITE THINGS in the source datasets and the
    target -- across exactly the boundary this study transfers over. Loading
    force_mom_1.csv uniformly, which is the obvious implementation, introduces a
    47-DRAG-COUNT error on Ahmed. The effect we are trying to MEASURE is smaller
    than that.

    So the filename is never written here. datasets.py owns it.
    """
    ds = DATASETS[dataset]

    matches = sorted(case_dir.glob(ds.const_force_csv))
    if ds.name in ("ahmed", "windsor"):
        matches = [m for m in matches if "varref" not in m.name]

    if not matches:
        raise FileNotFoundError(
            f"No constant-reference force file ('{ds.const_force_csv}') in "
            f"{case_dir}."
        )

    with matches[0].open() as fh:
        rows = list(csv.reader(fh))

    # Ahmed writes "cd, cl" and " 2.38e-01, -9.45e-02" -- note the spaces.
    header = [h.strip() for h in rows[0]]
    values = [v.strip() for v in rows[1]]

    if ds.cd_column not in header:
        raise KeyError(
            f"'{ds.cd_column}' not in {matches[0].name} header {header}. Ahmed "
            f"and Windsor use lowercase 'cd'; DrivAer uses 'Cd'."
        )

    return float(values[header.index(ds.cd_column)])


def validate(case_dir: Path, dataset: str) -> dict:
    """
    Compare our Cd against the published one. THIS IS EXIT CRITERION 2.

        < 2%    PASS
        2-5%    INVESTIGATE -- normal orientation, or an unclean mesh
        > 5%    FAIL -- sign error or wrong reference area. DO NOT PROCEED.

    WHY VALIDATE AT ALL, RATHER THAN TRUSTING THE CODE:
        Every failure mode here is silent. A flipped normal, a wrong rho, a
        double-divided shear term -- all produce a number, none produce an error.
        The published Cd is the only thing that can say the number is right.
    """
    ours = surface_forces(case_dir, dataset)
    theirs = published_cd(case_dir, dataset)

    abs_err = abs(ours["Cd"] - theirs)
    rel_err = abs_err / abs(theirs)
    verdict = "PASS" if rel_err < 0.02 else "INVESTIGATE" if rel_err < 0.05 else "FAIL"

    return {
        **ours,
        "Cd_published": theirs,
        "abs_error_counts": float(abs_err * 1000),   # 1 count = 0.001 Cd
        "rel_error": float(rel_err),
        "verdict": verdict,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python src/forces.py <case_dir> <ahmed|windsor|drivaer>")
        sys.exit(1)

    r = validate(Path(sys.argv[1]), sys.argv[2])

    print(f"\n  {r['dataset']}  /  {r['case']}   ({r['n_cells']:,} facets)")
    print(f"  {'-' * 48}")
    print(f"  Cd (ours)         {r['Cd']:>14.6f}")
    print(f"       pressure     {r['Cd_pressure']:>14.6f}")
    print(f"       viscous      {r['Cd_viscous']:>14.6f}")
    print(f"  Cd (published)    {r['Cd_published']:>14.6f}")
    print(f"  {'-' * 48}")
    print(f"  error             {r['abs_error_counts']:>11.3f} counts")
    print(f"  relative          {r['rel_error'] * 100:>13.3f} %")
    print(f"  VERDICT           {r['verdict']:>14}")
    print()