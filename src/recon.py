# recon.py
# ============================================================
# PURPOSE:
#   Phase 0 reconnaissance. Open one raw surface mesh from each of the three
#   datasets (AhmedML, WindsorML, DrivAerML) and report everything needed to
#   populate the compatibility table (plan, Sec. 7.4). This script writes no
#   data and trains nothing. It exists solely to convert assumptions into
#   verified facts before any GPU time is spent.
#
# WHY THIS APPROACH:
#   Every dominant failure mode in this study is SILENT. A permuted variable
#   order produces a smoothly converging loss and a worthless model
#   (plan Rem 4.1, verified at loss.py:423). A wrong streamwise axis makes the
#   integral loss optimize a force component that is not drag
#   (plan Rem 4.2). A wrong reference area shifts every Cd by a constant
#   factor. None of these raise an error. The only defence is to inspect the
#   raw files and assert what we find.
#
#   Phase 0 has already found that the three datasets disagree in ways the
#   plan did not anticipate:
#     - Windsor ships its surface as .vtu, not .vtp
#     - force_mom_*.csv means CONSTANT reference in Ahmed/Windsor but
#       VARIABLE reference in DrivAer (the suffixed file inverts)
#     - Ahmed publishes no frontal area at all; Windsor and DrivAer do
#     - Ahmed uses lowercase headers ("cd"), DrivAer uppercase ("Cd")
#   This script is written to survive those differences rather than assume
#   them away.
#
# INPUTS:
#   A case directory containing one geometry file (.stl) and one surface
#   file (.vtp or .vtu), plus optional force/geometry CSVs.
#
# OUTPUTS:
#   A dict per case, printed and written to recon_<dataset>.json.
#   Nothing is modified on disk apart from that JSON.
#
# DEPENDENCIES:
#   pyvista, numpy. No torch, no CUDA, no PhysicsNeMo. Runs on a laptop.
# ============================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


# Surface data may be stored in either container. Windsor uses .vtu for what
# is semantically a surface; Ahmed and DrivAer use .vtp. We must not assume.
SURFACE_SUFFIXES = (".vtp", ".vtu")


def find_one(case_dir: Path, patterns: list[str]) -> Path | None:
    """
    Return the first file in case_dir matching any of the given glob patterns.

    WHY THIS FUNCTION EXISTS:
        Filenames differ across the three datasets (ahmed_1.stl,
        windsor_1.stl, drivaer_1.stl; boundary_1.vtp vs boundary_1.vtu).
        Hardcoding names would make this script dataset-specific, which
        defeats its purpose.

    WHY THIS IMPLEMENTATION:
        Returns None rather than raising, so a missing optional file (e.g.
        Ahmed has no geo_ref.csv) is reported as a finding rather than
        crashing the run.
    """
    for pattern in patterns:
        matches = sorted(case_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def summarize_fields(mesh: pv.DataSet) -> dict:
    """
    Report every data array on the mesh: its name, where it lives, its shape,
    and its value range.

    WHY THIS FUNCTION EXISTS:
        This is the core of the compatibility table. Three questions are
        answered here that cannot be answered any other way:

        1. WHAT ARE THE FIELDS CALLED? The datapipe must be told, and the
           names are not guaranteed to match across datasets.

        2. ARE THEY ON POINTS OR ON CELLS? Area-weighted integration requires
           cell data. Getting this wrong silently mis-weights every facet.

        3. ARE THEY DIMENSIONAL OR ALREADY NON-DIMENSIONAL? A pressure field
           spanning roughly +/-500 is Pascals. One spanning roughly +/-2 is
           already a pressure coefficient. There is no metadata that says
           which; the value range is the only evidence. Applying
           non-dimensionalization twice, or not at all, produces a wrong Cd
           with no error message.

    WHY THIS IMPLEMENTATION:
        Reports n_components explicitly, because a 3-vector wall-shear field
        stored as three separate scalar arrays and one stored as a single
        (N,3) array require different handling in the datapipe -- and both
        conventions appear in the wild.
    """
    out = {"point_arrays": {}, "cell_arrays": {}}

    for scope, store in (("point_arrays", mesh.point_data),
                         ("cell_arrays", mesh.cell_data)):
        for name in store.keys():
            arr = np.asarray(store[name])
            out[scope][name] = {
                # Order matters: the surface loss functions index positionally
                # (loss.py:423 -- channel 0 must be pressure, channel 1 must be
                # x-wall-shear). The order arrays appear in here is the order
                # the datapipe will see them unless we intervene.
                "n_components": 1 if arr.ndim == 1 else int(arr.shape[1]),
                "dtype": str(arr.dtype),
                # The value range is our ONLY evidence for units. See docstring.
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
            }

    # The order of keys is itself a finding. Record it explicitly rather than
    # relying on the reader to notice it in the dict above.
    out["point_array_order"] = list(mesh.point_data.keys())
    out["cell_array_order"] = list(mesh.cell_data.keys())
    return out


def infer_streamwise_axis(bounds: tuple) -> dict:
    """
    Report the bounding-box extents so the streamwise axis can be identified.

    WHY THIS FUNCTION EXISTS:
        drag_loss_fn multiplies pressure by normals[:, :, 0] -- the
        x-component of the surface normal (plan Rem 4.2). The drag loss
        therefore ASSUMES the freestream runs along +x, in every dataset.
        If any body is oriented differently, the integral loss optimizes a
        force component that is not drag, silently.

    WHY THIS IMPLEMENTATION:
        A road vehicle is longest along the flow direction. The longest
        bounding-box extent is therefore strong evidence for the streamwise
        axis -- but it is EVIDENCE, not proof. This function reports the
        extents and its inference separately, so a human decides. It does not
        return a bare answer.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    extents = {"x": xmax - xmin, "y": ymax - ymin, "z": zmax - zmin}
    longest = max(extents, key=extents.get)
    return {
        "bounds": [float(b) for b in bounds],
        "extents": {k: float(v) for k, v in extents.items()},
        # Inference, not fact. A human confirms this against the published
        # force conventions before it is trusted.
        "longest_axis_is": longest,
        "matches_domino_assumption_x": longest == "x",
    }


def read_csv_raw(path: Path | None) -> dict | None:
    """
    Read a small CSV as raw header/value strings, without pandas.

    WHY THIS FUNCTION EXISTS:
        The force CSVs are the ground truth against which the Phase-2 force
        integrator is validated (plan Crit 8.1). They are also where the
        reference-area trap lives: force_mom_1.csv means CONSTANT reference in
        Ahmed and Windsor but VARIABLE reference in DrivAer.

    WHY THIS IMPLEMENTATION:
        Deliberately does NOT parse into named columns. DrivAer writes "Cd",
        Ahmed writes "cd" with a leading space in the value line. Any lookup by
        column name would either KeyError or silently miss. We record the raw
        text and let a human read it, because a parser that "helpfully" handles
        the difference would hide the fact that a difference exists.
    """
    if path is None or not path.exists():
        return None
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return {
        "filename": path.name,
        "raw_lines": lines,
    }


def inspect(case_dir: Path) -> dict:
    """
    Inspect one simulation case and return everything the compatibility table
    needs.

    WHY THIS FUNCTION EXISTS:
        Exit Criterion 7.1 requires the compatibility table to be populated
        "from inspected data, not from documentation." NVIDIA's own README has
        already been shown to contradict its own repository (plan Rem 15.2), so
        documentation is not evidence. This function is the inspection.
    """
    stl = find_one(case_dir, ["*.stl"])
    surface = find_one(case_dir, [f"*{s}" for s in SURFACE_SUFFIXES])

    result: dict = {
        "case_dir": str(case_dir),
        "geometry_file": stl.name if stl else None,
        "surface_file": surface.name if surface else None,
        # The container format is itself a finding: Windsor ships its surface
        # as .vtu while Ahmed and DrivAer use .vtp. A loader written against
        # .vtp alone will simply not find Windsor's surface.
        "surface_format": surface.suffix if surface else None,
    }

    # ---- Geometry (STL) -----------------------------------------------------
    if stl is not None:
        geom = pv.read(stl)
        result["stl"] = {
            "n_cells": int(geom.n_cells),
            "n_points": int(geom.n_points),
            **infer_streamwise_axis(geom.bounds),
        }

    # ---- Surface fields -----------------------------------------------------
    if surface is not None:
        surf = pv.read(surface)
        result["surface"] = {
            "n_cells": int(surf.n_cells),
            "n_points": int(surf.n_points),
            "vtk_type": type(surf).__name__,
            **summarize_fields(surf),
        }

    # ---- Ground-truth forces and geometry parameters ------------------------
    # These are the Phase-2 validation targets and the source of the
    # reference-area convention. Read raw; do not parse.
    result["csvs"] = {}
    for pattern in ["force_mom_?.csv", "force_mom_*ref_?.csv",
                    "geo_ref_?.csv", "geo_parameters_?.csv"]:
        for path in sorted(case_dir.glob(pattern)):
            result["csvs"][path.name] = read_csv_raw(path)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/recon.py <case_dir> [<case_dir> ...]")
        sys.exit(1)

    results = []
    for arg in sys.argv[1:]:
        case_dir = Path(arg)
        if not case_dir.is_dir():
            raise FileNotFoundError(
                f"Not a directory: {case_dir}\n"
                f"Expected a case directory containing an .stl and a "
                f".vtp or .vtu surface file."
            )
        print(f"--- inspecting {case_dir} ---", file=sys.stderr)
        results.append(inspect(case_dir))

    print(json.dumps(results, indent=2))