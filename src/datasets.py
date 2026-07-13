# datasets.py
# ============================================================
# PURPOSE:
#   The single source of truth for how the three datasets differ. Every field
#   name, unit, reference area, freestream velocity, and file convention is
#   recorded here ONCE, with its provenance. Nothing else in this project may
#   hardcode a field name or a reference area.
#
# WHY THIS FILE EXISTS:
#   Phase 0 reconnaissance found that AhmedML, WindsorML, and DrivAerML are
#   incompatible in five separate ways, ALL of which fail silently:
#
#     1. force_mom_<i>.csv means CONSTANT reference area in Ahmed and Windsor,
#        but VARIABLE reference area in DrivAer. Same filename, opposite
#        meaning. Loading it uniformly -- the obvious thing to do -- mixes
#        conventions across exactly the boundary this study transfers over.
#        For Ahmed that is a 47-drag-count error; the effect we are trying to
#        measure is smaller than that.
#
#     2. Windsor ships its surface as .vtu with fields on POINTS. Ahmed and
#        DrivAer ship .vtp with fields on CELLS. A loader written for one finds
#        nothing on the other.
#
#     3. Windsor stores wall shear as three SEPARATE scalar arrays, in the
#        order cfx, cfz, cfy -- z before y. Stacking them in array order yields
#        a permuted vector. The DoMINO surface losses index positionally
#        (loss.py:423: channel 0 is pressure, channel 1 is x-wall-shear), so a
#        permutation produces a smoothly converging loss and a worthless model.
#
#     4. Freestream velocity spans 39x (Ahmed 1 m/s, DrivAer 38.889 m/s).
#        Wall shear scales as U^2, so the dimensional shear fields differ by a
#        factor of ~1500 between datasets. Ahmed's raw range is +/-0.036 Pa;
#        DrivAer's is +/-40 Pa. Training on both without normalizing would let
#        DrivAer dominate the loss entirely.
#
#     5. Windsor publishes NO dimensional pressure -- only cpavg. Ahmed and
#        DrivAer publish both. The canonical output space is therefore not a
#        preference; the coefficient form is the only representation all three
#        datasets can supply.
#
# WHY THIS IMPLEMENTATION:
#   Plain frozen dataclasses, no cleverness. Every value carries a comment
#   naming its source: a paper, a CSV we inspected, or a recon output. Nothing
#   here is inferred or remembered. If a value cannot be traced, it is not in
#   this file.
#
# INPUTS:  None. This is a constants module.
# OUTPUTS: DATASETS, a dict keyed by dataset name.
#
# DEPENDENCIES:
#   None (stdlib only). Deliberately importable without pyvista, so that
#   config and tests can read it cheaply.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceFields:
    """
    Where the surface data lives, and what it is called.

    WHY THIS EXISTS:
        Nothing is named the same thing in any two datasets. There is no
        convention to fall back on; the names must be looked up.
    """

    # Container format. Windsor is the odd one out.
    # Source: recon_all.json, Phase 0.
    file_glob: str

    # "point" or "cell". Windsor puts fields on points; the others on cells.
    # This matters for area-weighted integration, which requires cell data.
    # Source: recon_all.json, Phase 0.
    scope: str

    # Pressure COEFFICIENT field. Present in all three; the only common
    # pressure representation (Windsor has no dimensional pressure at all).
    cp: str

    # Dimensional pressure, in Pa. None where the dataset does not publish it.
    # Source: recon_all.json, Phase 0.
    p: str | None

    # Wall shear. Either ONE 3-vector field (Ahmed, DrivAer -- dimensional, Pa)
    # or THREE separate scalar fields (Windsor -- already coefficients).
    # If wss_vector is set, wss_components is None, and vice versa.
    wss_vector: str | None

    # Windsor only. Given here in the CANONICAL order (x, y, z) -- note this is
    # NOT the order they appear in the file, which is cfx, cfz, cfy. Reading
    # them in file order would silently permute y and z.
    # Source: recon_all.json, Phase 0.
    wss_components: tuple[str, str, str] | None

    # ** HOW TO GET OUTWARD NORMALS. Three datasets, three different answers. **
    #
    # This is not a style choice. Each dataset needs a different method, and using
    # the wrong one produces a plausible Cd that is silently wrong. Determined
    # empirically in Phase 2 by validating against published Cd:
    #
    #   "clean"     Ahmed. The raw mesh has 58,308 open edges -- it is stitched
    #               from per-patch exports with duplicated vertices at the seams.
    #               To VTK this is an OPEN surface, and for an open surface
    #               "outward" is undefined, so auto_orient_normals returns an
    #               inconsistent mix. clean(1e-6) merges the coincident vertices:
    #               open edges drop to 0, is_manifold becomes True, and
    #               auto_orient_normals then works correctly on its own.
    #
    #   "shipped"   Windsor. The dataset PROVIDES a Normals array -- unit length,
    #               consistently oriented. Use it.
    #               ** DO NOT CALL clean() ON WINDSOR. ** At tolerances loose
    #               enough to affect its topology, clean() MERGES CELLS and leaves
    #               the cell arrays at their old length -- 4.99M cells against a
    #               9.92M-entry cfxavg array. The data no longer corresponds to the
    #               geometry, and pyvista only warns.
    #
    #   "flip"      DrivAer. Ships no normals, AND clean() does not help: 7,383
    #               open edges before, 7,383 after. The cracks are not duplicate
    #               vertices. But consistent_normals=True makes the orientation
    #               locally coherent, leaving only a single GLOBAL sign to fix --
    #               and that sign is settled against the published Cd. Verified:
    #               un-flipped gives Cd = -0.253; flipped gives +0.31093 against a
    #               published 0.31092.
    normals: str   # "clean" | "shipped" | "flip"


@dataclass(frozen=True)
class Dataset:
    """Everything needed to load one dataset and produce comparable numbers."""

    name: str

    # ---- Flow conditions ----------------------------------------------------
    # Freestream velocity, m/s. Fixed across all cases within a dataset (all
    # three datasets vary geometry only, never boundary conditions).
    u_inf: float

    # Kinematic viscosity, m^2/s. Recorded for completeness; not used for
    # surface-only work, but needed to reproduce or extend the datasets.
    nu: float | None

    # Density, kg/m^3.
    #
    # ** NOT PUBLISHED BY ANY DATASET. ** These are incompressible OpenFOAM-style
    # solvers working in KINEMATIC units -- pressure is stored as p/rho -- so rho
    # never appears in their inputs. Assuming 1.225 (sea-level air) is the
    # natural mistake and it is WRONG.
    #
    # Backed out empirically in Phase 2, not guessed. Each dataset publishes BOTH
    # p and Cp, so their ratio IS q:
    #
    #     Ahmed:  p / Cp = 0.500000  (std 1.8e-9 over 1.1M facets)
    #             q = 0.5, U = 1.0  ->  rho = 2q/U^2 = 1.0
    #
    # With rho=1.225 the Ahmed viscous drag came out 22% low and Cd was 5% short.
    # With rho=1.0 it lands within 0.03% of the published value.
    #
    # Windsor is immune -- it publishes Cf directly, so rho never enters its Cd.
    rho: float | None

    # ---- Reference area -----------------------------------------------------
    # The CONSTANT reference area, m^2. This is the one this study uses; see
    # the module docstring and const_force_csv below.
    a_ref_const: float | None

    # ---- Ground-truth force files -------------------------------------------
    # ** THE TRAP. Read this before touching either field. **
    #
    # Ahmed and Windsor:  force_mom_<i>.csv         = CONSTANT reference
    #                     force_mom_varref_<i>.csv  = variable (per-case) area
    # DrivAer:            force_mom_<i>.csv         = VARIABLE (per-case) area
    #                     force_mom_constref_<i>.csv = CONSTANT reference
    #
    # The unsuffixed filename means OPPOSITE things. This study uses the
    # CONSTANT-reference file everywhere, because the headline usability metric
    # is a design RANKING (plan Def. 3.2): with a per-case area, a morph that
    # changes frontal area moves its own Cd for reasons that have nothing to do
    # with the flow, and the ranking would partly reflect geometry rather than
    # aerodynamics.
    #
    # Source: AhmedML paper Table 3 ("force_mom_i.csv: ... using constant
    # A_ref"; "force_mom_varref_i.csv: ... case-dependant A_ref"), and
    # DrivAerML geo_ref_<i>.csv, which publishes aRef (per-case) alongside
    # aRefRef (constant) -- confirming which is which arithmetically:
    #   Cd_const = Cd_var * (aRef / aRefRef)
    #   0.3035 * (2.223 / 2.17) = 0.3109  [matches force_mom_constref_1.csv]
    const_force_csv: str
    var_force_csv: str

    # Force-coefficient column header. Ahmed and Windsor write lowercase "cd"
    # (with a leading space in the value row); DrivAer writes "Cd". A lookup by
    # the wrong case KeyErrors, or silently misses under a lenient parser.
    # Source: recon_all.json, Phase 0.
    cd_column: str

    # ---- File naming --------------------------------------------------------
    stl_glob: str
    surface: SurfaceFields

    # ---- Coordinate transform -----------------------------------------------
    # ** WINDSOR IS Y-UP. AHMED AND DRIVAER ARE Z-UP. **
    #
    # WindsorML paper: "for all the simulations in this work, y is upwards, which
    # differs to the original Windsor geometry where z is upwards." Recon
    # confirms it: Windsor's STL spans y in [0, 0.475] (height) and z in [-0.194,
    # +0.194] (width).
    #
    # THE PROBLEM THIS CAUSES:
    #   The model has FOUR output channels: [Cp, Cf_x, Cf_y, Cf_z]. For Ahmed and
    #   DrivAer, channel 2 is LATERAL shear. For Windsor, channel 2 is VERTICAL
    #   shear. The network cannot know that -- it has ONE neuron for channel 2,
    #   and it is being asked to fit two different physical quantities into it
    #   depending on which body it happens to be looking at.
    #
    #   The geometry encoder has the same problem: it sees y-up STL coordinates
    #   for Windsor and z-up for the others.
    #
    #   Drag is SAFE either way -- channel 1 is streamwise in all three, verified
    #   from the STL bounds. But H5 (does a second body family buy anything?)
    #   depends on Windsor contributing real signal, and a quarter of the source
    #   data with two channels transposed contributes signal we cannot quantify.
    #
    # THE FIX:
    #   Swap y and z for Windsor -- in the STL coordinates, the surface normals,
    #   AND the shear channels. All three, or the transform is worse than the
    #   problem: y-up geometry feeding z-up targets is a NEW inconsistency.
    #
    #   This transforms Windsor away from its published convention. That is a
    #   deliberate, documented choice, and the writeup states it: "Windsor's
    #   published convention is y-up; we transformed it into the z-up convention
    #   shared by Ahmed and DrivAer, so that all four target channels have
    #   consistent physical meaning across the source distribution."
    #
    # NOTE: forces.py does NOT apply this. It works in each dataset's own frame
    # and validates against each dataset's own published Cd -- which is correct,
    # and which is why it reports Cl along Windsor's y (matching cl = -0.0611 to
    # 0.4 counts). The swap is a TRAINING-DATA transform, applied in
    # preprocess.py only.
    swap_yz: bool

    # ---- Provenance ---------------------------------------------------------
    source: str


# The streamwise axis is +x in ALL THREE datasets. Verified in Phase 0 from the
# STL bounding boxes (recon_all.json: longest_axis_is == "x" for each).
#
# This matters because drag_loss_fn multiplies pressure by normals[:, :, 0] --
# the x-component of the surface normal (plan Rem 4.2). Had any dataset been
# oriented differently, the integral loss would have optimized a force
# component that is not drag, without complaint. It does not. One trap defused.
STREAMWISE_AXIS = "x"
# ** THE VERTICAL AXIS IS NOT CONSISTENT ACROSS DATASETS. **
#
# WindsorML paper: "for all the simulations in this work, y is upwards, which
# differs to the original Windsor geometry where z is upwards." Our recon
# confirms it: Windsor's STL spans y in [0, 0.475] (height) and z in [-0.194,
# +0.194] (width). Ahmed and DrivAer have z up and y lateral.
#
# CONSEQUENCE: drag is SAFE -- x is streamwise in all three (see
# STREAMWISE_AXIS above), and drag depends only on the x-component. But LIFT,
# SIDE FORCE, and the y/z components of wall shear are NOT comparable across
# datasets without a coordinate swap. The canonical Cf_y and Cf_z channels mean
# different physical directions in Windsor than in the other two.
#
# This study reports drag, so the headline metric is unaffected. But the model
# is TRAINED on all four channels, and a permuted y/z in one third of the
# training data is exactly the kind of silent corruption this project exists to
# avoid. Decide explicitly in Phase 3 whether to swap Windsor's axes into the
# Ahmed/DrivAer convention. Do not leave this to chance.
VERTICAL_AXIS = {"ahmed": "z", "windsor": "y", "drivaer": "z"}

# ** WINDSOR IS YAWED. **
# WindsorML paper: "The model is yawed by -2.5 degrees around the y-axis ... so
# generating a positive side force consistent with the experiment." Ahmed and
# DrivAer are at zero yaw. Windsor's flow is therefore ASYMMETRIC by design.
# Recorded because it is a real physical difference between the source bodies
# and the target, and belongs in the study's stated limitations.
YAW_DEGREES = {"ahmed": 0.0, "windsor": -2.5, "drivaer": 0.0}


# The canonical output space. Fixed order, enforced at the datapipe boundary.
#
# This ordering is NOT a preference. The DoMINO surface loss functions index
# positionally (loss.py:423-427): channel 0 MUST be pressure, channel 1 MUST be
# x-wall-shear. A permuted target tensor produces a well-formed, smoothly
# decreasing loss that regresses the wrong channels against each other.
#
# The COEFFICIENT form is likewise forced, not chosen: Windsor publishes no
# dimensional pressure, so [Cp, Cf_x, Cf_y, Cf_z] is the only representation
# all three datasets can supply.
CANONICAL_VARIABLES = ("Cp", "Cf_x", "Cf_y", "Cf_z")


DATASETS: dict[str, Dataset] = {
    "ahmed": Dataset(
        name="ahmed",
        # AhmedML paper, Boundary conditions: "the free-stream velocity
        # U_inf = 1 m s^-1". The Reynolds number (7.68e5 on body height) is set
        # through the viscosity, NOT the velocity. This is easy to get wrong --
        # Windsor and DrivAer both run at realistic road speeds, and 1 m/s
        # looks like a typo. It is not.
        u_inf=1.0,
        nu=3.75e-7,          # AhmedML paper, Boundary conditions
        rho=1.0,             # VERIFIED Phase 2: p/Cp = 0.5 exactly, U=1.0
        # AhmedML paper, Geometry: "a reference area of A_ref = 0.112".
        # NOT published in any CSV -- the dataset ships no geo_ref file. This
        # number exists only in the paper.
        a_ref_const=0.112,
        const_force_csv="force_mom_*.csv",        # CONSTANT (paper Table 3)
        var_force_csv="force_mom_varref_*.csv",   # variable  (paper Table 3)
        cd_column="cd",      # lowercase, and the value row has a leading space
        stl_glob="ahmed_*.stl",
        surface=SurfaceFields(
            file_glob="boundary_*.vtp",
            scope="cell",
            cp="static(p)_coeffMean",
            p="pMean",
            wss_vector="wallShearStressMean",   # 3-vector, DIMENSIONAL (Pa)
            wss_components=None,
            normals="clean",     # 58,308 open edges -> 0. Verified Phase 2.
        ),
        swap_yz=False,       # already z-up
        source="arxiv.org/abs/2407.20801 + recon_all.json",
    ),

    "windsor": Dataset(
        name="windsor",
        u_inf=30.0,          # WindsorML paper: "freestream velocity of 30 m s-1"
        nu=None,             # TODO: not yet retrieved from the WindsorML paper
        # Not needed: Windsor publishes Cf directly, so rho never enters its Cd.
        # Left None deliberately -- if code tries to use it, that is a bug worth
        # crashing on rather than silently defaulting.
        rho=None,
        # WindsorML paper: "The reference frontal area is defined by the vehicle
        # height and width and for the baseline geometry is 0.112 m^2."
        # Coincidentally identical to Ahmed's -- both are ~1/4-scale models.
        a_ref_const=0.112,
        const_force_csv="force_mom_*.csv",        # CONSTANT (matches Ahmed)
        var_force_csv="force_mom_varref_*.csv",   # variable
        cd_column="cd",
        stl_glob="windsor_*.stl",
        surface=SurfaceFields(
            # ** .vtu, not .vtp. Windsor ships its SURFACE in a volume
            # container. A glob for *.vtp finds nothing here. **
            file_glob="boundary_*.vtu",
            # ** Fields on POINTS, not cells. Area-weighted integration needs
            # cell data, so Windsor must be converted before integrating. **
            scope="point",
            cp="cpavg",
            p=None,          # ** No dimensional pressure published. At all. **
            wss_vector=None,
            # ** THREE separate scalars, and the file order is cfx, cfz, cfy --
            # z BEFORE y. Listed here in canonical (x, y, z) order. Reading
            # them in file order silently swaps two channels. **
            # These are ALREADY coefficients; unlike Ahmed and DrivAer, they
            # need no division by q.
            wss_components=("cfxavg", "cfyavg", "cfzavg"),
            normals="shipped",   # ** NEVER clean() -- it destroys the cell arrays **
        ),
        swap_yz=True,        # ** y-up -> z-up. See the field docstring. **
        source="arxiv.org/abs/2407.19320 + recon_all.json",
    ),

    "drivaer": Dataset(
        name="drivaer",
        # DrivAerML paper: "freestream velocity of U_inf = 38.889 m/s".
        u_inf=38.889,
        nu=1.507e-5,         # DrivAerML paper
        # VERIFIED Phase 2, by the same p/Cp method used for Ahmed -- not assumed
        # by analogy. DrivAer publishes both pMeanTrim and CpMeanTrim:
        #     p / Cp = 756.1772  (std 2.7e-5 over 8.8M facets)
        #     U = 38.889  ->  rho = 2q/U^2 = 1.0000000000036
        # Same convention as Ahmed: incompressible solver in kinematic units.
        rho=1.0,
        # DrivAerML paper: "The reference frontal area A = 2.17 m^2 is used for
        # force and moment coefficients." Confirmed independently by
        # geo_ref_1.csv, which reports aRefRef = 2.17 (constant) alongside
        # aRef = 2.223 (this morph's own area).
        a_ref_const=2.17,
        # ** INVERTED relative to Ahmed and Windsor. The suffixed file is the
        # constant one here. **
        const_force_csv="force_mom_constref_*.csv",   # CONSTANT
        var_force_csv="force_mom_*.csv",              # variable (per-morph)
        cd_column="Cd",      # uppercase, unlike Ahmed and Windsor
        stl_glob="drivaer_*.stl",
        surface=SurfaceFields(
            file_glob="boundary_*.vtp",
            scope="cell",
            cp="CpMeanTrim",
            p="pMeanTrim",
            wss_vector="wallShearStressMeanTrim",   # 3-vector, DIMENSIONAL (Pa)
            wss_components=None,
            normals="flip",      # clean() does nothing (7,383 -> 7,383). Global flip.
        ),
        swap_yz=False,       # already z-up
        source="arxiv.org/abs/2408.11969 + recon_all.json",
    ),
}


def dynamic_pressure(dataset: Dataset, rho: float = 1.225) -> float:
    """
    Return q = 0.5 * rho * U_inf^2 for a dataset.

    WHY THIS FUNCTION EXISTS:
        Ahmed and DrivAer publish wall shear DIMENSIONALLY (Pa); Windsor
        publishes it already as a coefficient. Converting the first two into
        the canonical Cf requires q, and q differs by a factor of ~1500 across
        the datasets because U_inf spans 39x.

    WHY THIS IMPLEMENTATION:
        rho defaults to 1.225 kg/m^3 (sea-level standard) but is exposed as an
        argument rather than hardcoded, because NONE of the three datasets
        publishes rho directly -- they are incompressible simulations specified
        by kinematic viscosity. The published Cp fields let us verify the
        assumption in Phase 2: if our computed Cp from p and q does not match
        the dataset's own Cp field, rho is wrong. Do not trust this default
        until that check passes.
    """
    return 0.5 * rho * dataset.u_inf**2


if __name__ == "__main__":
    # Smoke test: print the table, so the traps are visible at a glance.
    print(f"{'':10} {'U_inf':>8} {'A_ref':>8}  {'const force file':<28} "
          f"{'surface':<18} {'scope':<6}")
    for ds in DATASETS.values():
        a = f"{ds.a_ref_const}" if ds.a_ref_const else "UNKNOWN"
        print(f"{ds.name:10} {ds.u_inf:>8} {a:>8}  {ds.const_force_csv:<28} "
              f"{ds.surface.file_glob:<18} {ds.surface.scope:<6}")
    print()
    print("Note: the CONSTANT-reference file is the UNSUFFIXED one for Ahmed")
    print("and Windsor, but the SUFFIXED one for DrivAer. See module docstring.")