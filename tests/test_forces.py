# test_forces.py
# ============================================================
# PURPOSE:
#   Exit Criterion 2, executable. Assert that the force integrator reproduces the
#   published Cd for each dataset.
#
# WHY THIS IS A TEST AND NOT A SCRIPT:
#   Because it must FAIL LOUDLY when someone breaks it -- and someone will. The
#   five bugs Phase 2 found were all silent, and every one of them produced a
#   confident, plausible float. The only thing that caught them was comparison
#   against ground truth. That comparison must therefore run automatically,
#   forever, not by hand when someone remembers to.
#
#   Specifically, this test would catch:
#     - a regression in any of the three normal strategies
#     - someone "simplifying" the three branches into one code path
#     - someone changing rho back to 1.225 because it looks more physical
#     - someone loading force_mom_1.csv uniformly across datasets
#     - someone calling clean() on Windsor
#
# ** SCOPE LIMITATION -- READ THIS. **
#   These tests run on ONE case per dataset (run_1), because that is what is on
#   the development laptop. The plan requires >= 20 cases per dataset.
#
#   One case proves the METHOD is right. It does NOT prove the method is ROBUST
#   across morphs -- a single case can pass by luck, and the datasets contain
#   geometries that vary substantially. In particular, DrivAer's "flip" strategy
#   assumes the global normal sign is the same for every morph, which is plausible
#   but UNVERIFIED.
#
#   Exit Criterion 2 is therefore NOT fully cleared by this file. It clears when
#   test_integrator_robustness_across_morphs (below) runs on real data.
#
# RUN:  python -m pytest tests/test_forces.py -v
# ============================================================

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forces import published_cd, surface_forces, validate


# Where the Phase-0 reconnaissance cases live. Overridable, because this test
# must run unchanged on the cluster, where the path is different.
DATA_ROOT = Path(os.environ.get("DOMINO_DATA", r"D:\data"))

CASES = {
    "ahmed":   DATA_ROOT / "ahmedml" / "run_1",
    "windsor": DATA_ROOT / "windsorml" / "run_1",
    "drivaer": DATA_ROOT / "drivaerml" / "run_1",
}

# Exit Criterion 2. The plan sets these bands:
#     < 2%   pass
#     2-5%   investigate -- normal orientation, or an unclean mesh
#     > 5%   FAIL -- sign error or wrong reference area. Do not proceed.
GATE = 0.02

# What we actually achieved in Phase 2. Recorded so a REGRESSION is caught, not
# just an outright failure: if Ahmed's error jumps from 0.03% to 1.5%, that still
# passes the gate but something has changed and we want to know.
ACHIEVED = {
    "ahmed":   0.00029,      # 0.068 counts
    "windsor": 0.00516,      # 1.665 counts
    "drivaer": 0.00001,      # 0.002 counts
}


def _skip_if_missing(dataset: str) -> Path:
    case = CASES[dataset]
    if not case.is_dir():
        pytest.skip(
            f"{case} not present. Set DOMINO_DATA to the directory containing "
            f"ahmedml/, windsorml/, drivaerml/."
        )
    return case


# ============================================================
# EXIT CRITERION 2 -- the integrator reproduces published Cd
# ============================================================

@pytest.mark.parametrize("dataset", ["ahmed", "windsor", "drivaer"])
def test_integrator_matches_published_cd(dataset):
    """
    ** THIS IS EXIT CRITERION 2. **

    Integrate the GROUND-TRUTH surface fields and compare against the Cd the
    dataset authors published. If these disagree, the integrator is wrong -- and
    every Cd this project ever produces, from every model, is wrong with it.

    This must pass BEFORE any model is trained. If the integrator carries a sign
    error, a bad cross-geometry transfer number cannot be attributed to physics
    rather than arithmetic, and that distinction cannot be recovered after the
    fact.
    """
    case = _skip_if_missing(dataset)
    r = validate(case, dataset)

    assert r["verdict"] == "PASS", (
        f"{dataset}: Cd = {r['Cd']:.6f} vs published {r['Cd_published']:.6f} "
        f"({r['abs_error_counts']:.2f} counts, {r['rel_error']*100:.2f}%). "
        f"Gate is {GATE*100}%. DO NOT PROCEED -- see the module docstring of "
        f"forces.py for the five bugs that produce exactly this symptom."
    )
    assert r["rel_error"] < GATE


@pytest.mark.parametrize("dataset", ["ahmed", "windsor", "drivaer"])
def test_no_regression_against_phase2_baseline(dataset):
    """
    Catch a SILENT DEGRADATION, not just an outright failure.

    A change that takes Ahmed from 0.03% to 1.5% error still passes the 2% gate --
    and it means something is broken. This asserts we stay within an order of
    magnitude of what Phase 2 actually achieved.
    """
    case = _skip_if_missing(dataset)
    r = validate(case, dataset)

    ceiling = max(ACHIEVED[dataset] * 10, 0.001)
    assert r["rel_error"] < ceiling, (
        f"{dataset}: error {r['rel_error']*100:.3f}% is far above the "
        f"{ACHIEVED[dataset]*100:.3f}% achieved in Phase 2. Still under the gate, "
        f"but something has changed."
    )


# ============================================================
# THE PHYSICS MUST BE PHYSICAL
# ============================================================

@pytest.mark.parametrize("dataset", ["ahmed", "windsor", "drivaer"])
def test_viscous_drag_is_positive(dataset):
    """
    ** THE BUG THAT SHOULD HAVE BEEN CAUGHT BY INSPECTION. **

    Skin friction OPPOSES motion. Viscous drag is positive. Always. A negative
    viscous drag is not a small error -- it is physically impossible, and it means
    the wall-shear sign convention is wrong.

    OpenFOAM's wallShearStress is the stress the WALL exerts on the FLUID, which
    is the negative of what a drag integral needs. Phase 2 got this wrong, and the
    only symptom was a negative number that nobody looked at for an hour.
    """
    case = _skip_if_missing(dataset)
    r = surface_forces(case, dataset)

    assert r["Cd_viscous"] > 0, (
        f"{dataset}: viscous drag = {r['Cd_viscous']:.6f}. Skin friction opposes "
        f"motion; this cannot be negative. Check the wall-shear sign convention."
    )


@pytest.mark.parametrize("dataset", ["ahmed", "windsor", "drivaer"])
def test_pressure_drag_dominates(dataset):
    """
    On a bluff body, pressure drag dominates viscous drag. This is the defining
    characteristic of bluff-body aerodynamics and it is why Cd -- not wall shear --
    is the headline metric of this study.

    If viscous ever exceeded pressure, either the body is not bluff (it is) or the
    integrator is wrong (it would be).
    """
    case = _skip_if_missing(dataset)
    r = surface_forces(case, dataset)

    assert r["Cd_pressure"] > r["Cd_viscous"], (
        f"{dataset}: pressure {r['Cd_pressure']:.4f} <= viscous "
        f"{r['Cd_viscous']:.4f}. On a bluff body this is not possible."
    )


@pytest.mark.parametrize("dataset", ["ahmed", "windsor", "drivaer"])
def test_cd_is_in_a_plausible_range(dataset):
    """
    A road car or bluff body has Cd roughly in [0.1, 0.6]. A number outside that
    is not a bad prediction -- it is a broken integrator. A flipped normal, for
    instance, produces a NEGATIVE Cd, which is exactly what Phase 2 saw first.
    """
    case = _skip_if_missing(dataset)
    r = surface_forces(case, dataset)

    assert 0.1 < r["Cd"] < 0.6, (
        f"{dataset}: Cd = {r['Cd']:.4f}, outside any plausible range for a bluff "
        f"body. Suspect normal orientation before suspecting the physics."
    )


# ============================================================
# THE REFERENCE-AREA TRAP
# ============================================================

def test_ahmed_and_drivaer_use_different_force_files():
    """
    ** THE 47-DRAG-COUNT TRAP. **

        Ahmed:   force_mom_<i>.csv          = CONSTANT reference
        DrivAer: force_mom_constref_<i>.csv = CONSTANT reference
                 force_mom_<i>.csv          = VARIABLE (per-morph) reference

    The unsuffixed filename means OPPOSITE THINGS in the source datasets and the
    target -- across exactly the boundary this study transfers over.

    This test asserts the config knows that. If someone "tidies up" datasets.py by
    making all three use force_mom_*.csv, this fails -- and it should, because the
    resulting error on Ahmed is 47 drag counts, which is LARGER THAN THE EFFECT
    THIS STUDY IS TRYING TO MEASURE.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from datasets import DATASETS

    assert "constref" not in DATASETS["ahmed"].const_force_csv
    assert "constref" not in DATASETS["windsor"].const_force_csv
    assert "constref" in DATASETS["drivaer"].const_force_csv, (
        "DrivAer's constant-reference file is force_mom_CONSTREF_*.csv. Its "
        "unsuffixed force_mom_*.csv is the VARIABLE-reference one -- the opposite "
        "of Ahmed and Windsor."
    )


@pytest.mark.parametrize("dataset", ["ahmed", "windsor", "drivaer"])
def test_published_cd_is_readable(dataset):
    """
    The header case differs across datasets: Ahmed and Windsor write lowercase
    'cd' (with a leading space in the value row); DrivAer writes 'Cd'. A lookup by
    the wrong case KeyErrors -- or, under a lenient parser, silently misses.
    """
    case = _skip_if_missing(dataset)
    cd = published_cd(case, dataset)
    assert 0.1 < cd < 0.6


# ============================================================
# ROBUSTNESS -- NOT YET SATISFIED
# ============================================================

@pytest.mark.skip(
    reason="Requires >= 20 cases per dataset. Only run_1 is on the dev laptop. "
           "Enable when the data lands on ARC -- EXIT CRITERION 2 IS NOT FULLY "
           "CLEARED UNTIL THIS RUNS."
)
@pytest.mark.parametrize("dataset", ["ahmed", "windsor", "drivaer"])
def test_integrator_robustness_across_morphs(dataset):
    """
    ** THE GAP IN EXIT CRITERION 2, STATED HONESTLY. **

    Everything above validates ONE case per dataset. That proves the METHOD is
    right. It does NOT prove the method is ROBUST across morphs.

    The specific risk: DrivAer's "flip" strategy assumes the global normal sign is
    the SAME for every morph. That is plausible -- they came off the same meshing
    pipeline -- but it is UNVERIFIED, and if a single morph came out with the
    opposite handedness, its Cd would come back NEGATIVE and we would notice; but
    a subtler variation might not announce itself at all.

    The plan requires >= 20 cases per dataset. Until this runs, EC2 is
    "method verified, robustness pending" -- and the writeup must say so.
    """
    root = DATA_ROOT / f"{dataset}ml"
    cases = sorted(d for d in root.iterdir() if d.is_dir())[:20]

    if len(cases) < 20:
        pytest.skip(f"only {len(cases)} cases available; need 20")

    failures = []
    for case in cases:
        r = validate(case, dataset)
        if r["verdict"] != "PASS":
            failures.append(
                f"{case.name}: {r['rel_error']*100:.2f}% "
                f"(Cd {r['Cd']:.4f} vs {r['Cd_published']:.4f})"
            )

    assert not failures, (
        f"{dataset}: {len(failures)}/{len(cases)} cases outside the 2% gate:\n  "
        + "\n  ".join(failures)
    )