# test_metrics.py
# ============================================================
# PURPOSE:
#   Exit Criterion 2, executable. The metrics are pure functions, so they can be
#   tested exhaustively with synthetic data -- no meshes, no GPU, no cluster.
#
# WHY TEST METRICS AT ALL -- THEY ARE "JUST ARITHMETIC":
#   Because the cancellation detector is the study's main epistemic safeguard, and
#   a safeguard that silently fails is worse than no safeguard. If the sign of
#   E_bias were inverted, or the area weighting dropped, the model would look
#   HONEST precisely when it was cheating -- and nothing else in the pipeline
#   would notice.
#
#   The test below constructs a model that is deliberately, catastrophically wrong
#   in a way that CANCELS, and asserts that the detector catches it. That single
#   test is the reason this file exists.
#
# RUN:  python -m pytest tests/ -v
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics import cp_field_error, design_ranking, drag_counts, evaluate


# ============================================================
# ACCURACY
# ============================================================

def test_perfect_prediction_is_zero_counts():
    cd = [0.30, 0.31, 0.29, 0.32]
    r = drag_counts(cd, cd)
    assert r["cd_mae_counts"] == pytest.approx(0.0)


def test_one_count_is_one_thousandth_of_cd():
    """
    The unit definition. 1 count = 0.001 Cd. If this ever changes, every number
    in the deliverable sentence changes with it.
    """
    r = drag_counts([0.301], [0.300])
    assert r["cd_mae_counts"] == pytest.approx(1.0)


def test_mae_is_absolute_not_signed():
    """
    Over-prediction and under-prediction must both count as error. If they
    cancelled here, a model that was wildly wrong in both directions would report
    zero MAE -- which is the same class of bug as the cancellation this study is
    built to detect.
    """
    r = drag_counts([0.310, 0.290], [0.300, 0.300])
    assert r["cd_mae_counts"] == pytest.approx(10.0)


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(42)
    true = rng.normal(0.30, 0.02, 30)
    pred = true + rng.normal(0.0, 0.005, 30)
    r = drag_counts(pred, true)
    lo, hi = r["cd_mae_counts_ci95"]
    assert lo <= r["cd_mae_counts"] <= hi
    assert lo < hi                      # a degenerate interval means a broken bootstrap


def test_ci_is_reproducible():
    """
    The bootstrap seed is fixed. A confidence interval that changes between runs
    is not a confidence interval -- it is a random number the reader will mistake
    for one.
    """
    cd_t = [0.30, 0.31, 0.29, 0.32, 0.28, 0.33]
    cd_p = [0.31, 0.30, 0.30, 0.31, 0.29, 0.32]
    assert drag_counts(cd_p, cd_t) == drag_counts(cd_p, cd_t)


# ============================================================
# USABILITY
# ============================================================

def test_perfect_ranking():
    true = [0.28, 0.29, 0.30, 0.31, 0.32]
    pred = [0.30, 0.31, 0.32, 0.33, 0.34]      # constant +0.02 offset
    r = design_ranking(pred, true)
    assert r["spearman_rho"] == pytest.approx(1.0)


def test_a_biased_model_can_still_rank_perfectly():
    """
    ** THE POINT OF HAVING TWO METRICS. **

    This model is off by 20 drag counts -- a large, obvious error. And it ranks
    every design correctly, because the bias is CONSTANT and cancels in every
    pairwise comparison.

    Such a model is entirely usable in an optimization loop. Cd MAE alone would
    condemn it; rho alone would exonerate it. Report both, and understand that
    they answer different questions.
    """
    true = [0.28, 0.29, 0.30, 0.31, 0.32]
    pred = [t + 0.02 for t in true]

    assert drag_counts(pred, true)["cd_mae_counts"] == pytest.approx(20.0)
    assert design_ranking(pred, true)["spearman_rho"] == pytest.approx(1.0)


def test_an_accurate_model_can_rank_terribly():
    """
    ** THE CONVERSE, AND THE MORE DANGEROUS CASE. **

    This model has a small MAE. It also gets the ORDER exactly backwards. It would
    pick the worst design every single time, and its Cd error would look
    reassuring right up until it did.

    rho, not MAE, is the deployability gate.
    """
    true = [0.280, 0.290, 0.300, 0.310, 0.320]
    pred = [0.320, 0.310, 0.300, 0.290, 0.280]      # perfectly reversed

    assert design_ranking(pred, true)["spearman_rho"] == pytest.approx(-1.0)
    assert drag_counts(pred, true)["cd_mae_counts"] < 25   # looks acceptable!


def test_small_n_carries_a_warning():
    """
    At reduced scale we rank ~20 designs. The warning must travel WITH the number,
    into the logs, so it cannot be quietly dropped at writeup time.
    """
    r = design_ranking([0.30, 0.31, 0.29], [0.30, 0.32, 0.28])
    assert r["warning"] != ""


def test_too_few_designs_raises():
    with pytest.raises(ValueError):
        design_ranking([0.30, 0.31], [0.30, 0.32])


# ============================================================
# HONESTY -- the cancellation detector
# ============================================================

def test_perfect_field_has_no_error_and_no_bias():
    cp = np.array([-1.0, 0.5, -0.3, 0.9])
    a = np.array([1.0, 1.0, 1.0, 1.0])
    r = cp_field_error(cp, cp, a)
    assert r["cp_l2_area_wtd"] == pytest.approx(0.0)
    assert r["cp_signed_bias"] == pytest.approx(0.0)


def test_THE_CANCELLATION_DETECTOR():
    """
    ============================================================
    ** THE MOST IMPORTANT TEST IN THIS PROJECT. **
    ============================================================

    A model that over-predicts pressure on the front of the car by +0.5 and
    under-predicts it on the back by -0.5, on equal areas.

    Integrate that and the errors CANCEL EXACTLY. The Cd comes out right. The loss
    curve looked fine. Every headline number says the model works.

    It does not work. It has learned a pressure field that is wrong EVERYWHERE, in
    a way that happens to integrate correctly FOR THIS BODY. Change the body --
    which is precisely what an optimization loop does -- and there is no reason
    whatsoever for the cancellation to survive.

    THE ASSERTION:
        E_L2  is LARGE   -> the local errors are real and big
        E_bias is ZERO   -> and they cancel under integration

    If this test ever fails, the study's main epistemic safeguard is broken and
    the model will look HONEST precisely when it is cheating.
    """
    cp_true = np.array([-1.0, -1.0, -1.0, -1.0])
    cp_pred = np.array([-0.5, -0.5, -1.5, -1.5])      # +0.5 front, -0.5 back
    areas = np.array([1.0, 1.0, 1.0, 1.0])

    r = cp_field_error(cp_pred, cp_true, areas)

    assert r["cp_l2_area_wtd"] == pytest.approx(0.5)       # LARGE local error
    assert r["cp_signed_bias"] == pytest.approx(0.0)       # and it CANCELS
    assert r["cp_bias_to_l2_ratio"] == pytest.approx(0.0)  # the signature


def test_a_consistent_offset_is_NOT_cancellation():
    """
    The counter-case, and the reason bias alone is not enough either.

    This model is uniformly wrong by +0.5 everywhere. Same L2 as the test above.
    But the bias is +0.5, not zero -- because nothing cancels.

    That is a DIFFERENT pathology, and a much more benign one: a consistent offset
    ranks designs perfectly. bias_to_l2_ratio near 1 means "offset"; near 0 means
    "cancellation". The two must not be confused.
    """
    cp_true = np.array([-1.0, -1.0, -1.0, -1.0])
    cp_pred = cp_true + 0.5

    r = cp_field_error(cp_pred, cp_true, np.ones(4))

    assert r["cp_l2_area_wtd"] == pytest.approx(0.5)       # same L2 as above
    assert r["cp_signed_bias"] == pytest.approx(0.5)       # but NO cancellation
    assert r["cp_bias_to_l2_ratio"] == pytest.approx(1.0)  # the signature of offset


def test_area_weighting_actually_weights():
    """
    A large error on a tiny facet must not dominate a small error on a huge one.

    This matters because CFD meshes refine exactly where the interesting physics
    is -- separation lines, stagnation points, wakes. An UNWEIGHTED error would
    therefore be dominated by the regions with the most facets rather than the
    regions carrying the most force, which is precisely backwards.
    """
    cp_true = np.array([0.0, 0.0])
    cp_pred = np.array([1.0, 0.0])           # big error, but on a tiny facet
    areas = np.array([0.01, 99.99])

    r = cp_field_error(cp_pred, cp_true, areas)

    # Unweighted RMS would be 0.707. Area-weighted must be far smaller.
    assert r["cp_l2_area_wtd"] < 0.05


def test_negative_area_raises():
    with pytest.raises(ValueError):
        cp_field_error([0.0], [0.0], [-1.0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cp_field_error([0.0, 0.0], [0.0], [1.0, 1.0])


# ============================================================
# THE COMBINED ENTRY POINT
# ============================================================

def test_evaluate_returns_all_three_metrics():
    """
    One entry point, so that no run can report Cd without also reporting the
    ranking and the bias. A Cd quoted alone IS the self-deception.
    """
    cd_t = [0.28, 0.29, 0.30, 0.31, 0.32]
    cd_p = [0.29, 0.30, 0.31, 0.32, 0.33]
    cp_t = np.array([-1.0, -1.0, 0.5, 0.5])
    cp_p = np.array([-0.9, -1.1, 0.6, 0.4])
    a = np.ones(4)

    r = evaluate(cd_p, cd_t, cp_p, cp_t, a)

    for key in ("cd_mae_counts", "spearman_rho", "cp_signed_bias",
                "cp_l2_area_wtd", "cp_bias_to_l2_ratio"):
        assert key in r, f"{key} missing -- the three-metric contract is broken"


def test_cp_without_areas_raises():
    """You do not get to report Cd and skip the honesty metric."""
    with pytest.raises(ValueError):
        evaluate([0.30, 0.31, 0.32], [0.30, 0.31, 0.32], cp_pred=[0.1, 0.2])