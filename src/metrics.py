# metrics.py
# ============================================================
# PURPOSE:
#   The three numbers this study reports. Nothing else is a headline.
#
#     ACCURACY   Cd mean absolute error, in drag counts.
#     USABILITY  Spearman rank correlation across designs.
#     HONESTY    Area-weighted Cp error, and its SIGNED BIAS.
#
# WHY THREE AND NOT ONE -- THE SELF-DECEPTION THIS FILE EXISTS TO PREVENT:
#   A surrogate can obtain approximately the right Cd by CANCELLING ERRORS --
#   over-predicting pressure forward, under-predicting aft, and landing on a
#   plausible integral. It is right for the wrong reasons, and it will fail the
#   moment it is used to CHOOSE BETWEEN designs, which is the only thing anyone
#   actually wants a surrogate for.
#
#   Cd alone cannot see this. The signed bias can. That is the whole argument.
#
# DEPENDENCIES: numpy, scipy. No pyvista, no torch. Pure functions -- arrays in,
#   dicts out. Testable on a laptop with no data and no GPU.
# ============================================================

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

# 1 drag count = 0.001 Cd. The unit aerodynamicists actually speak in, and the
# unit the deliverable sentence is written in.
COUNTS_PER_CD = 1000.0


def drag_counts(cd_pred, cd_true) -> dict:
    """
    ACCURACY. Mean absolute Cd error, in drag counts.

    WHY DRAG COUNTS RATHER THAN A PERCENTAGE:
        Because that is how the answer will be used. "Within 15 counts" means
        something to an aerodynamicist; "4.8% relative error" does not, and it
        obscures the fact that the same percentage means very different things on
        a Cd of 0.24 and a Cd of 0.31.

    WHY Cd RATHER THAN WALL SHEAR:
        Wall shear is small, noisy, dominated by near-wall behaviour the model
        never resolves, and secondary to pressure drag on a bluff body. It is the
        metric most likely to look poor for reasons unrelated to the research
        question. It is retained as SUPPORTING evidence -- it is where
        separation-resolution failure becomes visible -- but never as the headline.

    ALSO RETURNS A BOOTSTRAP CI:
        With 20 held-out designs at reduced scale, the MAE is a noisy statistic.
        Reporting it without an interval invites the reader to over-read it. The
        interval is the honest width of what we actually know.
    """
    cd_pred = np.asarray(cd_pred, dtype=np.float64)
    cd_true = np.asarray(cd_true, dtype=np.float64)

    if cd_pred.shape != cd_true.shape:
        raise ValueError(f"shape mismatch: {cd_pred.shape} vs {cd_true.shape}")

    errors = np.abs(cd_pred - cd_true) * COUNTS_PER_CD
    mae = float(errors.mean())

    # Bootstrap the mean. 2000 resamples is plenty for a 95% interval and costs
    # nothing on arrays this small.
    rng = np.random.default_rng(seed=0)   # fixed: the CI must be reproducible
    n = len(errors)
    boot = np.array([
        errors[rng.integers(0, n, n)].mean() for _ in range(2000)
    ])

    return {
        "cd_mae_counts": mae,
        "cd_mae_counts_ci95": [float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))],
        "cd_max_error_counts": float(errors.max()),
        "n_designs": int(n),
    }


def design_ranking(cd_pred, cd_true) -> dict:
    """
    USABILITY. Spearman rank correlation between predicted and true Cd.

    WHY THIS, AND NOT Cd MAE, IS THE DEPLOYABILITY GATE:
        A surrogate exists to choose between designs. A model that cannot order
        two cars by drag is USELESS FOR OPTIMIZATION -- regardless of how small its
        absolute error is. You could have a 5-count MAE and still pick the wrong
        design every time, if the errors are correlated with the thing you are
        optimizing.

        Conversely a model with a large but CONSISTENT bias ranks perfectly and is
        entirely usable: the offset cancels in the comparison.

        So: report both, and understand that they answer different questions.
        Cd MAE answers "is it accurate?". rho answers "is it useful?".

    WHY THE p-VALUE MATTERS HERE MORE THAN USUAL:
        At reduced scale we rank ~20 designs. Spearman on n=20 is noisy, and a
        rho of 0.6 may not be distinguishable from chance. Report the p-value and
        do not quote rho without it.
    """
    cd_pred = np.asarray(cd_pred, dtype=np.float64)
    cd_true = np.asarray(cd_true, dtype=np.float64)

    if len(cd_pred) < 3:
        raise ValueError(
            f"Spearman on {len(cd_pred)} designs is meaningless. Need >= 3, and "
            f"realistically >= 15 to say anything."
        )

    rho, p = spearmanr(cd_pred, cd_true)

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "n_designs": int(len(cd_pred)),
        # A blunt reminder, carried in the output itself so it survives into the
        # logs and cannot be forgotten at writeup time.
        "warning": (
            "n < 15 -- rho is not reliable at this sample size"
            if len(cd_pred) < 15 else ""
        ),
    }


def cp_field_error(cp_pred, cp_true, areas) -> dict:
    r"""
    HONESTY. Area-weighted Cp error, and the SIGNED BIAS.

    With per-facet areas a_i, weights w_i = a_i / sum(a), and error
    e_i = Cp_pred - Cp_true:

        E_L2   = sqrt( sum_i w_i * e_i^2 )     <- MAGNITUDE of local error
        E_bias =       sum_i w_i * e_i         <- ** THE CANCELLATION DETECTOR **

    ** WHY E_bias IS THE MOST IMPORTANT NUMBER IN THIS FILE **

        Consider a model that over-predicts pressure on the front of the car and
        under-predicts it on the back, by equal amounts. Integrate that and the
        errors CANCEL: you get approximately the right Cd. The model looks good.

        It is not good. It has learned a pressure field that is locally wrong
        everywhere, and it happens to be wrong in a way that integrates to
        roughly the right answer FOR THIS BODY. Change the body -- which is
        exactly what an optimization loop does -- and there is no reason the
        cancellation should survive.

        LARGE E_L2 WITH E_bias NEAR ZERO IS THE SIGNATURE OF THIS.
        It is invisible to Cd. It is invisible to the loss curve. It is only
        visible here.

        This is registered as H3, and it is EXPECTED at small fine-tuning budgets.
        If observed: REPORT IT. It is a finding, not a bug, and it is the single
        most useful thing this study can tell a practitioner. Do NOT "fix" it away.

    WHY AREA-WEIGHTED:
        An unweighted mean over facets over-counts small facets, and CFD meshes
        refine exactly where the interesting physics is -- separation lines,
        stagnation points, wakes. An unweighted error is therefore dominated by
        the regions with the most facets, not the regions with the most force.
    """
    cp_pred = np.asarray(cp_pred, dtype=np.float64)
    cp_true = np.asarray(cp_true, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)

    if not (cp_pred.shape == cp_true.shape == areas.shape):
        raise ValueError(
            f"shape mismatch: cp_pred={cp_pred.shape}, cp_true={cp_true.shape}, "
            f"areas={areas.shape}. All three must be per-facet."
        )
    if np.any(areas < 0):
        raise ValueError("negative facet area -- the mesh or the reader is broken")

    w = areas / areas.sum()
    e = cp_pred - cp_true

    l2 = float(np.sqrt(np.sum(w * e**2)))
    bias = float(np.sum(w * e))

    return {
        "cp_l2_area_wtd": l2,
        "cp_signed_bias": bias,          # <- the cancellation detector
        # A dimensionless read on the pathology: bias small relative to L2 means
        # the local errors are large and cancelling. Near 1 means a consistent
        # offset (which ranks fine). Near 0 means cancellation (which does not).
        "cp_bias_to_l2_ratio": float(abs(bias) / l2) if l2 > 0 else 0.0,
        "n_facets": int(len(e)),
    }


def evaluate(cd_pred, cd_true, cp_pred=None, cp_true=None, areas=None) -> dict:
    """
    Compute everything for one arm, in the schema the plan logs.

    WHY ONE ENTRY POINT:
        So that no run can accidentally report Cd without also reporting the
        ranking and the bias. The whole point of the three-metric design is that
        they are reported TOGETHER -- a Cd quoted alone is exactly the
        self-deception this file exists to prevent.
    """
    out: dict = {}
    out.update(drag_counts(cd_pred, cd_true))
    out.update(design_ranking(cd_pred, cd_true))

    if cp_pred is not None:
        if cp_true is None or areas is None:
            raise ValueError(
                "cp_pred given without cp_true and areas. The honesty metric "
                "needs all three -- do not report Cd without it."
            )
        out.update(cp_field_error(cp_pred, cp_true, areas))

    return out