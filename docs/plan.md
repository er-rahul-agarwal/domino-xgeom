# Cross-Geometry Generalization of DoMINO

**Can a neural operator trained only on simple bluff bodies predict drag on a real car?**

Working plan. This document is the project's memory — across sessions and collaborators,
nothing persists except what is written here. It is a *prediction* when first written and
a record of *findings* thereafter. **Update it after every phase.**

> **Working with an AI assistant?** Read [Appendix C — Collaboration Contract](#appendix-c--collaboration-contract)
> **first**. The dominant failure modes of this study are silent, and an agreeable
> collaborator will help you commit them.

---

## Status

| | |
|---|---|
| **Phase** | **0 complete** (2026-07-11). [Exit Criterion 0](#exit-criterion-0--cleared) cleared. |
| **Next** | Phase 1 — baseline reproduction on DrivAerML |
| **Blocker** | Cluster access. All `environment.md` cluster fields are TODO. |
| **Last result** | `recon_all.json` — one case from each dataset inspected |

**Upstream, pinned:**

```
physicsnemo         @ 59aaf59b48901bd8df2c9bad83f6d05ea47d8c04
physicsnemo-curator @ 86533e581b3550326d89e97cb4d4126e7061b416
physicsnemo-cfd     @ 0d2305e1777351569b1795ce38884ee945491d28
```

**Single source of truth for dataset differences: [`src/datasets.py`](../src/datasets.py).**
No other file may hardcode a field name, a reference area, or a freestream velocity.

---

## Scale

**This study runs at reduced scale.** Every component is scale-invariant by
construction: case counts and run budgets live in `conf/base.yaml`, never in code.
Scaling to the full study means editing four numbers and requesting more wall-clock —
nothing is rewritten.

| | **This study** | Full study |
|---|---|---|
| Ahmed | **80** | 500 |
| Windsor | **40** | 355 |
| DrivAer | **40** (20 holdout + 20 FT pool) | 500 |
| Arms | **A1, A2, A3** | A1, A2, A3a, A3 |
| Recovery budgets | **N ∈ {0, 5, 20}** | N ∈ {0, 5, 10, 25, 50, 100} |
| Seeds | **1** | 3 |
| **Total runs** | **5** | 22 |
| **Raw storage** | **~55 GB** | ~590 GB |

**Why.** Storage on the ARC cluster is shared and finite; this is one of several
projects competing for a 500 GB home quota. The reduced scale was chosen deliberately
to preserve **every technique** in the full study while cutting only **statistical
power**.

**What is preserved.** All seven silent traps ([§5](#5-the-seven-traps)) are still
live and still must be handled. The datapipe still unifies three incompatible dataset
formats. The force integrator is still validated to 2% against ground truth. The
fine-tuning path still hits the epoch trap. **Nothing about the engineering is easier
at this scale.**

**What is lost — stated plainly, not buried:**

- **Statistical power.** Spearman ρ over 20 designs is noisy. Report the confidence
  interval, and do not over-read it.
- **Seed spread.** One seed per N. **The recovery curve is indicative, not evidential.**
  At N=5, the identity of the five cases dominates the outcome — with one seed, that
  variance is invisible rather than absent.
- **It is not a curve.** Three points (N = 0, 5, 20) is a line segment. **Do not
  annotate a "knee" that three points cannot resolve.**
- **A3a is cut.** H5 (source diversity) cannot be adjudicated. It remains registered
  and unanswered.
- **Wake evidence is cut** (49 GB per case). **H2 becomes *asserted*, not *shown*** —
  the exact gap the full plan called out as a reviewer's first question.
- **Exit Criterion 1 is relaxed** from R² > 0.95 to **R² > 0.80**, because A1 trains on
  ~20 cases rather than 400. See [EC1](#exit-criterion-1).

**The honest summary:** this study establishes whether the *pipeline* works and gives a
*directional* answer on transfer. **It does not establish the recovery cost with
publishable precision.** That is a scale limitation, not a design flaw, and it is
stated wherever a number is reported.

## Contents

- [1. The question](#1-the-question)
- [2. Registered hypotheses](#2-registered-hypotheses)
- [3. What we measure](#3-what-we-measure)
- [4. What DoMINO actually learns](#4-what-domino-actually-learns)
- [5. The seven traps](#5-the-seven-traps) ← **read before touching anything**
- [6. Phases](#6-phases)
- [7. Risk register](#7-risk-register)
- [8. Quick reference](#8-quick-reference)
- [Appendix A — Project structure](#appendix-a--project-structure)
- [Appendix B — Session restart card](#appendix-b--session-restart-card)
- [Appendix C — Collaboration contract](#appendix-c--collaboration-contract)

---

## 1. The question

> **If a DoMINO surrogate is trained only on simple bluff bodies (Ahmed and Windsor), can
> it predict the drag of a realistic road car (DrivAer) it has never seen — and if not,
> how many high-fidelity runs of the real car does it take to fix?**

### Why this is a real gap

DrivAerML comprises 500 parametrically morphed variants of **one** vehicle — a DrivAer
notchback. Every geometry DoMINO has ever seen has the same wheels, the same mirrors, the
same topology. Nothing in the literature answers the question every OEM evaluating this
technology actually has: *will this work on my car, which is not a DrivAer notchback?*

### This is not the paper's OOD

**Say this explicitly in the writeup, or a reader who knows the paper will conclude the
work has been redone.**

| | Their OOD | **Our OOD** |
|---|---|---|
| **Axis** | Drag force outside training min–max | **Geometric topology** |
| **Body** | Same DrivAer notchback (morphs) | **Different body entirely** |
| **Unseen** | Output range | **Wheels, mirrors, C-pillar** |

The DoMINO paper *does* evaluate OOD samples — the test split reserves 10%, of which ~20%
are OOD by drag force. But every sample, train and test, ID and OOD, is a morph of the same
notchback.

Their result **sharpens** this study rather than pre-empting it. DoMINO demonstrably
handles OOD drag *values* well. If it fails on OOD *geometry*, the contrast isolates
precisely which axis of generalization breaks — a cleaner finding than either alone.

### Why upward, not downward

**Downward** (DrivAer → Ahmed/Windsor): train on the complex car, test on the simple block.
Will probably succeed. Establishes nothing — generalizing from complex to simple is not a
claim anyone needs. *Retained as a control.*

**Upward** (Ahmed + Windsor → DrivAer): train on cheap simple bodies, test on the expensive
realistic car. Cheap data, expensive target. **This is the experiment**, and the direction
in which failure is *informative*.

---

## 2. Registered hypotheses

Stated **before** execution. Falsifiable. **Nothing here is proved. None may be cited as
results.** If a result contradicts one, the hypothesis is **refuted** — it is not amended,
softened, or reinterpreted. A refuted hypothesis is a finding. A rewritten one is a lie.

### H1 — Upward transfer degrades

A DoMINO model trained on AhmedML + WindsorML will show materially worse Cd error on
DrivAerML than a model trained on DrivAerML itself.

**Mechanism** ([§4.4](#44-why-unseen-geometry-should-break-it)): the point-convolution
kernels are *learned*. No configuration of points resembling a wheelarch, a mirror stalk,
or a notchback C-pillar ever passes through them during Ahmed/Windsor training. The SDF
will report that geometry is present; it will not supply a learned representation of what
that geometry *does to the flow*.

**Status:** registered. Expected to hold. *That expectation is the point, not a hedge.*

### H2 — Error localizes to absent features

Under cold upward transfer, surface Cp error will concentrate on the **wheels,
wheelhouses, mirrors, and C-pillar/rear-window** — precisely the features absent from the
source bodies — while simple zones (roof, hood) remain comparatively accurate.

**Status:** registered. Adjudicate in Phase 5. **Do not revise after seeing the error map.**

### H3 — Error cancellation precedes accuracy

At small fine-tuning budgets, the model will achieve plausible Cd *before* it achieves an
accurate Cp field: area-weighted Cp L2 error will remain large while the **signed bias**
approaches zero — large local errors cancelling in the integral. Design-ranking correlation
will lag absolute Cd accuracy.

**Status:** registered. Adjudicate via `E_bias`. **If observed, report it — do not "fix" it
away. It is a finding.**

### H4 — Downward transfer succeeds *(control)*

A model trained on DrivAerML will transfer successfully to Ahmed and Windsor.

**Why it matters:** if only A3 were run and it failed, a skeptic says *your pipeline is
broken*. A2 succeeding on the same pipeline forecloses that objection. **It is the reason
the negative result will be believed.**

**Status:** registered as control. **A2 is a gate — run it before A3.**

### H5 — Source diversity buys transfer *(interpretive control)*

Ahmed + Windsor will show materially lower cold-transfer Cd error than Ahmed alone.

**What this is:** an interpretive control, **not a scaling law**. With two points it can
say whether adding one body family moves the number — nothing more. If it does *not* move,
the barrier is topological rather than one of source-set size: a stronger reading of H1.

**Status:** registered. Adjudicate from A3a vs A3 cold numbers. **Two points are not a
curve. Do not draw a line through it.**

> **Why a negative result is the strong outcome.** The study cannot fail into
> worthlessness. If upward transfer succeeds, cheap bluff-body data buys real-car
> predictions — a cost argument with money attached. If it fails, the limits of geometric
> generalization in neural operators are demonstrated, and the recovery curve states
> exactly what repair costs.

---

## 3. What we measure

Three numbers. The second two exist to prevent a specific self-deception.

### Accuracy — Cd MAE in drag counts

`1 count = 0.001 Cd`. The unit aerodynamicists actually speak in.

Chosen over wall shear deliberately. Wall shear is small, noisy, dominated by near-wall
behaviour the model never resolves, and secondary to pressure drag on a bluff body. It is
the metric most likely to look poor for reasons unrelated to the research question.
*Retained as supporting evidence — it is where separation-resolution failure becomes
visible — but not as the headline.*

### Usability — Spearman ρ

Rank correlation between predicted and true Cd across designs.

**A surrogate that cannot order two designs by drag is useless for optimization regardless
of its absolute error.** This metric, not Cd MAE, determines whether the model is
deployable in a design loop.

### Honesty — area-weighted Cp error and its signed bias

With per-face areas `a_i`, weights `w_i = a_i / Σa_j`, and error `e_i = Cp_pred - Cp_true`:

```
E_L2   = sqrt( Σ w_i · e_i² )     ← magnitude of local error
E_bias =       Σ w_i · e_i        ← THE CANCELLATION DETECTOR
```

> **`E_bias` is the whole point.** A surrogate can obtain approximately the right Cd by
> **cancelling errors** — over-predicting pressure forward, under-predicting aft, landing
> on a plausible integral. It is right for the wrong reasons and will fail the moment it is
> used to *choose between* designs.
>
> **Large `E_L2` with `E_bias ≈ 0` means exactly that.** It is invisible to Cd alone.
> **Log both, every run, without exception.**

---

## 4. What DoMINO actually learns

This section exists because [H1](#h1--upward-transfer-degrades) and
[H2](#h2--error-localizes-to-absent-features) are claims about *why the geometry encoder
fails to represent an unseen feature*. Those claims cannot be defended — in review, in an
interview, or to yourself — without the mechanism they rest on.

### 4.1 The operator

DoMINO approximates the solution operator mapping a **geometry** to the flow fields it
induces:

```
G : S  ↦  ( u, p, ν_t  in the volume ;  p, τ_w  on the surface )
```

where `S` is the vehicle surface, presented as a triangulated STL. **This study concerns
the surface branch only.**

> **Stripped of branding:** `G` is approximated by supervised regression against CFD ground
> truth, trained with MSE and Adam. There is no reasoning, no physical model, and no
> understanding of fluid dynamics inside DoMINO. It is a **learned interpolator over a
> geometry-to-field mapping**.
>
> This is not a criticism — it is the fact that makes the project's question well-posed. A
> system that *understood* aerodynamics would generalize to a wheel because it knows what a
> wheel does to flow. **An interpolator generalizes to a wheel only if something wheel-like
> was inside its training distribution.**

**Why "operator" and not "network":** the object learned is a map between *function spaces*.
Solutions can be queried at arbitrary points — the model is not tied to the mesh it trained
on. The paper demonstrates this by evaluating on a uniform 10M-point cloud sampled from the
STL instead of the simulation mesh, recovering the same drag regression (R² = 0.96 both
ways). This **mesh-independence** is why transfer to a new body is a coherent question at
all.

### 4.2 Three stages

**Stage 1 — Global geometry encoding.** The STL point cloud is projected onto a structured
grid using **learnable point-convolution kernels**:

```
y_i = Σ_j  f( x_i , x_j , d_ij )
```

`f` is a fully connected network; the sum runs over neighbours found by a **ball query**
(custom GPU kernels in NVIDIA Warp). Two parameters govern it: the neighbour count, and the
**radius of influence** `r`. Large `r` captures long-range interaction; small `r` captures
fine detail. **DoMINO is *multi-scale* precisely in this sense — it uses a range of radii
simultaneously.**

Geometry reaches the domain grid two ways: directly, via more kernels; and **iteratively**,
via CNN blocks propagating features from the surface grid. *(That is the "Iterative" in the
name.)* Finally the **signed distance field** and its gradients are appended.

**Stage 2 — Local encoding.** The solution at any point is dominated by information in its
*locality*. So rather than use the dense global encoding, DoMINO **extracts** a small
sub-region encoding per query point. *(That is the "Decomposable".)* **This is what lets it
scale to ~150M-element meshes without downsampling** — the failure mode that killed prior
ML surrogates.

**Stage 3 — Aggregation.** Around each point, a stencil of `p+1` neighbours is built (by
analogy with finite-volume stencils). Each stencil point carries coordinates, SDF, and
normals; these pass through a basis-function network, concatenate with the local encoding,
and the `p+1` predictions combine by **inverse-distance weighting**.

### 4.3 The loss

Per-variable MSE plus an area-weighted MSE term. Adam, reduce-on-plateau from 1e-3 to 1e-6,
500 epochs, AMP fp16.

> ⚠️ **The loss is where [Trap 1](#trap-1--variable-ordering-is-hardcoded) lives.** The
> implementations index into the target tensor *positionally*.

### 4.4 Why unseen geometry should break it

**This is the load-bearing argument of the whole study.**

1. The point-convolution kernels are **learned**. `f` is fit to the training geometries. Its
   parameters encode which local point-cloud configurations matter and how they map to flow
   features.

2. The local encoding is likewise learned — the paper states that local geometry
   representations *"extract the essential information required to predict solution fields
   in different regions."* **Which features are "essential" is determined by the training
   distribution.**

3. **Ahmed and Windsor contain no wheel, no wheelhouse, no mirror, and no notchback
   C-pillar.** No configuration of points resembling a wheelarch, or a mirror stalk shedding
   a vortex, ever passes through `f` during training.

4. Consequently the kernels have no basis on which to encode them, and the aggregation
   network has never been asked to map such a local encoding to a surface pressure. **The
   SDF will faithfully report that geometry *is* there — but a faithful SDF of an unseen
   feature is not a learned representation of what that feature does to the flow.**

This makes a **spatially localized** prediction: error should concentrate on the specific
absent features, and **not** on the roof or hood, which Ahmed and Windsor do resemble. That
is falsifiable, and Phase 5 is built to adjudicate it.

> **The optimistic counter-argument, stated fairly.** DoMINO is *local* by construction —
> predictions depend on a sub-region and a `p+1` stencil, not the global body. If the local
> flow physics around a mirror resembles the local physics around *some* bluff-body feature
> the model has seen — a sharp edge, a separation line, a slanted surface — locality could
> carry it further than the global-topology argument suggests. Ahmed's slant *does* produce
> geometry-induced separation; Windsor's squareback *does* produce a base wake.
>
> **The question is empirical. That is exactly why it is worth the compute.**

### 4.5 Published baseline

| Surface field | Rel. L2 (paper) | Docs example run |
|---|---|---|
| Pressure | 0.1505 | 0.101 |
| X-wall-shear | 0.2124 | 0.138 |
| Y-wall-shear | 0.3020 | 0.174 |
| Z-wall-shear | 0.3359 | 0.198 |
| **Drag R²** | **0.96** | **0.983** |

Two baselines exist and they differ — different splits, different configs. **Target the
docs example run** (it is the configuration shipped in the repo, and therefore the one you
are actually reproducing) and say so. If your number lands between the two, fine. **If it
lands outside both, something is wrong.** Do not average them, and do not quietly pick the
flattering one.

---

## 5. The seven traps

**Phase 0 found seven ways the three datasets are incompatible. Every one fails silently —
a smoothly converging loss curve and a worthless model. Not one raises an error.**

> This is the entire justification for the exit-criterion discipline. **The dominant risks
> in this project are silent, and the criteria are the only detectors.**

### Trap 1 — Variable ordering is hardcoded

`loss.py:423–427` (at `59aaf59`):

```python
pres_true = output_true[:, :, 0] * normals[:, :, 0]   # channel 0 IS pressure
wx_true   = output_true[:, :, 1]                       # channel 1 IS x-wall-shear
```

**Channel 0 must be pressure. Channel 1 must be x-wall-shear.** Nothing validates this —
the tensor carries no names, so nothing *can*. A permuted target produces a well-formed,
smoothly decreasing loss that regresses the wrong channels against each other.

**Fix:** enforce `[Cp, Cf_x, Cf_y, Cf_z]` at the datapipe boundary. **Assert it. Do not
trust it.**

*Also note:* only channel 1 enters the drag loss — τ_y and τ_z are absent. Channels 2 and 3
could be swapped without the *drag* loss noticing.

### Trap 2 — Fine-tuning silently trains nothing

**There is no `retraining.py`.** It was removed; fine-tuning happens via `resume_dir` +
`train.py`. But `train.py:646`:

```python
init_epoch = load_checkpoint(
    to_absolute_path(cfg.resume_dir),
    models=model, optimizer=optimizer,
    scheduler=scheduler, scaler=scaler, device=dist.device,
)
if init_epoch != 0:
    init_epoch += 1
epoch_number = init_epoch
```

**The A3 checkpoint was saved at epoch 500.** Setting `train.epochs=50` starts the loop at
501, which exceeds 50 — **so it exits immediately, having trained zero steps, and writes a
checkpoint.** No crash. No warning.

> **Every fine-tuned model at every N would be identical to the cold model. The recovery
> curve — the study's headline figure — would be a flat line, and the conclusion would be
> "fine-tuning does not help."**

Also: the optimizer *and scheduler* are restored, so "fine-tune at a small learning rate" is
not something you get for free — you inherit whatever the schedule had decayed to.

**Fix (Patch #1).** `load_checkpoint` skips arguments that are `None`
(`physicsnemo/utils/checkpoint.py:964` — *"Objects that are `None` are silently skipped"*).
Under a fine-tuning flag: **(a)** call it with `models=model` only — no optimizer, no
scheduler, no scaler; **(b) force `init_epoch = 0`**, discarding the returned value. Gate it
so the ordinary crash-resume path is unchanged.

### Trap 3 — `force_mom_1.csv` means opposite things

| | `force_mom_<i>.csv` | Suffixed file |
|---|---|---|
| Ahmed | **constant** ref | `_varref` = variable |
| Windsor | **constant** ref | `_varref` = variable |
| **DrivAer** | **variable** ref | `_constref` = **constant** |

**The unsuffixed file inverts meaning across exactly the boundary this study transfers
over.** Loading `force_mom_1.csv` uniformly — the obvious implementation — yields
constant-reference truth from the source datasets and variable-reference truth from the
target.

**The magnitude is not marginal: 47 drag counts on Ahmed** (0.2385 vs 0.2854); 7.4 on
DrivAer. **The headline metric is measured in drag counts. The effect being measured is
smaller than the error this would introduce.**

Confirmed three ways: the AhmedML paper (Table 3), the WindsorML paper (states the
convention explicitly), and DrivAerML's `geo_ref_<i>.csv`, which publishes `aRef = 2.223`
(this morph) alongside `aRefRef = 2.17` (constant) — and `0.3035 × (2.223/2.17) = 0.3109`,
reconciling the two files arithmetically.

**Decision: use the constant-reference file throughout.** Which means `force_mom_*.csv` for
Ahmed and Windsor but **`force_mom_constref_*.csv` for DrivAer.**

**Why constant:** the usability metric is a design *ranking*. Under a per-case area, a morph
that changes frontal area moves its own Cd for reasons unrelated to the flow — the ranking
would partly reflect geometry rather than aerodynamics.

### Trap 4 — Windsor's surface is a `.vtu` with point data

| | Container | Fields on |
|---|---|---|
| Ahmed | `.vtp` (PolyData) | **cells** |
| **Windsor** | **`.vtu` (UnstructuredGrid)** | **points** |
| DrivAer | `.vtp` (PolyData) | **cells** |

Windsor ships its *surface* in a *volume* container. **A loader globbing for `*.vtp` finds
nothing.** And area-weighted integration requires *cell* data — Windsor must be converted
(`point_data_to_cell_data()`) before any force is computed.

*Note:* PhysicsNeMo-Curator ships worked examples for `ahmedml` and `drivaerml` but **not**
Windsor. **Windsor is the custom dataset, and the bulk of this project's engineering delta
lives there.**

### Trap 5 — Windsor's wall shear is three scalars, z before y

Ahmed and DrivAer store one 3-vector. Windsor stores **three separate scalar arrays**, and
the file order is:

```
cfxavg , cfzavg , cfyavg          ← Z BEFORE Y
```

**Stacking them in array order produces a permuted vector** — and given Trap 1, a permuted
vector is a silently wrong model.

### Trap 6 — U∞ spans 39×, so wall shear spans ~1500×

| | Ahmed | Windsor | DrivAer |
|---|---|---|---|
| **U∞** | **1 m/s** | 30 m/s | 38.889 m/s |
| Shear range | ±0.036 | ±0.02 | **±40** |

**Ahmed runs at 1 m/s. Not 30.** The AhmedML paper sets Re = 7.68e5 through the *viscosity*
(ν = 3.75e-7), not the velocity. It looks like a typo. It is not.

Wall shear scales as U², so the *dimensional* shear fields differ by ~1500× between Ahmed
and DrivAer. **Training on both without non-dimensionalizing would let DrivAer dominate the
loss entirely** — Ahmed's contribution would be numerical noise. This is not a subtle
weighting issue; it is three orders of magnitude.

Windsor is the exception: it publishes Cf directly, already non-dimensional. Ahmed and
DrivAer must be divided by `q = ½ρU²`.

> ⚠️ **None of the three datasets publishes ρ** — they are incompressible simulations
> specified by kinematic viscosity. **Phase 2 must verify the assumed ρ** by recomputing Cp
> from `p` and `q` and checking against each dataset's own published Cp field. If they
> disagree, ρ is wrong.

### Trap 7 — Windsor is y-up and yawed −2.5°

WindsorML paper: *"for all the simulations in this work, **y is upwards**, which differs to
the original Windsor geometry where z is upwards."* Recon confirms it — Windsor's STL spans
y ∈ [0, 0.475] (height) and z ∈ [−0.194, +0.194] (width). Ahmed and DrivAer are z-up.

**Drag is safe.** x is streamwise in all three, and drag depends only on the x-component.

**Lift, side force, and the y/z shear channels are not.** Canonical `Cf_y` and `Cf_z` mean
*different physical directions* in Windsor. The model trains on all four channels, so a
permuted y/z in one third of the training data is exactly the silent corruption this project
exists to prevent. **Phase 3 must decide explicitly whether to swap Windsor's axes.**

**And Windsor is yawed −2.5°**, generating a side force by design. Ahmed and DrivAer are at
zero yaw. **Windsor's flow is asymmetric where the others are symmetric.** This is a real
physical difference between source and target and belongs in the limitations.

### ✅ Defused — the streamwise axis *is* x

DoMINO's drag loss multiplies pressure by `normals[:, :, 0]` — the x-component. Had any
dataset been oriented differently, the integral loss would have optimized a force component
that is not drag. **All three are x-streamwise. Verified from the STL bounding boxes, not
assumed.**

### Trap 8 — Wall shear has the opposite sign convention

OpenFOAM's `wallShearStress` is the stress the **wall exerts on the fluid** — the *negative*
of what a drag integral needs.

**Symptom: viscous drag came out negative.** Skin friction opposes motion. **A negative
viscous drag is physically impossible**, and it should have been the first thing anyone
noticed.

**Fix:** negate `wallShearStressMean` / `wallShearStressMeanTrim`. **Windsor is exempt** — it
publishes Cf directly, already in the right convention.

### Trap 9 — ρ = 1.0, not 1.225

**No dataset publishes ρ.** These are incompressible solvers in **kinematic units** —
pressure is stored as `p/ρ` — so ρ never appears in their inputs. **Assuming sea-level air
(1.225) is the natural mistake, and it is wrong.**

**Found by:** each dataset publishes **both `p` and `Cp`**, so their ratio *is* `q`.

| | `p / Cp` | σ | U∞ | → ρ |
|---|---|---|---|---|
| Ahmed | **0.500000** | 1.8e-9 *(1.1M facets)* | 1.0 | **1.0** |
| DrivAer | **756.177161** | 2.7e-5 *(8.8M facets)* | 38.889 | **1.0000000000036** |

**That is not an estimate. It is an identity.**

With ρ = 1.225, Ahmed's viscous term was 22% low and its Cd 5% short — comfortably inside the
range where a result still looks believable.

> **The plan predicted this check** — *"Phase 2 must verify the assumed ρ by recomputing Cp
> from p and q and checking against each dataset's own published Cp field."* It was right,
> and it paid.

### Trap 10 — Three datasets need three different normal strategies

**Undocumented anywhere. A single code path silently fails on two of three.**

| | Ships normals? | `clean()` | **Strategy** |
|---|---|---|---|
| **Ahmed** | No | **58,308 → 0 open edges** ✅ | `clean()` → `auto_orient_normals` |
| **Windsor** | **Yes** ✅ | **destroys the cell arrays** ❌ | **use theirs. Never clean.** |
| **DrivAer** | No | **7,383 → 7,383. No effect** ❌ | `consistent_normals` → **global flip** |

**Ahmed's** mesh is stitched from per-patch exports with duplicated vertices at the seams —
geometrically closed, topologically full of cracks. **To VTK this is an *open* surface, and
for an open surface "outward" is undefined** — there is no inside to be outside of. So
`auto_orient_normals` returns an *inconsistent mix*. `clean(1e-6)` merges the coincident
vertices; VTK then orients correctly on its own.

**DrivAer** ships no normals *and* `clean()` doesn't help — its cracks are not duplicate
vertices. But `consistent_normals=True` makes the orientation locally coherent, leaving a
single **global** sign, settled against the published Cd. Un-flipped: **−0.2533**. Flipped:
**+0.31093** against a published **0.31092**.

### Trap 11 — `clean()` destroys Windsor's cell arrays

At tolerances loose enough to affect Windsor's topology, `clean()` **merges cells** while
leaving the cell arrays at their old length — **4.99M cells against a 9.92M-entry `cfxavg`.**

**pyvista only *warns*.** The resulting Cd is computed from **fields that no longer
correspond to the facets they belong to** — and comes out as a perfectly plausible float.

**Caught only because the open-edge guard refused to proceed.** Chasing the edge count down
by loosening the tolerance — the obvious thing to do — would have produced a number, and it
would have been meaningless.

### Trap 12 — Inward normals

Ahmed's first Cd: **−0.218** against a published **+0.238**. Same magnitude, opposite sign.

**The most visible of the five, and the least dangerous** — a negative Cd announces itself.
**The other four do not.**

---

> ### Twelve traps, and not one raises an error
>
> **Phase 0 found seven. Phase 2 found five more.** Every single failure in this project so
> far has produced a **confident, plausible number**. Not one has produced a stack trace.
>
> **The exit criteria are not bureaucracy. They are the only detectors that exist.**

### The full compatibility table

| | **Ahmed** | **Windsor** | **DrivAer** |
|---|---|---|---|
| Surface file | `boundary_*.vtp` | **`boundary_*.vtu`** | `boundary_*.vtp` |
| Fields on | **cell** | **point** | **cell** |
| Cp field | `static(p)_coeffMean` | `cpavg` | `CpMeanTrim` |
| Dimensional p | `pMean` | **none published** | `pMeanTrim` |
| Shear storage | 3-vector, **Pa** | **3 scalars**, already Cf | 3-vector, **Pa** |
| Shear name(s) | `wallShearStressMean` | `cfxavg`,`cfzavg`,`cfyavg` | `wallShearStressMeanTrim` |
| **U∞** | **1 m/s** | 30 m/s | 38.889 m/s |
| ν | 3.75e-7 | — | 1.507e-5 |
| Constant A_ref | 0.112 m² *(paper only)* | 0.112 m² | 2.17 m² |
| **Const-ref file** | `force_mom_*` | `force_mom_*` | **`force_mom_constref_*`** |
| Var-ref file | `force_mom_varref_*` | `force_mom_varref_*` | **`force_mom_*`** |
| Cd header | `cd` | `cd` | `Cd` |
| Streamwise | **x** | **x** | **x** |
| **Vertical** | **z** | **y** | **z** |
| **Yaw** | 0° | **−2.5°** | 0° |

*Source: `src/recon.py` + the three dataset papers. Encoded with provenance in
`src/datasets.py`.*

### The canonical output space

```
y = [ Cp , Cf_x , Cf_y , Cf_z ]      ← fixed order, enforced at the datapipe
```

**This is forced, not chosen.** The *ordering* is forced by Trap 1. The *coefficient form*
is forced by Trap 6 — **Windsor publishes no dimensional pressure**, so this is the only
representation all three datasets can supply.

---

## 6. Phases

Every phase has an **exit criterion** that must be cleared before proceeding. They exist
because the dominant failures here are silent.

---

### Phase 0 — Reconnaissance ✅

**Goal:** eliminate every unknown capable of silently corrupting a later phase. No GPU work.

**Done:**
- Repo structure, two-repository split, `.gitignore` before anything else
- Upstream cloned and **pinned**; hashes in `environment.md`
- Tree read; plan corrected against reality — three files the plan depended on **do not
  exist**
- `src/recon.py` run on one case from each dataset
- Compatibility table populated from **inspected data**, not documentation
- `src/datasets.py` written — single source of truth, everything with provenance
- **Seven silent traps found** ([§5](#5-the-seven-traps))

#### Exit Criterion 0 — CLEARED

*Cleared 2026-07-11.* Repository exists; `external/` verified against the actual clone;
upstream pinned; compatibility table populated from inspected data; canonical output space
committed (`[Cp, Cf_x, Cf_y, Cf_z]` — forced, not preferred); frontal-area convention
declared (**constant reference throughout**, from the correct file per dataset).

---

### Phase 1 — Baseline reproduction

**Goal:** demonstrate the pipeline reproduces the published DoMINO result on DrivAerML.
**This is not a contribution — it is the calibration of the instrument.** If the paper's
numbers cannot be recovered, no later number means anything.

**Steps:**
1. Preprocess DrivAerML via PhysicsNeMo-Curator (the supported path)
2. `compute_statistics.py` → populates `surface_factors`. **Retain these — they become a
   leakage hazard in Phase 3.**
3. Train, surface only. `torchrun --nproc_per_node=8 train.py`
4. Compare against the [docs example run](#45-published-baseline)

**Knobs:** `model.surface_points_sample` is the OOM knob. `model.geom_points_sample` must be
< the number of points on the STL.

**Cost:** NVIDIA reports ~4 hours on 8×H100 (down from ~5 days). The experimental matrix is
therefore affordable.

#### Exit Criterion 1

**Drag R² > 0.80** on the held-out DrivAer morphs.

> ⚠️ **This is relaxed from the full study's R² > 0.95, and the relaxation must be
> stated wherever the number is reported.** NVIDIA's 0.983 came from ~400 training
> morphs. A1 here trains on ~20. **R² is a function of training-set size as much as of
> pipeline correctness**, and it would be dishonest to present a low R² as a pipeline
> failure — or a high one as vindication.
>
> **What EC1 actually tests at this scale:** that the data loads, the loss decreases,
> the checkpoint saves, the model produces a surface field, and the force integrator
> turns that field into a plausible Cd. **It is a smoke test for the instrument, not a
> demonstration of accuracy.**
>
> **If R² < 0.80 — stop and debug.** Something is broken, and it is more likely the
> variable ordering ([Trap 1](#trap-1--variable-ordering-is-hardcoded)) or the force
> integrator than the sample count.
>
> **If R² > 0.95 on 20 training cases — be suspicious, not pleased.** Check for leakage
> before celebrating.

**Do not proceed with a broken instrument.**

---

### Phase 2 — Force integration and metrics harness

**Goal:** build the measurement apparatus **once, correctly**. Pure software; no GPU.

> **This precedes the experiment, not follows it.** The temptation is to run the interesting
> transfer arm first and work out measurement afterwards. If the force integrator carries a
> sign or area error, a bad transfer number **cannot be attributed to physics rather than
> arithmetic** — and the distinction cannot be recovered after the fact.

**Drag** is the streamwise component of integrated surface traction:

```
F_x = ∮ (−p·n̂)·x̂ dS   +   ∮ τ_w·x̂ dS
      └─ pressure drag ─┘   └─ viscous ─┘
       (dominant on a bluff body)
```

**Sign conventions will bite.** Normal orientation (inward vs outward), the sign of the
pressure term, and which axis is streamwise vary between datasets and between VTK writers.
**Validate against ground truth before trusting a single prediction.**

**Validation protocol.** Run the integrator on **ground-truth** surface fields, ≥ 20 cases
per dataset, compare against the published constant-reference Cd:

| Relative Cd error | Action |
|---|---|
| **< 2%** | Pass. Proceed. |
| 2–5% | Investigate — likely cell-area or normal-orientation. |
| **> 5%** | **Sign error or wrong reference area. Do not proceed.** |

**Cross-check against PhysicsNeMo-CFD**, which contains the force utilities used to produce
the published drag-R² plots. **Where the two disagree, PhysicsNeMo-CFD is correct.**

**Also verify ρ here** ([Trap 6](#trap-6--u-spans-39-so-wall-shear-spans-1500)) — recompute
Cp from `p` and `q` and check against each dataset's own Cp field.

#### Exit Criterion 2 — PARTIALLY CLEARED

**Method verified. Robustness pending.**

| | Our Cd | Published | Error | |
|---|---|---|---|---|
| **Ahmed** | 0.238554 | ... 
| **Windsor** | 0.320846 | ...
| **DrivAer** | 0.310927 | ...

Gate is 2%. Worst case is a quarter of that. `src/forces.py` is committed with 19 tests
(`tests/test_forces.py`), plus 18 for the metrics harness. **37 passing.**

> ⚠️ **WHAT IS NOT YET CLEARED.** All of the above runs on **one case per dataset**...
>
> **The specific risk:** DrivAer's normal strategy assumes...
>
> `test_integrator_robustness_across_morphs` is written and **`@pytest.mark.skip`-ed**...

---

### Phase 3 — Unified multi-body datapipe

**Goal:** make Ahmed, Windsor, and DrivAer indistinguishable to DoMINO. **This is the
hardest engineering in the project and where silent bugs live.**

**The extension point is NOT `openfoam_datapipe.py`** — that file does not exist. The
datapipe is `physicsnemo/datapipes/cae/domino_datapipe.py` (library code), and preprocessing
has moved to PhysicsNeMo-Curator, which ships `ahmedml` and `drivaerml` examples. **Windsor
is the custom one.**

**Must handle, per [§5](#5-the-seven-traps):**
- Windsor `.vtu` + point data → cell data
- Windsor's three shear scalars, **in canonical x/y/z order, not file order**
- Windsor's **y-up axis** — decide explicitly on the swap
- Non-dimensionalization of Ahmed and DrivAer shear by `q`
- **Assert** `[Cp, Cf_x, Cf_y, Cf_z]` at the boundary

**Bounding boxes — a research decision, not a config detail:**

| Option | Mechanism | Consequence |
|---|---|---|
| **A (chosen)** | Per-body boxes | All bodies at unit scale. **Makes shape transfer the object of study**; discards absolute size. |
| B | One global box | Preserves relative scale, but Ahmed occupies a small corner of DrivAer's box — the model may never learn to use the full domain. |

**A is selected** because the question is about *shape* generalization, not scale. When asked
*"did you normalize scale away?"*, the answer must be a confident yes, with a reason.

> ⚠️ **The scaling-factor leak.** `compute_statistics.py` produces `surface_factors` over
> "the" dataset. For a mixed Ahmed+Windsor training set, compute them over the **combined
> source set only**, and apply those **same, unchanged** factors to DrivAer at test time.
>
> **Recomputing statistics on DrivAer leaks target information into a supposedly cold
> transfer.** The resulting number would be optimistic, would look entirely reasonable, and
> **would invalidate the central claim of the study.**
>
> **This applies per source set.** A3a trains on Ahmed alone and has its *own* factors.
> Reusing A3's Ahmed+Windsor factors for A3a leaks Windsor into the arm whose entire purpose
> is to *remove* Windsor.

#### Exit Criterion 3

**A single assertion passes on a batch drawn from each of the three bodies:** field names,
variable ordering, units, and normalization are identical. **Written as a test, not as a
belief.**

---

### Phase 4 — The experiment

#### The arms — run in this order

| Arm | Train | Test | What it establishes |
|---|---|---|---|
| **A1** | DrivAer (20) | DrivAer (20 holdout) | Instrument check. **Smoke test, not accuracy** — see [EC1](#exit-criterion-1). |
| **A2** | DrivAer (40) | Ahmed + Windsor | **Downward control (H4). GATE — run before A3.** |
| **A3** | **Ahmed (80) + Windsor (40)** | **DrivAer (20 holdout)** | **THE EXPERIMENT (H1).** |

> **A3a is cut at this scale.** H5 (source diversity buys transfer) **remains registered
> and unadjudicated.** It is not refuted, not confirmed, and not quietly dropped — it is
> a question this study cannot answer, and the writeup says so.

> **A2 is a gate, not filler.** If the downward control fails, the pipeline is broken and
> A3 is uninterpretable. **There is no point spending the run.**

#### The recovery curve — three points, not a curve

From the **A3** checkpoint, fine-tune on **N ∈ {0, 5, 20}** DrivAer cases from the
20-case fine-tuning pool, evaluating on the **20-case holdout that is never fine-tuned
on**.

⚠️ **See [Trap 2](#trap-2--fine-tuning-silently-trains-nothing) before running this.**
The naive `resume_dir` approach trains **zero steps** and writes a checkpoint anyway.

> **Three points is a line segment.** The full study runs six budgets × three seeds
> precisely because **at small N the identity of the cases dominates the outcome**. With
> one seed, that variance is not absent — **it is invisible.**
>
> **Do not annotate a knee.** Three points cannot resolve one. **Do not fit a curve.**
> Plot the points, connect them if you must, and say in the caption that this is
> directional evidence and nothing more.

**5 runs total:** A1, A2, A3, FT N=5, FT N=20. *(N=0 is the A3 cold checkpoint — no
extra run.)*

> **A2 is a gate, not filler.** If the downward control fails, the pipeline is broken and A3
> is uninterpretable. **There is no point spending the run.**

> **What A3a is, and is not.** With A3 alone, a failure is ambiguous: did transfer break
> because the *topology* is unseen, or because two source bodies is not a distribution? A3a
> resolves that ambiguity.
>
> **The confound is accepted, not closed.** Ahmed is 500 cases; Ahmed+Windsor is 855. If A3
> beats A3a, **diversity and sample count are not separated.** Closing it needs a third arm
> (Ahmed+Windsor subsampled to 500) — deliberately not run, because that is the first step
> of the *diversity-scaling study*, which is a different project. **State the confound.**

#### The recovery curve

From the **A3** checkpoint, fine-tune on N ∈ {0, 5, 10, 25, 50, 100} DrivAer cases,
evaluating throughout on a **fixed holdout that is never fine-tuned on** (100 cases, reserved
from the outset).

⚠️ **See [Trap 2](#trap-2--fine-tuning-silently-trains-nothing) before running this.** The
naive `resume_dir` approach trains **zero steps** and produces a flat line.

> **Three seeds per N, or the curve is an anecdote.** Run each N with three independent draws
> of *which* N cases are used. **At N=5, the identity of the five cases will dominate the
> outcome.** Plot mean and spread.

**A3a contributes a single cold (N=0) point**, for comparison against A3's cold number.

#### Logging schema — one JSON per run, written on completion

```json
{
  "arm": "A3",              // A1 | A2 | A3a | A3
  "n_finetune": 25,
  "seed": 1,
  "cd_mae_counts":      0.0,   // HEADLINE
  "cd_pressure_counts": 0.0,   // decomposition: where does the error live?
  "cd_viscous_counts":  0.0,
  "spearman_rho":       0.0,   // usability
  "cp_l2_area_wtd":     0.0,   // honesty
  "cp_signed_bias":     0.0,   // ← CANCELLATION DETECTOR
  "drag_r2":            0.0
}
```

**22 runs total** (4 arms + 6 budgets × 3 seeds).

#### Exit Criterion 4

Arms run **in order** — A1, **A2 (gate)**, A3a, A3. Recovery curve populated at 6 values of
N × 3 seeds. Every run logged. **H5 adjudicated from the A3a-vs-A3 cold numbers.**

---

### Phase 5 — Error localization

**Goal:** establish **why** it fails, not merely **that** it fails. *This is the section that
distinguishes an engineer from someone who ran a training script.*

#### Per-zone error — adjudicating H2

| Zone | In source? | Expectation under H2 |
|---|---|---|
| Front end / stagnation | Yes (both) | Accurate |
| Hood, windscreen, roof | Yes (both) | Accurate |
| **C-pillar / rear window** | **No** | **Large error** |
| Rear end / base wake | Partly (Windsor) | Moderate |
| **Wheels, wheelhouses** | **No** | **Large error** |
| **Mirrors** | **No** | **Large error** |
| Underbody | No | Large error |

> **The prediction is registered in advance.** If it holds, the study has a *mechanistic*
> explanation rather than a number. If it does not, **that is the more interesting outcome**,
> and it was found honestly rather than rationalized afterwards. **Do not revise the
> hypothesis after seeing the error map.**

**Normalize before concluding.** Wheels and wheelarches are also where the *true* Cp field
has the highest variance. A model with uniformly poor spatial resolution would show peak
error there too, for reasons unrelated to the training distribution. **Compare A3's per-zone
profile against A1's** — A1 has seen wheels. **If A1's error map has the same shape, just
smaller, H2 is not supported.**

#### Wake evidence — showing the mechanism, not asserting it

This study is surface-only **for every metric**. That is correct for the *metric*. **It is
not sufficient for the *mechanism*.**

> H2 claims the model fails because it has no learned representation of a C-pillar vortex.
> With surface data alone you can show **that** the C-pillar pressure is wrong. You cannot
> show **why** — the cause lives in the flow *off* the surface.
>
> **The reviewer's question writes itself:** *"You claim it fails to represent the C-pillar
> vortex. Show me the vortex."*

**The bounded fix — five DrivAer cases only**, selected **after** A3, once the error ranking
is known:

| Cases | Selection | Purpose |
|---|---|---|
| 2 | Worst cold-transfer Cd error | Where the mechanism should be most visible |
| 2 | Best cold-transfer Cd error | Control: is the wake right when the drag is right? |
| 1 | Median | Reference |

Two planes suffice: a centreline x–z cut and an x–y cut through the wheel.

> **Scope guard.** Five cases. Qualitative. No volume metrics in the headline, no volume
> model, no volume loss term. **The purpose is one figure panel that makes H2 *shown* rather
> than *asserted*.** If this starts growing into a volume-field study — **stop.**
>
> Cost: 5 × 49 GB ≈ 245 GB. Bounded and affordable. **Five hundred cases would be 24.5 TB —
> that is the failure this guard exists to prevent.**

#### The three figures

**There are only three. Anything beyond them is dilution.**

**F1 — The Recovery Curve** *(the money figure — it carries the deliverable sentence)*
`x`: fine-tuning budget N. `y`: Cd MAE in drag counts.
Lines: A1 baseline (horizontal reference), A3 cold at N=0, A3 fine-tuned. Shaded band for
seed spread. **Annotate the knee.**
Mark A3a's cold number as a **bare point** at N=0 — **never joined to A3 with a line.**

**F2 — The Honesty Check**
Surface Cp error painted on the DrivAer body; three views (side, rear, underside); diverging
colormap centred at zero **so sign is visible**. Two panels: cold vs fine-tuned at the knee.
**The reader should *see* error concentrate on wheels, mirrors, and C-pillar — then recede.**
**Plus one wake panel**: predicted vs true velocity on a centreline cut. *The surface map
shows **where** the pressure is wrong; the wake cut shows **why**.*

**F3 — The Ranking Test**
Predicted vs true Cd across held-out designs; diagonal reference; Spearman ρ inset. Two
panels, cold vs fine-tuned. **This is where a reader decides whether they could actually
optimize with this model.**

#### Exit Criterion 5

Three figures exist, F2 including the wake panel. Per-zone error table populated and
**normalized against A1**. **H2 explicitly adjudicated — held or refuted.** The failure
location **and its physical cause** stated in one sentence, with the wake cut as evidence for
the second half.

---

### Phase 6 — Deliverable

> **The reader will spend approximately eight minutes on this and will not read a methods
> section. The README *is* the deliverable; the code is the evidence.**

**Structure:**

1. **The question.** One sentence.
2. **The answer.** One sentence, **with the number in it**.
3. **Figure F1.**
4. **Why this should be believed** — baseline reproduction (EC1) and force-integrator
   validation (EC2). Two rows each. *Credibility is a down-payment; spend it early.*
5. **Figures F2, F3.**
6. **What it means for practice** — the N-runs recommendation.
7. **Limitations**, stated before anyone else finds them.
8. **Reproduction instructions.**

#### The target sentence

> *A DoMINO surrogate trained only on simple bluff bodies predicts DrivAer drag to within
> **\_\_\_** counts and preserves design ranking at Spearman ρ = **\_\_\_** — but only after
> fine-tuning on **\_\_\_** high-fidelity runs. Below that, it obtains the right drag for the
> wrong reasons: errors concentrate on the wheels and C-pillar and cancel in the integral.*

**The blanks are unknown. That is the point, and it is what makes the study worth running.**

#### Limitations — state them yourself

- Surface-only for all metrics; volume data used qualitatively, five cases, mechanism only.
- **A single architecture.** No claim that other surrogates behave the same way.
- **Two source bodies is a thin "distribution."** A3a tests whether the second body buys
  anything, but two points do not establish how much diversity transfer requires — **and the
  sample-count confound (500 vs 855) is not separated.**
- **Windsor is yawed −2.5° and y-up**; Ahmed and DrivAer are neither. The source bodies are
  not a homogeneous family.
- Geometry variation only, at fixed inlet conditions — a limitation the AhmedML authors flag
  about their own dataset.
- **No uncertainty quantification.** The model does not know when it is extrapolating.

#### Declared future work

> **The encoder-vs-aggregation ablation.** Freeze the geometry encoder, fine-tune only the
> aggregation network — then the reverse. This separates two very different failures:
>
> - *If freezing the encoder recovers performance:* the encoder **did** encode the wheel.
>   Geometric generalization is fine; the **flow mapping** is what does not transfer.
> - *If it does not, and the reverse does:* the encoder itself is impoverished. The learned
>   kernels genuinely failed to represent an unseen feature. **This is the strong form of H1
>   — a statement about neural operators in general, not about drag.**
>
> **Deliberately excluded — not because it is uninteresting, but because it is *too*
> interesting.** Its headline (*"where does geometric generalization break in a neural
> operator?"*) is a different and arguably better paper than this one (*"can it predict drag
> on an unseen car, and what does recovery cost?"*). **Pursuing both gives a reader two
> half-arguments instead of one whole one.**
>
> **The same applies to diversity scaling.** A3a gives two points on a curve that needs many
> more. Both are named here — rather than omitted and hoped-about — because **that is the
> difference between scope discipline and an oversight.**

**Also deferred:** fidelity transfer (DrivAerNet RANS → DrivAerML HRLES; also CC-BY-NC, which
is a second reason), UQ, volume-field prediction, physics losses.

---

## 7. Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| **Variable ordering** → loss trains against wrong targets | 0, 3 | [Trap 1](#trap-1--variable-ordering-is-hardcoded). Assert in datapipe. **Loss curve looks healthy.** |
| **Fine-tuning trains nothing** → flat recovery curve | 4 | [Trap 2](#trap-2--fine-tuning-silently-trains-nothing). Patch #1. **Writes a checkpoint; no error.** |
| **Wrong force file** → 47-count systematic error | 2, 3 | [Trap 3](#trap-3--force_mom_1csv-means-opposite-things). Use `src/datasets.py`. |
| Sign / normal-orientation error in integration | 2 | Validate against GT to < 2% **before** trusting any prediction. |
| **Scaling-factor leak** → cold transfer looks too good | 3 | Source-set factors only. Separate files per source set. |
| **Holdout contamination** | 4 | Separate directory, never referenced by any `ft_*` config. |
| A3 fails so completely the cold number is uninterpretable | 4 | **Anticipated (H1).** The recovery curve is the deliverable — build it from the start, not as a rescue. |
| A3a ≈ A3 read as "diversity doesn't help" when it's a sample-count artefact | 4 | The confound is **stated, not silently resolved.** |
| Recovery curve dominated by seed at small N | 4 | 3 seeds per N; report spread, **never a single line**. |
| Storage blowup (49 GB/case volumes) | 0, 5 | Surface-only. STL + surface = 800 MB/case. Volumes **only** for the 5 wake cases. |
| Scope creep | All | Explicitly deferred list above. |
| OOM during training | 1, 4 | Lower `model.surface_points_sample` — **for all arms, not just one.** |

---

## 8. Quick reference

**Question:** Ahmed + Windsor → DrivAer. Can it predict drag on a body it has never seen, and
what does repair cost?

**Why it should break:** the point-convolution kernels are **learned**. No wheel, mirror, or
C-pillar ever passes through them during Ahmed/Windsor training.

**Not the paper's OOD:** theirs = drag outside training range, *same body*. Ours = *different
body*. **Say this explicitly.**

**Canonical target:** `[Cp, Cf_x, Cf_y, Cf_z]` — fixed order (Trap 1), coefficient form
(Trap 6). **Both forced, neither chosen.**

**Metrics:** Cd MAE in drag counts *(accuracy)* · Spearman ρ *(usability)* · `E_L2` and
`E_bias` *(honesty — the cancellation detector)*.

**Arms, in order:** A1 (instrument) → **A2 (gate)** → A3a (interpretive control) → **A3 (the
experiment)**.

**Recovery:** fine-tune A3 on N ∈ {0,5,10,25,50,100}, 3 seeds each, fixed never-fine-tuned
holdout. **22 runs.**

**Figures:** F1 recovery curve *(annotate the knee)* · F2 Cp error map + wake panel · F3
ranking scatter.

**Exit criteria:** EC0 compatibility table · EC1 Drag R² > 0.95 · EC2 integrator within 2% ·
EC3 cross-body assertion · EC4 all arms + curve · EC5 three figures + H2 adjudicated.

**Hypotheses:** H1 registered *(expected to hold)* · H2 adjudicate in Phase 5 · H3 adjudicate
via `E_bias` · H4 control · H5 interpretive control. **None are proved. None may be cited as
results.**

---

## Appendix A — Project structure

### The two-repository principle

**Their code and your code do not mix.** PhysicsNeMo is cloned as a pinned dependency — never
forked, never edited in place.

Three reasons, ascending:

1. **The upstream moves.** Training went from ~5 days to ~4 hours across releases; physics-loss
   support "might introduce breaking changes." **If your changes are smeared across a fork,
   rebasing is archaeology.**
2. **Your contribution becomes invisible.** Edit in place and your entire engineering delta —
   which is genuinely small, *and that is a virtue* — disappears into a diff against a 100k-line
   repo. **A reviewer, an interviewer, or you in six months cannot see what you did.**
3. **Reproducibility.** *"Clone at commit `abc123`, apply these patches, run these configs"* is
   a reproducible instruction. *"Here is my fork"* is not.

### Your repository

```
domino-xgeom/
│
├── README.md                    # the eight-minute deliverable (Phase 6)
├── environment.md               # commit hashes, CUDA, GPU, driver
├── requirements.txt             # local analysis env (pinned)
├── .gitignore                   # external/, data, checkpoints
│
├── docs/
│   └── plan.md                  # THIS FILE. The project's memory.
│
├── external/                    # ── THEIRS. Pinned. Gitignored. NEVER EDITED.
│   ├── physicsnemo/             #    model, train.py, test.py
│   ├── physicsnemo-curator/     #    preprocessing → zarr/npy
│   └── physicsnemo-cfd/         #    force utils — Phase-2 cross-check ORACLE
│
├── patches/                     # ── YOUR ONLY EDITS TO THEIR CODE
│   ├── train_finetune.py        #    Patch #1: weights-only load, init_epoch=0
│   ├── domino_datapipe.py       #    + Windsor path class, .vtu handling
│   ├── apply.sh                 #    idempotent — you WILL re-clone
│   └── UPSTREAM.md              #    what changed, why, against which commit
│
├── conf/                        # ── YOUR Hydra configs. One per arm. Diffable.
│   ├── base.yaml
│   ├── a1_drivaer_baseline.yaml
│   ├── a2_downward.yaml         #    GATE — run before A3
│   ├── a3a_ahmed_only.yaml      #    own scaling factors — do NOT reuse A3's
│   ├── a3_upward.yaml           #    ** THE EXPERIMENT **
│   └── ft_N{0,5,10,25,50,100}.yaml
│
├── src/                         # ── YOUR analysis. Pure functions. No GPU.
│   ├── datasets.py              #    ** SINGLE SOURCE OF TRUTH for dataset diffs **
│   ├── recon.py                 #    Phase 0
│   ├── forces.py                #    Phase 2 — SINGLE SOURCE OF TRUTH for Cd
│   ├── metrics.py               #    Phase 2 — drag counts, Spearman, signed bias
│   ├── zones.py                 #    Phase 5 — surface segmentation
│   ├── collect.py               #    Phase 4 — results/*.json → dataframe
│   └── figures.py               #    Phase 5 — F1, F2, F3
│
├── tests/                       # ── Exit criteria, executable.
│   ├── test_forces.py           #    GT integration within 2%  → EC2
│   └── test_datapipe.py         #    cross-body assertion      → EC3
│
├── results/                     # ── APPEND-ONLY. One JSON per run. 22 files.
│   └── {arm}_{N}_{seed}.json
│
└── figures/
    ├── F1_recovery_curve.pdf
    ├── F2_cp_error_map.pdf
    └── F3_ranking_scatter.pdf
```

### Storage — outside the repository

```
/scratch/domino-xgeom/
│
├── raw/                         # ── STL + SURFACE ONLY. NO VOLUMES.
│   ├── ahmed/    <case>/{ahmed_N.stl,   boundary_N.vtp, *.csv}
│   ├── windsor/  <case>/{windsor_N.stl, boundary_N.vtu, *.csv}   ← .vtu!
│   └── drivaer/  <case>/{drivaer_N.stl, boundary_N.vtp, *.csv}
│
├── wake/                        # ── ** EXCEPTION: 5 CASES ONLY. Phase 5. **
│   └── drivaer/  <case>/*.vtu   #    2 worst + 2 best + 1 median
│                                #    fetched AFTER A3, once errors are known
│                                #    5 × 49 GB ≈ 245 GB
│
├── processed/                   # ── curator output
│   ├── source_train/            #    Ahmed + Windsor  → A3
│   ├── ahmed_only_train/        #    Ahmed alone      → A3a
│   ├── drivaer_train/           #                     → A1
│   ├── drivaer_holdout/         #    ** NEVER FINE-TUNED ON. NEVER. **
│   └── ft_pools/N{...}_seed{...}/
│
├── scaling/                     # ── global and singular, ON PURPOSE
│   ├── source_factors.json      #    Ahmed + Windsor ONLY
│   ├── ahmed_only_factors.json  #    Ahmed ONLY — do NOT reuse the above
│   └── drivaer_factors.json     #    A1 baseline only
│
└── checkpoints/
    ├── a1/  a2/  a3a/  a3/
    └── ft_N{...}_seed{...}/
```

### The structural decisions that are not cosmetic

**`patches/` keeps your contribution visible.** Your entire modification to DoMINO is small:
the fine-tuning fix, and a Windsor path class. **That is an honest delta and it should be
legible as such.** `apply.sh` must be idempotent — you *will* re-clone. `UPSTREAM.md` must
record the commit the patches were written against, **because they will silently rot when the
upstream file changes.**

**Storage: surface-only, with one bounded exception.**

| | Per case | 500 cases |
|---|---|---|
| Surface (`.vtp`/`.vtu`) + STL | **~800 MB** | ~400 GB |
| Volume (`.vtu`) | **~49 GB** | **24.5 TB** |

**A factor of 61.** Downloading volumes *"in case we want wake analysis later"* is how this
project dies in Phase 0. The Phase-5 exception is **planned, not opportunistic**: five cases,
chosen *because* you now know which ones matter. **A speculative bulk download is chosen
because you do not.**

**`scaling/` is global, not per-arm — deliberately.** This layout exists to make the leak
**awkward to commit**. If scaling factors lived inside each arm's output directory — which is
the natural thing, and what `compute_statistics.py` encourages — then regenerating them for
DrivAer would feel like *housekeeping* rather than like **the methodological error it is**.
Making the file global and singular means **overwriting it is a conscious act**.

**`drivaer_holdout/` is a separate directory for the same reason.** Keeping it out of every
`ft_*` config makes contamination **something you would have to do on purpose**.

**`results/` is append-only JSON, not a notebook.** 22 runs. Accumulate them in a notebook's
memory and a kernel restart destroys them — along with any hope of regenerating the figures.
One JSON per run, written on completion; `figures.py` reads the directory. **A figure you
cannot regenerate from disk is an anecdote.**

### Naming caveat — Modulus vs PhysicsNeMo

The DoMINO paper says **NVIDIA Modulus** and links to `github.com/NVIDIA/modulus`. It was
renamed **PhysicsNeMo**. **Same codebase.** Tutorials and paths containing `modulus` are the
older name, not a different project.

---

## Appendix B — Session restart card

```
════════════════════════════════════════════════════════════════════
STATUS: PHASE 0 + PHASE 2 COMPLETE. EC0, EC2 cleared.
 NEXT:   Phase 3 -- unified datapipe. It is a REWRITE of
         openfoam_datapipe.py, not a two-class patch. See below.
 DATA:   ON ARC. ~/data/{ahmedml,windsorml,drivaerml}
         80 / 40 / 40 cases, 52 GB. Verified.
 LAST:   EC2 cleared -- 60 cases integrated, all within 2%.
════════════════════════════════════════════════════════════════════

QUESTION
  Can DoMINO, trained only on simple bluff bodies (Ahmed + Windsor),
  predict drag on a realistic road car (DrivAer) it has never seen?
  If not, how many DrivAer runs does it take to fix?

  NOT the paper's OOD (drag outside range, SAME body). Ours is a
  DIFFERENT BODY.

UPSTREAM (pinned)
  physicsnemo         @ 59aaf59
  physicsnemo-curator @ 86533e5
  physicsnemo-cfd     @ 0d2305e

SINGLE SOURCE OF TRUTH
  src/datasets.py — every field name, reference area, U_inf, and file
  convention, with provenance. NOTHING ELSE MAY HARDCODE THEM.

────────────────────────────────────────────────────────────────────
 ⚠  SEVEN SILENT TRAPS. None were in the plan. None raise an error.
    All produce a converging loss and a worthless model.
    READ §5 BEFORE DOING ANYTHING.
────────────────────────────────────────────────────────────────────
  1. loss.py:423 — channel 0 MUST be pressure, channel 1 MUST be
     x-wall-shear. Hardcoded positional indexing.

  2. NO retraining.py. Fine-tune via resume_dir + train.py — BUT IT
     RESTORES THE EPOCH COUNTER. A 500-epoch checkpoint with
     train.epochs=50 EXITS IMMEDIATELY having trained NOTHING, and
     writes a checkpoint. RECOVERY CURVE WOULD BE A FLAT LINE.
     → Patch #1: weights-only load + force init_epoch=0.

  3. force_mom_<i>.csv = CONSTANT ref in Ahmed/Windsor,
                         VARIABLE ref in DrivAer.
     Same filename, opposite meaning, across exactly the boundary we
     transfer over. 47 DRAG COUNTS on Ahmed.

  4. Windsor surface is .vtu with POINT data. Ahmed/DrivAer are .vtp
     with CELL data. A *.vtp glob finds nothing in Windsor.

  5. Windsor wall shear = 3 separate scalars, file order cfx, cfz,
     cfy — Z BEFORE Y. Stacking in array order permutes channels.

  6. U_inf spans 39× (Ahmed 1 m/s, DrivAer 38.889). Dimensional wall
     shear spans ~1500×. Must non-dimensionalize or DrivAer dominates
     the loss entirely.  ⚠ ρ is not published by ANY dataset — verify
     it in Phase 2 against their own Cp fields.

  7. Windsor is Y-UP (not z-up) and YAWED −2.5°. Drag is safe (x is
     streamwise in all three). LIFT AND Cf_y/Cf_z ARE NOT COMPARABLE.
     Phase 3 must decide on an axis swap.

  ✓ DEFUSED: streamwise axis IS x in all three. Verified, not assumed.

  ── Phase 2 found five MORE ──────────────────────────────────

  8. WALL SHEAR HAS THE OPPOSITE SIGN. OpenFOAM reports
     wall-on-fluid; a drag integral needs fluid-on-wall. Symptom:
     NEGATIVE viscous drag -- physically impossible. Negate it.
     (Windsor exempt: publishes Cf directly.)

  9. rho = 1.0, NOT 1.225. Incompressible solvers in KINEMATIC
     units -- rho never appears in their inputs, so assuming air is
     the natural mistake. Backed out from p/Cp, which IS q:
       Ahmed   p/Cp = 0.500000    (std 1.8e-9)  -> rho = 1.0
       DrivAer p/Cp = 756.177161  (std 2.7e-5)  -> rho = 1.0
     Not an estimate. An identity. With 1.225, Cd was 5% short.

 10. THREE DATASETS, THREE NORMAL STRATEGIES. Undocumented
     anywhere. One code path silently fails on two of three.
       ahmed    "clean"    58,308 open edges -> 0. Then VTK
                           orients correctly on its own.
       windsor  "shipped"  Dataset PROVIDES normals. Use them.
                           ** NEVER clean() -- see trap 11 **
       drivaer  "flip"     clean() does NOTHING (7,383 -> 7,383).
                           consistent_normals + GLOBAL flip.
                           Unflipped -0.2533, flipped +0.31093.

 11. clean() DESTROYS WINDSOR'S CELL ARRAYS. Merges cells, leaves
     arrays at old length: 4.99M cells vs 9.92M-entry cfxavg.
     pyvista only WARNS. Cd then computed from fields that no
     longer match the facets. Caught ONLY by the open-edge guard.

 12. INWARD NORMALS. Ahmed Cd = -0.218 vs published +0.238.
     The most visible of the five, and the LEAST dangerous --
     a negative Cd announces itself. The other four do not.

  ══════════════════════════════════════════════════════════════
   TWELVE TRAPS. NOT ONE RAISES AN ERROR. Every failure so far
   has produced a confident, plausible number. Not one has
   produced a stack trace. The exit criteria are the ONLY
   detectors that exist.
  ══════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────
 ⚠  REDUCED SCALE. See §Scale. Scale-invariant by construction —
    case counts live in conf/base.yaml, never in code.
────────────────────────────────────────────────────────────────────
  Ahmed   80  cases  (full study: 500)
  Windsor 40  cases  (full study: 355)
  DrivAer 40  cases  (full study: 500) → 20 holdout + 20 FT pool
  ~55 GB raw. Storage is shared across several projects.

  CUT: A3a (H5 unadjudicated). Wake VTUs (H2 asserted, not shown).
       Seed spread (curve is indicative, not evidential).
  RELAXED: EC1 from R² > 0.95 to R² > 0.80 — A1 trains on 20 cases.

  NOTHING ABOUT THE ENGINEERING IS EASIER AT THIS SCALE. All seven
  traps are live. Scaling up = edit four numbers, ask for more
  wall-clock. Nothing is rewritten.

────────────────────────────────────────────────────────────────────
 ARMS — RUN IN THIS ORDER
────────────────────────────────────────────────────────────────────
  A1   DrivAer(20) → DrivAer holdout(20)   instrument SMOKE TEST
  A2   DrivAer(40) → Ahmed+Windsor         CONTROL — ** GATE **
                                           If this fails, A3 is
                                           uninterpretable. STOP.
  A3   Ahmed(80)+Windsor(40) → DrivAer(20) ** THE EXPERIMENT **
  +    recovery: fine-tune A3 on N ∈ {0, 5, 20}, ONE seed
  =    5 runs

  THREE POINTS IS NOT A CURVE. Do not annotate a knee. Do not fit.
  At small N the identity of the cases dominates — with one seed that
  variance is INVISIBLE, not absent.

────────────────────────────────────────────────────────────────────
 PHASES
────────────────────────────────────────────────────────────────────
  [✓] 0  Recon + compatibility table              EC0 CLEARED
  [✓] 2  Force integration + metrics              EC2 CLEARED
         ahmed 0.029% | windsor 0.516% | drivaer 0.001%
         Robustness: 20 cases x 3 datasets = 60. ALL PASS.
         40 tests. DrivAer's global-flip held across every morph.
  [ ] 1  Baseline reproduction (Drag R² > 0.80)   EC1  ← RELAXED
  [ ] 3  Unified datapipe (assertion passes)      EC3
  [ ] 4  Three arms + recovery (5 runs)           EC4
  [ ] 5  Error localization (NO wake evidence)    EC5
  [ ] 6  Writeup: 3 figures, 1 sentence

────────────────────────────────────────────────────────────────────
 HYPOTHESIS LEDGER
 Registered in advance. Mark held/refuted, WITH THE NUMBER THAT
 DECIDED IT. DO NOT REWRITE THEM.
────────────────────────────────────────────────────────────────────
  H1  upward transfer degrades           [ ] held / refuted: ______
  H2  error localizes to absent features [ ] held / refuted: ______
  H3  error cancellation precedes acc.   [ ] held / refuted: ______
  H4  downward transfer succeeds (ctrl)  [ ] held / refuted: ______
  H5  source diversity buys transfer     [ ] held / refuted: ______

────────────────────────────────────────────────────────────────────
 DEFERRED ON PURPOSE — do not let me scope-creep into these
────────────────────────────────────────────────────────────────────
  · fidelity transfer (DrivAerNet → DrivAerML)
  · uncertainty quantification
  · volume-field study
  · physics losses
  · encoder-vs-aggregation ablation      ← THE FOLLOW-UP PAPER
  · diversity SCALING study               ← THE OTHER FOLLOW-UP
    (A3a gives TWO points. It is a control, not a scaling law.)
```

---

## Appendix C — Collaboration contract

> An assistant reading this document does not need help *understanding* it — it will read the
> whole thing. What it needs is a statement of **how to behave**: what to verify before acting,
> what to refuse, and what to push back on. **The failure modes below are not hypothetical.
> They are the specific ways a well-meaning collaborator degrades this study.**

### The prompt — paste this first

```
You are collaborating on the project specified in the attached plan.
Read all of it before responding. Then follow this contract.

═══ FIRST, ESTABLISH STATE ═══
Do not assume we are at Phase 0. Read the Session Restart Card
(Appendix B) and the Hypothesis Ledger. If STATUS looks stale relative
to what I'm telling you, ASK before proceeding.

Helping me restart a phase I already finished is a real and costly
failure — you have no memory of prior sessions, and this document is
the only state that persists.

═══ DIVISION OF LABOUR ═══
I run the compute, the data, and the cluster. You do NOT have GPU
access, dataset access, or a persistent filesystem.

You own: code, reasoning, debugging, and honesty checks.
Do not offer to "run" anything. Do ask me to paste outputs, error
traces, JSON, and numbers.

═══ THINGS TO REFUSE, EVEN IF I ASK ═══
These are not preferences. They void the study.

1. SCALING-FACTOR LEAK. surface_factors are computed on the SOURCE set
   ONLY and applied UNCHANGED to DrivAer. If I propose recomputing them
   on DrivAer — for any reason, however sensible it sounds — REFUSE.
   This is the error most likely to survive review and invalidate the
   headline claim.

   COROLLARY: A3a (Ahmed-only) has its OWN factors. If I reuse A3's
   Ahmed+Windsor factors for A3a, Windsor's statistics leak into the
   arm whose ENTIRE PURPOSE is to remove Windsor. Stop me.

2. HOLDOUT CONTAMINATION. drivaer_holdout/ is never fine-tuned on, at
   any N, under any seed. If a config would touch it, stop me.

3. HYPOTHESIS REWRITING. H1–H5 are REGISTERED PREDICTIONS, written
   before the experiment. If results contradict them, the hypothesis is
   REFUTED — not amended, not softened, not reinterpreted.

   A refuted hypothesis is a finding. A rewritten one is a lie.
   Hold this line even if I want to move it.

4. SKIPPING EXIT CRITERIA. Each phase has one. They exist because the
   dominant failures here are SILENT. If I want to move to Phase N+1
   without clearing Phase N, say so plainly and make me decide
   consciously.

   A2 is a GATE: if the downward control fails, A3 is uninterpretable
   and there is no point running it.

5. OVERCLAIMING NOVELTY. The DoMINO paper ALREADY tests OOD samples —
   but along a different axis (drag outside training range, SAME body).
   Ours is a different BODY. If I draft anything implying we invented
   OOD testing for DoMINO, correct it.

6. OVERCLAIMING FROM A3a. A3a vs A3 is TWO POINTS. It says whether
   adding one body family moves the number. It does NOT say how many
   are needed, and it does NOT separate diversity from sample count
   (Ahmed=500, Ahmed+Windsor=855). If I draft a diversity scaling law —
   or draw a line through two points on F1 — correct it.

═══ WHAT TO BE SUSPICIOUS OF ═══
· A cold-transfer (A3, N=0) Cd that looks GOOD.
  The plan PREDICTS degradation. If the number is suspiciously fine,
  check leakage FIRST, then normalization, then the force integrator.
  A neural operator does not spontaneously learn what a wheel does to
  flow.

· A FLAT recovery curve, or fine-tuned models identical to the cold
  model → the epoch trap (Trap 2). Fine-tuning trained ZERO STEPS and
  wrote a checkpoint anyway. Check init_epoch.

· A3a and A3 cold numbers IDENTICAL to several digits → the same
  scaling factors were used for both, or the same checkpoint was
  evaluated twice. Different runs on different data should not agree
  exactly.

· A loss curve that converges beautifully but results that make no
  sense → variable ordering (Trap 1). The loss looks healthy while
  regressing the wrong channels.

· Cd within tolerance but Spearman ρ poor → the model cannot rank
  designs. It is unusable for optimization REGARDLESS of MAE.

· Large Cp L2 with near-zero signed bias → error cancellation. This is
  H3. REPORT IT. Do not "fix" it away.

═══ TONE ═══
Tell me when I am wrong. Tell me when a result smells off. Do not
validate a number just because I sound confident about it. If a claim
in my draft is stronger than the evidence, say so before a reviewer
does.

═══ WHAT I WANT FROM YOU THIS SESSION ═══
[fill in]
```

### Why each clause exists

| Clause | Failure it prevents | Why it needs stating |
|---|---|---|
| **Establish state first** | Restarting a completed phase | The assistant has no memory across sessions. **This document is the only persistent state, and a stale paste looks identical to a fresh one.** |
| **Division of labour** | Wasted turns; false offers to execute | No GPU, no data, no filesystem. Useful on code and reasoning; useless on compute. |
| **Refuse the scaling leak** | **Voided headline result** | Recomputing on the target is the *natural* thing to do and **feels like housekeeping**. A compliant assistant will happily help. |
| **Refuse holdout contamination** | Meaningless recovery curve | The curve is only interpretable on cases never fine-tuned on. |
| **Refuse hypothesis rewriting** | Post-hoc rationalization | **The registered-prediction discipline is the study's main epistemic asset.** It is also the easiest thing to quietly abandon when results disappoint. |
| **Refuse skipping exit criteria** | Silent corruption propagating | **Every dominant risk here fails quietly. The criteria are the only detectors.** |
| **Refuse novelty overclaim** | Immediate loss of credibility | Anyone who has read §4 of the paper will catch it. |
| **Refuse overclaiming from A3a** | A scaling law asserted from two points | **A3a is the most *tempting* result to over-read**, precisely because it looks like the start of a curve. |
| **Suspect a good cold number** | Celebrating a leak | H1 predicts failure. **A good result is more likely a bug than a breakthrough — check the bug first.** |

> **The most important clause is the one about suspicion.**
>
> An assistant's default failure mode is agreeableness. If a cold-transfer number comes back
> looking good and the human sounds pleased, the path of least resistance is to congratulate
> and move on. **The registered hypotheses exist precisely so that a good result is
> *surprising* and therefore *suspect*.**
>
> Make the assistant hold that line — and notice **this cuts both ways.** It is also a line
> *you* will be tempted to cross, at 2am, when the number finally looks right.

---

## Sources

**Model**
- Ranade et al., *DoMINO: A Decomposable Multi-scale Iterative Neural Operator*, NVIDIA —
  [arXiv:2501.13350](https://arxiv.org/abs/2501.13350)

**Datasets** *(all Ashton et al., CC-BY-SA)*
- *DrivAerML* — [arXiv:2408.11969](https://arxiv.org/abs/2408.11969)
- *AhmedML* — [arXiv:2407.20801](https://arxiv.org/abs/2407.20801)
- *WindsorML* — [arXiv:2407.19320](https://arxiv.org/abs/2407.19320)
- [caemldatasets.org](https://caemldatasets.org)

**Code**
- [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) — model, training
- [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator) — preprocessing
- [PhysicsNeMo-CFD](https://github.com/NVIDIA/physicsnemo-cfd) — force computation *(Phase-2
  oracle)*