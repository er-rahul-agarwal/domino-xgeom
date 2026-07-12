# Environment

Recorded at project start. Every number in this study is traceable to this
environment. If results fail to reproduce, this file is the first thing to check.

## Upstream repositories

Cloned into `external/`, which is gitignored -- see `.gitignore` for the rationale.
Our code is never mixed with theirs. To reproduce: clone each repository at the
commit below, then apply `patches/`.

| Repository          | Commit                                     | Purpose                                        |
|---------------------|--------------------------------------------|------------------------------------------------|
| physicsnemo         | 59aaf59b48901bd8df2c9bad83f6d05ea47d8c04   | DoMINO model; train / test / retraining        |
| physicsnemo-curator | 86533e581b3550326d89e97cb4d4126e7061b416   | Preprocessing: raw STL/VTP to zarr/npy         |
| physicsnemo-cfd     | 0d2305e1777351569b1795ce38884ee945491d28   | Force-computation oracle (Phase 2 cross-check) |

All three cloned 2026-07-11.

NOTE ON NAMING: PhysicsNeMo was formerly called "Modulus". The DoMINO paper and
older tutorials refer to `NVIDIA/modulus`. It is the same codebase under a new
name, not a different project.

## Local development machine

Used for: analysis code (`src/`), configs, figures, and Phase-0 reconnaissance on
a small subset of cases. NOT used for training.

- OS: Windows (PowerShell)
- Python: 3.13.2
- Git: 2.49.0.windows.1

CAVEAT: Python 3.13 is newer than most CUDA and ML packages currently support.
This is not yet a problem, because local code needs only pyvista, numpy, scipy,
and matplotlib. If the install fights us, the fix is a 3.11 virtual environment.
Recorded here so the decision is traceable rather than rediscovered later.

## Cluster

Used for: all 22 training runs, curator preprocessing, and compute_statistics.
Every field below is TODO because cluster access is not yet configured. These are
not optional -- the run matrix cannot be sized without them.

- Scheduler (Slurm / PBS / other): TODO
- GPU type and count per node: TODO
- Wall-clock limit per job: TODO
- Scratch quota: TODO
- Scratch purge policy: TODO
- Compute nodes have internet access: TODO
- CUDA version: TODO
- Driver version: TODO

## Reproduction

TODO -- to be filled once the pipeline runs end to end.