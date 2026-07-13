# download.py
# ============================================================
# PURPOSE:
#   Fetch the raw data for this study from HuggingFace. Surface and geometry
#   ONLY -- never volumes.
#
# ** WHY THIS IS A SCRIPT AND NOT A COMMAND YOU TYPE **
#
#   The volume files are 49 GB PER CASE. DrivAer's volume_1.vtu alone is 49 GB,
#   split across two .part files. A single careless wildcard -- `--include
#   "run_1/*"` instead of naming the files -- pulls 49 GB where you wanted 800 MB.
#   Do that for 40 cases and you have consumed 2 TB and blown a 500 GB quota.
#
#   So the includes are NAMED, explicitly, per file type, and the volume patterns
#   appear NOWHERE in this file. There is nothing to typo.
#
# ** THE SIZE ASYMMETRY, MEASURED **
#
#     DrivAer, one case:
#         drivaer_1.stl       142 MB
#         boundary_1.vtp      660 MB
#         *.csv                <1 MB
#         ------------------------------
#         what we take:      ~800 MB
#
#         volume_1.vtu     49,052 MB   <- 61x larger. NEVER DOWNLOADED.
#
#   Full DrivAer volumes would be 24.5 TB. This study takes ~55 GB total.
#
# ** SCALE INVARIANCE **
#   Case counts are ARGUMENTS, not constants. Scaling this study up to the full
#   500/355/500 means passing different numbers -- not editing code. That promise
#   is made in the plan and it has to be true.
#
# USAGE:
#     python src/download.py --out /path/to/data
#     python src/download.py --out /path/to/data --ahmed 500 --windsor 355 --drivaer 500
#     python src/download.py --out /path/to/data --dry-run
#
# DEPENDENCIES: huggingface_hub. No GPU. Runs on a login node.
# ============================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


# HuggingFace repo IDs. All three published by the same group (Ashton et al.),
# all CC-BY-SA.
REPOS = {
    "ahmed": "neashton/ahmedml",
    "windsor": "neashton/windsorml",
    "drivaer": "neashton/drivaerml",
}

# The reduced scale for this study. See the plan's Scale section for what these
# cost. They are DEFAULTS, overridable on the command line -- the full study is
# 500 / 355 / 500 and requires no code change.
DEFAULT_N = {
    "ahmed": 80,
    "windsor": 40,
    "drivaer": 40,
}

# ** THE FILES WE TAKE. **
#
# Named per-type rather than globbed, because a glob is how you accidentally
# download a 49 GB volume mesh. Note the surface file differs BY DATASET:
# Windsor ships .vtu where the others ship .vtp -- a Phase-0 finding, and a
# loader written for one finds nothing in the other.
#
# The CSVs are free (kilobytes) and we take all of them, because the reference-
# area trap means we need BOTH force files: the one we use, and the one we must
# be able to prove we did NOT use.
def includes_for(dataset: str, run: int) -> list[str]:
    stem = {"ahmed": "ahmed", "windsor": "windsor", "drivaer": "drivaer"}[dataset]
    surface_ext = "vtu" if dataset == "windsor" else "vtp"
    return [
        f"run_{run}/{stem}_{run}.stl",              # geometry
        f"run_{run}/boundary_{run}.{surface_ext}",  # surface fields
        f"run_{run}/*.csv",                         # forces + geo params
    ]


def sizes(dataset: str, n_cases: int) -> dict:
    """
    Query file sizes WITHOUT downloading anything.

    WHY THIS EXISTS:
        So that --dry-run can tell you what you are about to consume BEFORE you
        consume it. On a shared cluster with a hard quota, "download and find out"
        is not an acceptable workflow.
    """
    api = HfApi()
    info = api.repo_info(REPOS[dataset], repo_type="dataset", files_metadata=True)

    wanted = set()
    for run in range(1, n_cases + 1):
        stem = {"ahmed": "ahmed", "windsor": "windsor", "drivaer": "drivaer"}[dataset]
        ext = "vtu" if dataset == "windsor" else "vtp"
        wanted.add(f"run_{run}/{stem}_{run}.stl")
        wanted.add(f"run_{run}/boundary_{run}.{ext}")

    total = 0
    found = 0
    for sibling in info.siblings:
        name = sibling.rfilename
        if name in wanted:
            total += sibling.size or 0
            found += 1
        # CSVs: any file in one of our runs, ending .csv
        elif name.endswith(".csv"):
            run_dir = name.split("/")[0]
            if run_dir.startswith("run_"):
                try:
                    if int(run_dir[4:]) <= n_cases:
                        total += sibling.size or 0
                except ValueError:
                    pass

    return {
        "dataset": dataset,
        "n_cases": n_cases,
        "gb": total / 1e9,
        "files_matched": found,
        "files_expected": len(wanted),
    }


def download(dataset: str, n_cases: int, out_root: Path) -> None:
    """
    Fetch n_cases from one dataset into out_root/<dataset>ml/.

    WHY snapshot_download RATHER THAN A LOOP OF hf_hub_download:
        It parallelizes, and it resumes. On a cluster login node a 30 GB transfer
        WILL be interrupted -- by a timeout, a dropped VPN, a maintenance window.
        Re-running this script then picks up where it left off rather than
        starting over.
    """
    allow: list[str] = []
    for run in range(1, n_cases + 1):
        allow.extend(includes_for(dataset, run))

    dest = out_root / f"{dataset}ml"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n  {dataset}: {n_cases} cases -> {dest}", flush=True)

    snapshot_download(
        repo_id=REPOS[dataset],
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow,
        # NOT specifying ignore_patterns for volumes: allow_patterns is a
        # whitelist, so anything not named above is already excluded. Relying on
        # a blacklist to keep out a 49 GB file would be the wrong way round.
        max_workers=4,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="destination root; creates <out>/ahmedml/ etc.")
    p.add_argument("--ahmed", type=int, default=DEFAULT_N["ahmed"])
    p.add_argument("--windsor", type=int, default=DEFAULT_N["windsor"])
    p.add_argument("--drivaer", type=int, default=DEFAULT_N["drivaer"])
    p.add_argument("--dry-run", action="store_true",
                   help="report sizes; download nothing")
    args = p.parse_args()

    counts = {"ahmed": args.ahmed, "windsor": args.windsor, "drivaer": args.drivaer}

    print("\n  Querying file sizes (no download)...\n")
    print(f"  {'dataset':<10} {'cases':>6} {'size':>10}")
    print(f"  {'-' * 30}")

    total_gb = 0.0
    for name, n in counts.items():
        s = sizes(name, n)
        total_gb += s["gb"]
        print(f"  {name:<10} {n:>6} {s['gb']:>8.1f} GB")
        if s["files_matched"] != s["files_expected"]:
            print(f"             ** only {s['files_matched']}/{s['files_expected']} "
                  f"expected files found -- check the case count **")

    print(f"  {'-' * 30}")
    print(f"  {'TOTAL':<10} {'':>6} {total_gb:>8.1f} GB\n")

    if args.dry_run:
        print("  --dry-run: nothing downloaded.\n")
        return 0

    resp = input(f"  Download {total_gb:.1f} GB to {args.out}? [y/N] ")
    if resp.strip().lower() != "y":
        print("  aborted.\n")
        return 1

    for name, n in counts.items():
        download(name, n, args.out)

    print(f"\n  done. {total_gb:.1f} GB in {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())