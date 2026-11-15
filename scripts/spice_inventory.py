#!/usr/bin/env python3
"""Inventory the SPICE archive before choosing subsets.

Writes results/spice_inventory.json: molecule and conformer counts per subset,
plus how many molecules actually contain MORE THAN ONE conformer — which is the
number that decides whether this project has any data at all, because a decision
pair needs two conformers of the same molecule.

The sample size is recorded in the output and printed, and any figure that uses
these counts must say it was a sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlipfid.spice import inventory  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "spice_inventory.json"))
    ap.add_argument("--sample", type=int, default=6000,
                    help="molecules to inspect (0 = every molecule)")
    args = ap.parse_args()

    sample = args.sample or 10 ** 9
    inv = inventory(args.archive, sample=sample)
    inv["archive"] = str(args.archive)
    inv["note"] = ("counts from the first %d molecules, not the whole archive"
                   % inv["n_sampled"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inv, indent=2) + "\n")

    print(f"archive molecules : {inv['n_total_molecules']}")
    print(f"sampled           : {inv['n_sampled']}")
    print(f"{'subset':26s} {'molecules':>10s} {'conformers':>11s} "
          f"{'multi-conf':>11s}")
    for s, d in sorted(inv["by_subset"].items(), key=lambda kv: -kv[1]["conformers"]):
        print(f"{s:26s} {d['molecules']:10d} {d['conformers']:11d} "
              f"{d['multi_conformer']:11d}")
    total_multi = sum(d["multi_conformer"] for d in inv["by_subset"].values())
    print(f"\nmolecules with >1 conformer in sample: {total_multi}")
    if total_multi == 0:
        print("WARNING: no multi-conformer molecules; the decision-pair design "
              "cannot be built from this archive as read. Investigate the HDF5 "
              "layout before continuing.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
