#!/usr/bin/env python3
"""Build a HELD-OUT conformer benchmark from MPCONF196.

Why this exists. A reviewer established that both ML rows in this repo are
train/test contaminated: MACE-OFF23 trains on ~95% of SPICE v1, and ORB-omol's
OMol25 documents SPICE2 as a source. So the SPICE numbers measure *fidelity to
their own training data*, not generalisation. MPCONF196 is a different dataset
entirely (Rezac 2018; references revised 2023 at PNO-LCCSD(T)-F12 quality), with
small peptides and medium macrocycles. Neither model has seen it.

What it provides:
  ref.csv        183 pairs "X_a -> X_b" with a reference energy difference
  <name>/coord   Turbomole $coord block: x y z element, in BOHR
  <name>/mol.xyz element order

Units, verified not assumed: the shortest contacts in the raw coordinates are
N-H at 1.922 and C-H at 2.065. Those are correct only as BOHR (1.017 A and
1.093 A); as Angstrom they would be impossible. Same trap as SPICE, so the same
conversion. (scripts/check_artefacts.py does not read the held-out artefacts; an
earlier version of this docstring said it re-derived this conversion.)

Output: results/heldout_mpconf196.json
  {"pairs": [{id_a, id_b, compound, ref_kcal}], "conformers": {...}}

Note the gaps here are NOT restricted to 1 kcal/mol -- they run from -10 to +40
kcal/mol. That is the point: this benchmark measures whether ordering survives at
*all* gap sizes, which is exactly the psychometric curve the SPICE pair set can
only cover the bottom of.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BOHR2ANG = 0.52917721067
SYMBOLS = {
    "h": 1, "he": 2, "li": 3, "be": 4, "b": 5, "c": 6, "n": 7, "o": 8, "f": 9,
    "ne": 10, "na": 11, "mg": 12, "al": 13, "si": 14, "p": 15, "s": 16,
    "cl": 17, "ar": 18, "k": 19, "ca": 20, "fe": 26, "zn": 30, "br": 35,
    "i": 53,
}


def read_coord(path: Path) -> tuple[list[int], list[list[float]]]:
    """Turbomole $coord -> (atomic_numbers, positions in Angstrom)."""
    z: list[int] = []
    xyz: list[list[float]] = []
    inside = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s == "$coord":
            inside = True
            continue
        if not inside:
            continue
        if s.startswith("$"):
            break
        if not s:
            continue
        parts = s.split()
        if len(parts) < 4:
            raise ValueError(f"{path}: unexpected line {line!r}")
        elem = parts[3].split(".")[0].lower()
        if elem not in SYMBOLS:
            raise ValueError(f"{path}: unknown element {parts[3]!r}")
        z.append(SYMBOLS[elem])
        xyz.append([round(float(v) * BOHR2ANG, 6) for v in parts[:3]])
    if not z:
        raise ValueError(f"{path}: no $coord block")
    return z, xyz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent
                                          / "_data" / "mpconf196"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "results" / "heldout_mpconf196.json"))
    args = ap.parse_args()

    root = Path(args.root)
    ref = root / "ref.csv"
    if not ref.exists():
        raise SystemExit(f"missing {ref}; clone github.com/grimme-lab/benchmark-MPCONF196")

    pairs = []
    for line in ref.read_text().splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        rid, equation, value = line.split(",")
        a, _, b = equation.partition(" -> ")
        if not b:
            raise SystemExit(f"unparsed equation: {equation!r}")
        pairs.append({
            "id_a": f"{a.strip()}",
            "id_b": f"{b.strip()}",
            "compound": a.strip().rsplit("_", 1)[0],
            "ref_id": int(rid),
            "ref_kcal": float(value),
        })

    # Every conformer named by any pair, plus its geometry.
    needed = sorted({p["id_a"] for p in pairs} | {p["id_b"] for p in pairs})
    conformers: dict[str, dict] = {}
    for cid in needed:
        cdir = root / "MPCONF196" / cid
        coord = cdir / "coord"
        if not coord.exists():
            raise SystemExit(f"missing geometry {coord}")
        z, xyz = read_coord(coord)
        conformers[cid] = {
            "id": cid,
            "compound": cid.rsplit("_", 1)[0],
            "atomic_numbers": z,
            "positions": xyz,
            "charge": 0,
            "spin": 1,
        }

    # Sanity: a pair's two members must be the same molecule.
    bad = 0
    for p in pairs:
        if conformers[p["id_a"]]["atomic_numbers"] != conformers[p["id_b"]]["atomic_numbers"]:
            bad += 1
    if bad:
        raise SystemExit(f"{bad} pairs join different compositions - refusing")

    per_compound = defaultdict(int)
    for p in pairs:
        per_compound[p["compound"]] += 1

    out = {
        "source": "MPCONF196 (Rezac 2018, JCTC; references revised 2023, "
                  "PNO-LCCSD(T)-F12)",
        "note": "Held out from both ML models' training data. Gap sign is what "
                "the scorer consumes; ref_kcal is the reference difference "
                "E(B) - E(A).",
        "units": "positions in Angstrom (converted from Turbomole bohr)",
        "n_pairs": len(pairs),
        "n_conformers": len(conformers),
        "compounds": {k: v for k, v in sorted(per_compound.items())},
        "pairs": pairs,
        "conformers": conformers,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    gaps = sorted(abs(p["ref_kcal"]) for p in pairs)
    print(f"compounds   : {len(per_compound)}")
    print(f"conformers  : {len(conformers)}")
    print(f"pairs       : {len(pairs)}")
    print(f"|gap| kcal  : min {gaps[0]:.2f}  median {gaps[len(gaps)//2]:.2f}  max {gaps[-1]:.2f}")
    print(f"within 1 kcal/mol: {sum(1 for g in gaps if g <= 1)}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
