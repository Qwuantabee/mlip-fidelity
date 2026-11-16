#!/usr/bin/env python3
"""Build a second held-out benchmark from Folmsbee & Hutchison (2021).

Why: the MPCONF196 held-out set produced only 30 sub-kcal pairs, which is the
binding limitation on the contamination finding. Folmsbee is 708 drug-like
molecules with up to 10 optimised local minima each and DLPNO-CCSD(T) references,
so it supplies far more near-degenerate pairs while staying fully out of both ML
models' training data.

Three unit facts, each established by MEASUREMENT rather than assumption,
because all three have already burned this repo once:

1. Coordinates are ANGSTROM. Shortest contacts in a raw molfile are C-H 1.088;
   in bohr they would be impossible. (SPICE and MPCONF196 are bohr.)
2. The `dlpno` column is ALREADY kcal/mol. compile-conformer-data.py builds it as
   `(totalAtomE - ccsd) * hartree_to_kcal`, so multiplying by 627.5 again turns a
   0.52 kcal/mol spread into 327.7 - which is exactly the error this file's first
   probe made.
3. It is an ATOMIZATION energy: a more stable conformer has a LARGER value.
   Relative energy is therefore (max - value), and using the raw value inverts
   the ranking.

A fourth input fact, learned from review R2 (2026-11-16): the set is NOT all
neutral. Folmsbee keeps 86 compounds under geometries/CHG_jobs/, with their formal
charges in geometries/CHG-charges.txt (+1: 64, -1: 10, +2: 4, 0: 8). The first
version of this builder wrote charge 0, singlet for every conformer, so 74 charged
compounds (8.7% of the pairs) were scored as the wrong electronic state: ORB takes
the charge as input and RDKit's DetermineBonds needs it for the MMFF baseline.
Now a compound's charge comes from the directory its geometry lives in:
Neutral_jobs -> 0, CHG_jobs -> the value in CHG-charges.txt. A CHG_jobs compound
with no listed charge is EXCLUDED (and counted), never scored as neutral. Spin
multiplicity follows electron-count parity (even -> 1, odd -> 2); with the true
charges every conformer has an even count, so all are singlets.

Output: results/heldout_folmsbee.json in the same shape as the MPCONF196 file, so
the same scorer reads both.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/home/ubuntu/biotech-portfolio/_data/folmsbee")


def read_mol(path: Path) -> tuple[list[int], list[list[float]]]:
    """V2000 molfile -> (atomic_numbers, positions in Angstrom).

    RDKit is used rather than a hand parser because these files come from
    OpenBabel and the fixed-width columns are not always exactly 10 wide.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
    if m is None:
        raise ValueError(f"unreadable molfile {path}")
    conf = m.GetConformer()
    z = [a.GetAtomicNum() for a in m.GetAtoms()]
    xyz = [[round(v, 6) for v in conf.GetAtomPosition(i)] for i in range(m.GetNumAtoms())]
    return z, xyz


def read_charges(path: Path) -> dict[str, int]:
    """Parse geometries/CHG-charges.txt: one `<name> CHARGE=<int>` per line.

    Strict on purpose: a line that does not parse raises rather than being
    skipped, because a skipped line would silently turn a charged compound into a
    neutral one - the exact failure this function exists to prevent.
    """
    charges: dict[str, int] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        m = re.fullmatch(r"\s*(\S+)\s+CHARGE=\s*([+-]?\d+)\s*", line)
        if m is None:
            raise ValueError(f"{path}:{n}: unparseable charge line {line!r}")
        name, q = m.group(1), int(m.group(2))
        if name in charges and charges[name] != q:
            raise ValueError(f"{path}:{n}: conflicting charges for {name}")
        charges[name] = q
    return charges


def compound_charge(source_dir: str, name: str,
                    charges: dict[str, int]) -> int | None:
    """Formal charge of a compound, or None when it cannot be established.

    Neutral_jobs compounds are neutral by the dataset's own split. A CHG_jobs
    compound takes its listed charge; if it is not listed the answer is None and
    the caller must exclude the compound - never fall back to 0.
    """
    if source_dir == "Neutral_jobs":
        return 0
    if source_dir == "CHG_jobs":
        return charges.get(name)
    raise ValueError(f"unknown geometry directory {source_dir!r}")


def spin_multiplicity(atomic_numbers: list[int], charge: int) -> int:
    """Lowest multiplicity consistent with the electron count (2S+1)."""
    n_electrons = sum(int(z) for z in atomic_numbers) - int(charge)
    if n_electrons < 0:
        raise ValueError("negative electron count")
    return 1 if n_electrons % 2 == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA))
    ap.add_argument("--max-window-kcal", type=float, default=1.0,
                    help="keep only pairs whose reference gap is within this "
                         "(matches the SPICE decision window, which is the slice "
                         "the contamination comparison needs)")
    ap.add_argument("--max-conformers", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "heldout_folmsbee.json"))
    args = ap.parse_args()

    root = Path(args.root)
    table = root / "data-final.csv"
    if not table.exists():
        raise SystemExit(f"missing {table}; clone github.com/ghutchis/conformer-benchmark")

    charge_file = root / "geometries" / "CHG-charges.txt"
    if not charge_file.exists():
        raise SystemExit(f"missing {charge_file}; without it charged compounds "
                         "would be scored as neutral")
    charges = read_charges(charge_file)

    import pandas as pd
    df = pd.read_csv(table)
    df = df.dropna(subset=["dlpno", "geom", "name"])

    def find_geom(name: str, geom: str) -> tuple[Path, str] | None:
        hits = [(root / "geometries" / sub / name / geom, sub)
                for sub in ("Neutral_jobs", "CHG_jobs")
                if (root / "geometries" / sub / name / geom).exists()]
        if len(hits) > 1:
            raise SystemExit(f"{name}/{geom} exists in both Neutral_jobs and "
                             "CHG_jobs; its charge is ambiguous")
        return hits[0] if hits else None

    pairs: list[dict] = []
    conformers: dict[str, dict] = {}
    per_compound: dict[str, int] = defaultdict(int)
    skipped_geom = 0
    excluded_no_charge: list[str] = []
    no_geometry_dir: list[str] = []

    for name, g in df.groupby("name"):
        g = g.head(args.max_conformers)
        if len(g) < 2:
            continue
        sources = [s for s in ("Neutral_jobs", "CHG_jobs")
                   if (root / "geometries" / s / name).is_dir()]
        if len(sources) > 1:
            raise SystemExit(f"{name} has geometries in both Neutral_jobs and "
                             "CHG_jobs; its charge is ambiguous")
        if not sources:
            no_geometry_dir.append(name)
            continue
        q = compound_charge(sources[0], name, charges)
        if q is None:
            excluded_no_charge.append(name)
            continue
        # Atomization convention: larger = more stable, so relative = max - value.
        vals = g.dlpno.to_numpy(dtype=float)
        rel = vals.max() - vals          # kcal/mol, >= 0 by construction
        geoms = list(g.geom)
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                gap = float(rel[j] - rel[i])
                if abs(gap) > args.max_window_kcal or abs(gap) < 1e-9:
                    continue
                id_a = f"{name}|{geoms[i]}"
                id_b = f"{name}|{geoms[j]}"
                for cid, gm in ((id_a, geoms[i]), (id_b, geoms[j])):
                    if cid in conformers:
                        continue
                    hit = find_geom(name, gm)
                    if hit is None:
                        skipped_geom += 1
                        break
                    p, source = hit
                    try:
                        z, xyz = read_mol(p)
                    except Exception:  # noqa: BLE001
                        skipped_geom += 1
                        break
                    conformers[cid] = {
                        "id": cid, "compound": name, "atomic_numbers": z,
                        "positions": xyz, "charge": q,
                        "spin": spin_multiplicity(z, q),
                        "charge_source": ("CHG-charges.txt" if source == "CHG_jobs"
                                          else "Neutral_jobs"),
                    }
                else:
                    pairs.append({
                        "id_a": id_a, "id_b": id_b, "compound": name,
                        "ref_kcal": round(gap, 6),
                    })
                    per_compound[name] += 1

    if not pairs:
        raise SystemExit("no pairs built - check the window and the dataframe")

    # A pair's members must be the same molecule.
    for p in pairs:
        if conformers[p["id_a"]]["atomic_numbers"] != conformers[p["id_b"]]["atomic_numbers"]:
            raise SystemExit(f"{p['id_a']} joins different compositions")

    # The bug this file shipped with, as an invariant: no compound the source
    # lists as charged may reach the models as neutral.
    for c in conformers.values():
        listed = charges.get(c["compound"])
        if c["charge_source"] == "CHG-charges.txt" and c["charge"] != listed:
            raise SystemExit(f"{c['id']}: charge {c['charge']} != listed {listed}")
    charge_dist = Counter(c["charge"] for c in conformers.values())
    spin_dist = Counter(c["spin"] for c in conformers.values())
    charged_compounds = sorted({c["compound"] for c in conformers.values()
                                if c["charge"] != 0})
    charged_pairs = sum(1 for p in pairs
                        if conformers[p["id_a"]]["charge"] != 0)

    out = {
        "source": "Folmsbee & Hutchison 2021 (Int. J. Quantum Chem. e26381), "
                  "DLPNO-CCSD(T) single points on B3LYP-D3BJ minima",
        "note": "Not part of either ML model's training sets by construction "
                "(molecule-level overlap with SPICE is not checked here). "
                "ref_kcal is E(B) - E(A), built from the atomization convention "
                "(relative = max - value). Charges: Neutral_jobs -> 0, "
                "CHG_jobs -> geometries/CHG-charges.txt; spin multiplicity from "
                "electron-count parity.",
        "units": "positions in Angstrom (molfiles); dlpno column already kcal/mol",
        "window_kcal": args.max_window_kcal,
        "n_pairs": len(pairs),
        "n_conformers": len(conformers),
        "n_compounds": len(per_compound),
        "skipped_geometry": skipped_geom,
        "n_compounds_no_geometry_dir": len(no_geometry_dir),
        "n_compounds_excluded_no_charge": len(excluded_no_charge),
        "compounds_excluded_no_charge": excluded_no_charge,
        "n_charged_compounds": len(charged_compounds),
        "n_pairs_charged": charged_pairs,
        "charge_distribution_conformers": {str(k): v for k, v in sorted(charge_dist.items())},
        "spin_distribution_conformers": {str(k): v for k, v in sorted(spin_dist.items())},
        "pairs": pairs,
        "conformers": conformers,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    gaps = np.array([abs(p["ref_kcal"]) for p in pairs])
    print(f"compounds  : {len(per_compound)}")
    print(f"conformers : {len(conformers)}")
    print(f"pairs      : {len(pairs)}  (|gap| <= {args.max_window_kcal} kcal/mol)")
    print(f"|gap| kcal : median {np.median(gaps):.3f}  max {gaps.max():.3f}")
    print(f"skipped    : {skipped_geom} unreadable/missing geometries")
    print(f"charged    : {len(charged_compounds)} compounds, {charged_pairs} pairs; "
          f"conformer charges {dict(sorted(charge_dist.items()))}, "
          f"spins {dict(sorted(spin_dist.items()))}")
    print(f"excluded   : {len(excluded_no_charge)} compounds with no established "
          f"charge {excluded_no_charge[:10]}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
