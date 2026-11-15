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

Output: results/heldout_folmsbee.json in the same shape as the MPCONF196 file, so
the same scorer reads both.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

    import pandas as pd
    df = pd.read_csv(table)
    df = df.dropna(subset=["dlpno", "geom", "name"])

    def find_geom(name: str, geom: str) -> Path | None:
        for sub in ("Neutral_jobs", "CHG_jobs"):
            p = root / "geometries" / sub / name / geom
            if p.exists():
                return p
        return None

    pairs: list[dict] = []
    conformers: dict[str, dict] = {}
    per_compound: dict[str, int] = defaultdict(int)
    skipped_geom = 0

    for name, g in df.groupby("name"):
        g = g.head(args.max_conformers)
        if len(g) < 2:
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
                    p = find_geom(name, gm)
                    if p is None:
                        skipped_geom += 1
                        break
                    try:
                        z, xyz = read_mol(p)
                    except Exception:  # noqa: BLE001
                        skipped_geom += 1
                        break
                    conformers[cid] = {
                        "id": cid, "compound": name, "atomic_numbers": z,
                        "positions": xyz, "charge": 0, "spin": 1,
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

    out = {
        "source": "Folmsbee & Hutchison 2021 (Int. J. Quantum Chem. e26381), "
                  "DLPNO-CCSD(T) single points on B3LYP-D3BJ minima",
        "note": "Held out from both ML models' training data. ref_kcal is "
                "E(B) - E(A), built from the atomization convention "
                "(relative = max - value).",
        "units": "positions in Angstrom (molfiles); dlpno column already kcal/mol",
        "window_kcal": args.max_window_kcal,
        "n_pairs": len(pairs),
        "n_conformers": len(conformers),
        "n_compounds": len(per_compound),
        "skipped_geometry": skipped_geom,
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
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
