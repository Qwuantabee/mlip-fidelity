#!/usr/bin/env python3
"""Build the decision-pair manifest from SPICE, and export the conformers to score.

Two outputs, both committed:
  results/pairs_manifest.json   every decision pair with its QM gap (kcal/mol)
  results/conformers.jsonl      one line per conformer actually needed (Z, xyz)

The manifest is built once and reused by every model, so all models are scored on
byte-identical geometry. Regenerating it with the same --seed must give the same
file; that is asserted by a test.

Subsets: SPICE keys carry their subset as a prefix. The biomolecular slice
(dipeptides, solvated amino acids) is the one a protein-engineering reader cares
about, so it is kept as its own tag on every pair rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlipfid.decisions import HARTREE2KCAL, pairs_from_molecule  # noqa: E402
from mlipfid.spice import iter_molecules  # noqa: E402

BIO_SUBSETS = {"dipeptide", "solvated_amino_acids", "amino_acids"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--max-molecules", type=int, default=3000,
                    help="cap on molecules scanned (0 = all)")
    ap.add_argument("--max-pairs-per-molecule", type=int, default=20)
    ap.add_argument("--max-conformers-per-molecule", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict] = []
    conformers: dict[str, dict] = {}
    per_subset: dict[str, dict[str, int]] = {}
    n_mol = 0

    for mol in iter_molecules(args.archive, limit=args.max_molecules or None):
        n_mol += 1
        d = per_subset.setdefault(mol.subset, {"molecules": 0, "pairs": 0,
                                               "conformer_sets": 0})
        d["molecules"] += 1
        if mol.n < 2:
            continue
        # Cap conformers per molecule: SPICE molecules can hold dozens of
        # near-identical geometries, and pairwise growth is quadratic.
        idx = np.arange(mol.n)
        if mol.n > args.max_conformers_per_molecule:
            idx = np.sort(rng.choice(idx, args.max_conformers_per_molecule,
                                     replace=False))
        sub = [mol.conformers[i] for i in idx]
        energies = np.array([c.energy_hartree for c in sub])
        local = pairs_from_molecule(
            mol.key, mol.subset, energies,
            max_pairs=args.max_pairs_per_molecule, rng=rng)
        if not local:
            continue
        d["conformer_sets"] += 1
        # map local index -> global conformer id
        for p in local:
            a, b = sub[p.idx_a], sub[p.idx_b]
            id_a = f"{mol.key}|{idx[p.idx_a]}"
            id_b = f"{mol.key}|{idx[p.idx_b]}"
            for cid, c in ((id_a, a), (id_b, b)):
                if cid not in conformers:
                    conformers[cid] = {
                        "id": cid,
                        "molecule": mol.key,
                        "subset": mol.subset,
                        "atomic_numbers": [int(z) for z in c.atomic_numbers],
                        "positions": [[round(float(v), 5) for v in row]
                                      for row in c.positions],
                        "energy_hartree": float(c.energy_hartree),
                    }
            pairs.append({
                "molecule": mol.key,
                "subset": mol.subset,
                "is_bio": mol.subset in BIO_SUBSETS,
                "id_a": id_a,
                "id_b": id_b,
                "qm_gap_kcal": round(float(p.qm_gap_kcal), 6),
            })
            d["pairs"] += 1
        if args.max_molecules and n_mol >= args.max_molecules:
            break

    manifest = {
        "seed": args.seed,
        "n_molecules_scanned": n_mol,
        "n_pairs": len(pairs),
        "n_conformers": len(conformers),
        "n_bio_pairs": sum(1 for p in pairs if p["is_bio"]),
        "max_pairs_per_molecule": args.max_pairs_per_molecule,
        "max_conformers_per_molecule": args.max_conformers_per_molecule,
        "per_subset": per_subset,
        "pairs": pairs,
    }
    (out_dir / "pairs_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    with open(out_dir / "conformers.jsonl", "w") as f:
        for cid in sorted(conformers):
            f.write(json.dumps(conformers[cid]) + "\n")

    print(f"molecules scanned : {n_mol}")
    print(f"conformers exported: {len(conformers)}")
    print(f"decision pairs     : {len(pairs)} (bio {manifest['n_bio_pairs']})")
    for s, d in sorted(per_subset.items(), key=lambda kv: -kv[1]["pairs"]):
        print(f"  {s:24s} mol={d['molecules']:5d} pairs={d['pairs']:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
