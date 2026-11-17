#!/usr/bin/env python3
"""Build the decision-pair manifest from SPICE, and export the conformers to score.

Two outputs, both committed:
  results/pairs_manifest.json   every decision pair with its QM gap (kcal/mol)
  results/conformers.jsonl      one line per conformer actually needed (Z, xyz)

The manifest is built once and reused by every model, so all models are scored on
byte-identical geometry. Regenerating it with the same --seed must give the same
file; a test asserts that.

Design decisions that are measured, not assumed:

* Conformers are selected with `select.select_conformers` (energy-stratified
  inside a 10 kcal/mol window), not at random. On 40 real molecules, random
  subsampling of 12 conformers gave 120 sub-kcal decision pairs; stratified
  selection gave 784. Random selection would have produced a repo claiming that
  near-degenerate conformer pairs barely exist.
* Charge and multiplicity are taken from each molecule's SMILES, because ~5% of
  SPICE molecules are ions (formal charge measured up to +2 and -6). Running them
  as neutral returns finite energies that are simply the wrong electronic state.
* Quotas are per subset, and the biomolecular subsets (dipeptides, solvated amino
  acids) are taken in full, because they are the slice a protein-engineering
  reader actually cares about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlipfid.decisions import HARTREE2KCAL, pairs_from_molecule  # noqa: E402
from mlipfid.select import charge_and_multiplicity, select_conformers  # noqa: E402
from mlipfid.spice import BOHR2ANG, _as_str, read_molecule, slug_subset  # noqa: E402

# Subsets a protein/peptide chemist would recognise as their own data.
BIO_SUBSETS = {"dipeptide", "solvated_amino_acids", "amino_acids"}
# Taken in full; everything else is capped by --max-per-subset.
FULL_SUBSETS = BIO_SUBSETS


def subset_map(archive: str) -> dict[str, str]:
    """key -> subset slug, reading only the tiny `subset` dataset."""
    out: dict[str, str] = {}
    with h5py.File(archive, "r") as f:
        for key in f.keys():
            g = f[key]
            if "subset" in g:
                try:
                    out[key] = slug_subset(_as_str(g["subset"]))
                    continue
                except Exception:  # noqa: BLE001
                    pass
            out[key] = "spice"
    return out


def choose_keys(smap: dict[str, str], max_per_subset: int, seed: int) -> dict[str, list[str]]:
    """Which molecules to read, grouped by subset, deterministic for a seed."""
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[str]] = {}
    for key, sub in sorted(smap.items()):
        grouped.setdefault(sub, []).append(key)
    chosen: dict[str, list[str]] = {}
    for sub, keys in grouped.items():
        keys = sorted(keys)
        if sub in FULL_SUBSETS or max_per_subset <= 0 or len(keys) <= max_per_subset:
            chosen[sub] = keys
        else:
            idx = np.sort(rng.choice(len(keys), max_per_subset, replace=False))
            chosen[sub] = [keys[i] for i in idx]
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--max-per-subset", type=int, default=900,
                    help="cap on molecules from non-bio subsets (0 = all)")
    ap.add_argument("--max-pairs-per-molecule", type=int, default=20)
    ap.add_argument("--max-conformers-per-molecule", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    smap = subset_map(args.archive)
    chosen = choose_keys(smap, args.max_per_subset, args.seed)
    print("molecules in archive :", len(smap))
    for sub, keys in sorted(chosen.items(), key=lambda kv: -len(kv[1])):
        print(f"  selected {sub:24s} {len(keys):5d} of "
              f"{sum(1 for v in smap.values() if v == sub):5d}")

    pairs: list[dict] = []
    conformers: dict[str, dict] = {}
    per_subset: dict[str, dict[str, int]] = {}
    n_mol = 0
    n_skipped = 0
    n_charged = 0
    # Molecules whose SMILES is absent, so no charge could be read. The builder
    # chooses to COUNT rather than guess neutral: the number goes into the manifest
    # and the run summary, so a conformer run as neutral is never invisible. Measured
    # on SPICE 1.1.4: 0 of 19,238 molecules hit this path.
    n_missing_smiles = 0
    cap = args.max_conformers_per_molecule

    with h5py.File(args.archive, "r") as f:
        for sub in sorted(chosen):
            for key in chosen[sub]:
                n_mol += 1
                mol = read_molecule(f[key], key)
                if mol.n < 2:
                    n_skipped += 1
                    continue
                energies = mol.energies()
                keep = select_conformers(energies, cap)
                if len(keep) < 2:
                    n_skipped += 1
                    continue
                charge, spin = charge_and_multiplicity(
                    mol.smiles, mol.conformers[0].atomic_numbers,
                    on_missing="count")
                if mol.smiles is None or not str(mol.smiles).strip():
                    n_missing_smiles += 1
                if charge != 0:
                    n_charged += 1
                local = pairs_from_molecule(
                    mol.key, mol.subset, energies[keep],
                    max_pairs=args.max_pairs_per_molecule, rng=rng)
                if not local:
                    n_skipped += 1
                    continue
                d = per_subset.setdefault(mol.subset, {
                    "molecules": 0, "pairs": 0, "conformer_sets": 0,
                    "conformers": 0, "charged_molecules": 0})
                d["molecules"] += 1
                d["conformer_sets"] += 1
                if charge != 0:
                    d["charged_molecules"] += 1
                for p in local:
                    gi, gj = int(keep[p.idx_a]), int(keep[p.idx_b])
                    id_a = f"{mol.key}|{gi}"
                    id_b = f"{mol.key}|{gj}"
                    for cid, gi2 in ((id_a, gi), (id_b, gj)):
                        if cid in conformers:
                            continue
                        c = mol.conformers[gi2]
                        conformers[cid] = {
                            "id": cid,
                            "molecule": mol.key,
                            "subset": mol.subset,
                            "atomic_numbers": [int(z) for z in c.atomic_numbers],
                            # Angstrom, converted from the archive's bohr
                            "positions": [[round(float(v), 5) for v in row]
                                          for row in c.positions],
                            "energy_hartree": float(c.energy_hartree),
                            "charge": int(charge),
                            "spin": int(spin),
                        }
                        d["conformers"] += 1
                    pairs.append({
                        "molecule": mol.key,
                        "subset": mol.subset,
                        "is_bio": mol.subset in BIO_SUBSETS,
                        "id_a": id_a,
                        "id_b": id_b,
                        "qm_gap_kcal": round(float(p.qm_gap_kcal), 6),
                        "charge": int(charge),
                    })
                    d["pairs"] += 1

    manifest = {
        "seed": args.seed,
        "archive": str(args.archive),
        "units": {"positions": "angstrom", "source_positions": "bohr",
                  "bohr2ang": BOHR2ANG, "energy": "hartree",
                  "qm_gap": "kcal/mol = (E_B - E_A) * 627.5094740631"},
        "n_molecules_selected": n_mol,
        "n_molecules_with_pairs": sum(d["molecules"] for d in per_subset.values()),
        "n_molecules_skipped": n_skipped,
        "n_charged_molecules": n_charged,
        "n_missing_smiles": n_missing_smiles,
        "n_pairs": len(pairs),
        "n_conformers": len(conformers),
        "n_bio_pairs": sum(1 for p in pairs if p["is_bio"]),
        "max_pairs_per_molecule": args.max_pairs_per_molecule,
        "max_conformers_per_molecule": cap,
        "max_per_subset": args.max_per_subset,
        "per_subset": per_subset,
        "pairs": pairs,
    }
    (out_dir / "pairs_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    with open(out_dir / "conformers.jsonl", "w") as fh:
        for cid in sorted(conformers):
            fh.write(json.dumps(conformers[cid]) + "\n")

    print(f"\nmolecules read      : {n_mol}")
    print(f"molecules with pairs: {manifest['n_molecules_with_pairs']}")
    print(f"conformers exported : {len(conformers)}")
    print(f"decision pairs      : {len(pairs)} (bio {manifest['n_bio_pairs']})")
    print(f"charged molecules   : {n_charged}")
    print(f"molecules with no SMILES (charge unreadable, run neutral): "
          f"{n_missing_smiles}")
    for s, dd in sorted(per_subset.items(), key=lambda kv: -kv[1]["pairs"]):
        print(f"  {s:24s} mol={dd['molecules']:5d} pairs={dd['pairs']:6d} "
              f"conf={dd['conformers']:6d} charged={dd['charged_molecules']:4d}")
    if not pairs:
        print("ERROR: no decision pairs built", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
