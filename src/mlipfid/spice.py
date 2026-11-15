"""Read the SPICE HDF5 archive into plain numpy structures.

SPICE 1.1.x layout (as published, verified on the downloaded file):

    /<molecule_key>/<conformer_index>/{atomic_numbers, positions,
                                      dft_total_energy, dft_total_gradient,
                                      formation_energy, ...}

`molecule_key` carries the subset as its prefix (`dipeptide:AC-Ala`,
`solvated_amino_acids:...`, `DES_molecule:...`, `spice_...`), which is how we
separate the bimolecular slice (peptides, amino acids) from the small-molecule
one. The reader never assumes a fixed depth: it walks to the first group that
contains `dft_total_energy`, so a layout change upstream fails loudly with the
path it saw rather than silently returning zero molecules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class Conformer:
    atomic_numbers: np.ndarray   # (n_atoms,) int
    positions: np.ndarray        # (n_atoms, 3) float, Angstrom
    energy_hartree: float
    forces: np.ndarray | None = None


@dataclass
class Molecule:
    key: str
    subset: str
    conformers: list[Conformer]

    @property
    def n(self) -> int:
        return len(self.conformers)

    def energies(self) -> np.ndarray:
        return np.array([c.energy_hartree for c in self.conformers], dtype=float)


def subset_of(key: str) -> str:
    """SPICE keys are 'subset:name' for the newer subsets and bare for others."""
    return key.split(":", 1)[0] if ":" in key else "spice"


def _is_conformer_group(g: h5py.Group) -> bool:
    return "dft_total_energy" in g


def read_molecule(g: h5py.Group, key: str) -> Molecule:
    names = sorted(k for k, v in g.items()
                   if isinstance(v, h5py.Group) and _is_conformer_group(v))
    if not names:
        # Some archives store the single conformer directly on the molecule group.
        if _is_conformer_group(g):
            names = ["."]
        else:
            raise ValueError(
                f"{key}: no conformer group with dft_total_energy; children were "
                f"{list(g.keys())[:8]}")
    confs: list[Conformer] = []
    for name in names:
        cg = g if name == "." else g[name]
        energy = float(np.asarray(cg["dft_total_energy"]).reshape(-1)[0])
        z = np.asarray(cg["atomic_numbers"]).astype(int).reshape(-1)
        pos = np.asarray(cg["positions"], dtype=float).reshape(-1, 3)
        forces = None
        for fname in ("dft_total_gradient", "forces"):
            if fname in cg:
                forces = np.asarray(cg[fname], dtype=float).reshape(-1, 3)
                break
        confs.append(Conformer(z, pos, energy, forces))
    return Molecule(key=key, subset=subset_of(key), conformers=confs)


def iter_molecules(path: str | Path, keys: list[str] | None = None,
                   limit: int | None = None):
    """Yield Molecule objects; `keys` restricts to an explicit selection."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing archive {path}")
    n = 0
    with h5py.File(path, "r") as f:
        names = keys if keys is not None else list(f.keys())
        for key in names:
            if key not in f:
                continue
            yield read_molecule(f[key], key)
            n += 1
            if limit is not None and n >= limit:
                return


def inventory(path: str | Path, sample: int = 4000) -> dict:
    """Count molecules and conformers per subset, from a bounded sample.

    The archive has ~1M conformers; counting all of them costs minutes of
    metadata reads. A fixed sample with the sampled fraction reported alongside
    is enough to choose subsets, and the note says so on the figure.
    """
    by_subset: dict[str, dict[str, int]] = {}
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        take = keys if len(keys) <= sample else keys[:sample]
        for key in take:
            g = f[key]
            nconf = sum(1 for k, v in g.items()
                        if isinstance(v, h5py.Group) and _is_conformer_group(v))
            if nconf == 0 and _is_conformer_group(g):
                nconf = 1
            s = subset_of(key)
            d = by_subset.setdefault(s, {"molecules": 0, "conformers": 0,
                                         "multi_conformer": 0})
            d["molecules"] += 1
            d["conformers"] += nconf
            if nconf > 1:
                d["multi_conformer"] += 1
    return {"n_total_molecules": len(keys), "n_sampled": len(take),
            "by_subset": by_subset}
