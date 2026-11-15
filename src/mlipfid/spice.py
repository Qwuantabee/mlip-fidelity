"""Read the SPICE HDF5 archive into plain numpy structures.

SPICE 1.1.x layout, verified by direct inspection of the downloaded archive
(19,238 molecules, 1,110,165 conformers):

    /<molecule_key>/atomic_numbers      (n_atoms,)              int16
    /<molecule_key>/conformations       (n_conf, n_atoms, 3)    float32, BOHR
    /<molecule_key>/dft_total_energy    (n_conf,)               float64, Hartree
    /<molecule_key>/dft_total_gradient  (n_conf, n_atoms, 3)    float32, Hartree/bohr
    /<molecule_key>/subset              (1,)                    str (verbose)
    /<molecule_key>/smiles              (1,)                    str

Two traps that this reader exists to avoid, both of which produced plausible but
wrong output on the first attempt:

1. Conformers are STACKED datasets, not one subgroup per conformer. A reader that
   looks for a subgroup per conformer finds zero and silently reports one
   conformer per molecule — i.e. no decision pairs at all.
2. Positions are in BOHR. Passing them to a calculator as Angstrom shrinks every
   molecule by a factor of 0.529 and changes every energy, with no error raised.

`subset` is stored with a verbose upstream name ("SPICE Dipeptides Single Points
Dataset v1.2"); it is slugged to a short tag so results can be grouped.

This module never guesses: `read_molecule` raises with the keys it actually saw
if a molecule has neither layout.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# 1 bohr = 0.52917721067 Angstrom (CODATA 2018). Pinned by a test because a
# wrong factor shifts every geometry and every energy without raising anything.
BOHR2ANG = 0.52917721067

_SUBSET_SLUGS = (
    ("dipeptide", "dipeptide"),
    ("solvated amino acid", "solvated_amino_acids"),
    ("amino acid", "amino_acids"),
    ("des370k", "des370k"),
    ("des monomer", "des_monomers"),
    ("pubchem", "pubchem"),
    ("ion pair", "ion_pairs"),
)


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
    smiles: str | None = None

    @property
    def n(self) -> int:
        return len(self.conformers)

    def energies(self) -> np.ndarray:
        return np.array([c.energy_hartree for c in self.conformers], dtype=float)


def _as_str(v) -> str:
    """Decode the scalar string stored in a (1,) object/VLEN dataset.

    Accepts a Dataset, an array, or a bare scalar. An h5py Dataset is NOT an
    ndarray, so without this branch `str(dataset)` returns
    '<HDF5 dataset "subset": ...>' and every subset silently becomes a bogus
    slug — which is exactly how the inventory reported one useless subset.
    """
    if isinstance(v, h5py.Dataset):
        v = v[()]
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0] if v.size else ""
        return _as_str(v)
    return str(v)


def subset_of(key: str) -> str:
    """Fallback for archives where the subset dataset is missing."""
    return key.split(":", 1)[0] if ":" in key else "spice"


def slug_subset(verbose: str) -> str:
    """'SPICE Dipeptides Single Points Dataset v1.2' -> 'dipeptide'."""
    s = verbose.lower()
    for needle, slug in _SUBSET_SLUGS:
        if needle in s:
            return slug
    s = re.sub(r"[^a-z0-9]+", "_", verbose.lower()).strip("_")
    return s or "spice"


def _is_conformer_group(g: h5py.Group) -> bool:
    return "dft_total_energy" in g


def _stacked_positions(g: h5py.Group) -> np.ndarray | None:
    for name in ("conformations", "conformers", "positions"):
        if name in g and isinstance(g[name], h5py.Dataset):
            return np.asarray(g[name], dtype=float)
    return None


def read_molecule(g: h5py.Group, key: str) -> Molecule:
    """Read one molecule, handling the stacked layout first (the real one)."""
    subset = subset_of(key)
    if "subset" in g:
        try:
            subset = slug_subset(_as_str(g["subset"]))
        except Exception:  # noqa: BLE001 - malformed subset must not lose the molecule
            pass
    smiles = None
    if "smiles" in g:
        try:
            smiles = _as_str(g["smiles"])
        except Exception:  # noqa: BLE001
            smiles = None

    z = np.asarray(g["atomic_numbers"]).astype(int).reshape(-1)
    pos = _stacked_positions(g)
    if pos is not None:
        if pos.ndim == 2:            # (n_atoms, 3) single conformer, no stack
            pos = pos[None, ...]
        if pos.ndim != 3 or pos.shape[1:] != (len(z), 3):
            raise ValueError(
                f"{key}: conformations shape {pos.shape} does not match "
                f"{len(z)} atoms; expected (n_conf, {len(z)}, 3)")
        pos = pos * BOHR2ANG       # BOHR -> Angstrom, see module docstring
        energies = np.asarray(g["dft_total_energy"], dtype=float).reshape(-1)
        if len(energies) != pos.shape[0]:
            raise ValueError(
                f"{key}: {pos.shape[0]} conformers but {len(energies)} energies")
        forces = None
        for fname in ("dft_total_gradient", "forces"):
            if fname in g:
                # gradient w.r.t. position in Hartree/bohr; force IS -gradient.
                # A force per BOHR becomes a force per ANGSTROM by DIVIDING by
                # BOHR2ANG (x1.8897), not multiplying: the old `* BOHR2ANG` was
                # wrong by 0.529^2 = 0.28. Nothing downstream reads forces today, so
                # no published number was affected - but a future consumer would
                # have been silently off by 3.6x.
                forces = -np.asarray(g[fname], dtype=float) / BOHR2ANG
                break
        confs = [
            Conformer(z, pos[i], float(energies[i]),
                      None if forces is None else forces[i])
            for i in range(pos.shape[0])
        ]
        return Molecule(key=key, subset=subset, conformers=confs, smiles=smiles)

    # Older/alternate layout: one subgroup per conformer.
    names = sorted(k for k, v in g.items()
                   if isinstance(v, h5py.Group) and _is_conformer_group(v))
    if names:
        confs = []
        for name in names:
            cg = g[name]
            energy = float(np.asarray(cg["dft_total_energy"]).reshape(-1)[0])
            cz = np.asarray(cg["atomic_numbers"]).astype(int).reshape(-1)
            cpos = np.asarray(cg["positions"], dtype=float).reshape(-1, 3)
            confs.append(Conformer(cz, cpos, energy, None))
        return Molecule(key=key, subset=subset, conformers=confs, smiles=smiles)

    raise ValueError(
        f"{key}: no conformations dataset and no conformer group with "
        f"dft_total_energy; children were {list(g.keys())[:8]}")


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


def iter_smiles(path: str | Path) -> dict[str, str]:
    """key -> SMILES, for the molecules that carry it (cheap metadata pass)."""
    out: dict[str, str] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            g = f[key]
            if "smiles" in g:
                try:
                    out[key] = _as_str(g["smiles"])
                except Exception:  # noqa: BLE001
                    continue
    return out


def inventory(path: str | Path, sample: int = 4000) -> dict:
    """Count molecules and conformers per subset, from a bounded sample.

    `sample=0` inspects every molecule. Counts from a sample are labelled as such
    by the caller so no figure can present them as the whole archive.
    """
    by_subset: dict[str, dict[str, int]] = {}
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        take = keys if (sample <= 0 or len(keys) <= sample) else keys[:sample]
        for key in take:
            g = f[key]
            if "dft_total_energy" in g:
                nconf = int(np.asarray(g["dft_total_energy"]).size)
            else:
                nconf = sum(1 for k, v in g.items()
                            if isinstance(v, h5py.Group) and _is_conformer_group(v))
            if nconf == 0 and _is_conformer_group(g):
                nconf = 1
            s = subset_of(key)
            if "subset" in g:
                try:
                    s = slug_subset(_as_str(g["subset"]))
                except Exception:  # noqa: BLE001
                    pass
            d = by_subset.setdefault(s, {"molecules": 0, "conformers": 0,
                                         "multi_conformer": 0})
            d["molecules"] += 1
            d["conformers"] += nconf
            if nconf > 1:
                d["multi_conformer"] += 1
    return {"n_total_molecules": len(keys), "n_sampled": len(take),
            "by_subset": by_subset}
