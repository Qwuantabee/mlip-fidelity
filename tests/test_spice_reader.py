"""Tests for the SPICE reader against the archive's REAL on-disk layout.

Written after the first inventory returned "0 multi-conformer molecules" on a
1.1M-conformer archive. The cause was a guessed layout: the reader looked for one
subgroup per conformer and read positions as Angstrom. The archive actually
stores stacked datasets per molecule:

    /<molecule_key>/atomic_numbers        (n_atoms,)          int16
    /<molecule_key>/conformations         (n_conf, n_atoms, 3) float32, BOHR
    /<molecule_key>/dft_total_energy      (n_conf,)            float64, Hartree
    /<molecule_key>/subset                (1,)                str

So these tests build a miniature archive in that layout. They fail against the
old reader (which finds no conformer groups) and pin the two things that would
silently corrupt every number downstream: the conformer count and the unit.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlipfid.spice import BOHR2ANG, inventory, iter_molecules, read_molecule  # noqa: E402


def _stacked_archive(tmp_path: Path, n_conf: int = 5) -> Path:
    """One molecule, `n_conf` conformers, positions written in bohr like upstream."""
    p = tmp_path / "mini.hdf5"
    n_atoms = 3
    rng = np.random.default_rng(0)
    with h5py.File(p, "w") as f:
        g = f.create_group("molA")
        g.create_dataset("atomic_numbers", data=np.array([6, 1, 1], dtype="int16"))
        # 1.0 bohr along x is a deliberate, checkable distance
        pos_bohr = np.zeros((n_conf, n_atoms, 3), dtype="float32")
        pos_bohr[:, 1, 0] = 1.0
        pos_bohr[:, 2, 1] = 2.0
        pos_bohr += rng.normal(scale=1e-3, size=pos_bohr.shape).astype("float32")
        g.create_dataset("conformations", data=pos_bohr)
        g.create_dataset("dft_total_energy",
                         data=np.linspace(-40.0, -39.999, n_conf))
        g.create_dataset("subset", data=np.array(
            ["SPICE Dipeptides Single Points Dataset v1.2"], dtype=object))
        g.create_dataset("smiles", data=np.array(["[H]C([H])([H])"], dtype=object))
    return p


def test_stacked_layout_yields_every_conformer(tmp_path):
    """The old reader returned 1 conformer for a 50-conformer molecule."""
    mol = read_molecule(h5py.File(_stacked_archive(tmp_path), "r")["molA"], "molA")
    assert mol.n == 5


def test_positions_are_converted_from_bohr_to_angstrom(tmp_path):
    """If bohr were passed through as Angstrom every geometry is 1.89x too LARGE,
    which changes the energy of every model without any error being raised."""
    assert abs(BOHR2ANG - 0.52917721067) < 1e-9
    mol = read_molecule(h5py.File(_stacked_archive(tmp_path), "r")["molA"], "molA")
    # atom 2 sits 1.0 bohr from atom 0 on x; +-noise of 1e-3 bohr
    dx = mol.conformers[0].positions[1, 0] - mol.conformers[0].positions[0, 0]
    assert abs(dx - 1.0 * BOHR2ANG) < 0.01
    assert mol.conformers[0].positions[2, 1] == pytest.approx(2.0 * BOHR2ANG, abs=0.01)


def test_subset_is_slugged_from_the_verbose_upstream_string(tmp_path):
    mol = read_molecule(h5py.File(_stacked_archive(tmp_path), "r")["molA"], "molA")
    assert mol.subset == "dipeptide"


def test_energies_are_read_in_hartree_order(tmp_path):
    mol = read_molecule(h5py.File(_stacked_archive(tmp_path), "r")["molA"], "molA")
    e = mol.energies()
    assert len(e) == 5 and np.all(np.diff(e) > 0)


def test_inventory_counts_multi_conformer_molecules(tmp_path):
    inv = inventory(_stacked_archive(tmp_path), sample=0)
    assert inv["n_total_molecules"] == 1
    assert inv["by_subset"]["dipeptide"]["conformers"] == 5
    assert inv["by_subset"]["dipeptide"]["multi_conformer"] == 1


def test_iter_molecules_respects_limit_and_charge_spin_defaults(tmp_path):
    mols = list(iter_molecules(_stacked_archive(tmp_path), limit=1))
    assert len(mols) == 1
