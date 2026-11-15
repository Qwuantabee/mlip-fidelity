"""Tests for conformer selection and charge handling.

Both behaviours here were chosen by measurement, so both are pinned: a change that
reverts to random subsampling or to charge=0 for ions must fail loudly.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlipfid.select import (  # noqa: E402
    SELECT_WINDOW_KCAL, charge_and_multiplicity, max_pairs_for, select_conformers)


def test_selection_is_deterministic():
    rng = np.random.default_rng(0)
    e = np.sort(rng.uniform(-40.0, -39.98, 49))
    a = select_conformers(e, 12)
    b = select_conformers(e, 12)
    assert np.array_equal(a, b)


def test_selection_keeps_the_global_minimum():
    rng = np.random.default_rng(1)
    e = np.sort(rng.uniform(-40.0, -39.98, 49))
    idx = select_conformers(e, 12)
    assert 0 in idx, "the lowest conformer must always be kept"


def test_selection_excludes_high_energy_conformers():
    """A 30 kcal/mol strained conformer is not part of a conformational decision."""
    e = np.array([0.0, 0.001, 0.002, 0.05])   # last is ~31 kcal/mol up
    idx = set(select_conformers(e, 3).tolist())
    assert 3 not in idx


def test_selection_beats_random_sampling_on_pair_yield():
    """Pins the measured reason for energy-stratified selection (120 vs 784 pairs)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from mlipfid.decisions import pairs_from_molecule

    rng = np.random.default_rng(2)
    total_sel = total_rand = 0
    for _ in range(30):
        e = np.sort(rng.uniform(-40.0, -39.93, 49))
        sel = select_conformers(e, 12)
        rnd = np.sort(rng.choice(len(e), 12, replace=False))
        total_sel += len(pairs_from_molecule("m", "s", e[sel], max_pairs=0))
        total_rand += len(pairs_from_molecule("m", "s", e[rnd], max_pairs=0))
    assert total_sel > total_rand


def test_window_is_a_chemical_choice_not_a_number_typed_twice():
    assert 0 < SELECT_WINDOW_KCAL <= 20


def test_max_pairs_matches_choose_two():
    assert max_pairs_for(12) == 66


def test_ion_charge_comes_from_smiles():
    charge, spin = charge_and_multiplicity("[NH4+]", [7, 1, 1, 1, 1])
    assert charge == 1


def test_anion_charge_is_negative():
    charge, _ = charge_and_multiplicity("[O-]C=O", [6, 8, 8, 1])
    assert charge == -1


def test_neutral_singlet_default():
    charge, spin = charge_and_multiplicity("C", [6, 1, 1, 1, 1])
    assert (charge, spin) == (0, 1)


def test_unparseable_smiles_falls_back_instead_of_dropping_the_molecule():
    charge, spin = charge_and_multiplicity("not_a_smiles(((", [6])
    assert charge == 0 and spin == 1


def test_odd_electron_count_raises_multiplicity():
    """A radical must not be run as a closed-shell singlet.

    CH3 radical: 6 + 3 = 9 electrons, neutral -> odd -> doublet. (CH3 CATION is
    8 electrons and IS a singlet, which is the distinction this pins.)
    """
    charge, spin = charge_and_multiplicity("[CH3]", [6, 1, 1, 1])
    assert (charge, spin) == (0, 2)
    charge, spin = charge_and_multiplicity("[CH3+]", [6, 1, 1, 1])
    assert (charge, spin) == (1, 1)
