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
    charge, spin = charge_and_multiplicity("[O-]C=O", [6, 8, 8, 1])
    assert charge == -1
    # Formate has 24 electrons (odd count + the -1 charge): an even count, so the
    # electron-count rule gives a singlet. Asserting a doublet here would have
    # encoded a wrong expectation, not found a bug.
    assert spin == 1


def test_charge_parsing_works_without_rdkit():
    """The CI runner has no RDKit, and that is how the original bug surfaced.

    With RDKit absent, the old code fell into a bare `except: charge = 0` and ran
    every ion as a neutral - a finite, wrong electronic state rather than an
    error. This test removes RDKit from the import path and requires the charges
    to survive.
    """
    import builtins
    import importlib
    real_import = builtins.__import__

    def no_rdkit(name, *a, **k):
        if name.startswith("rdkit"):
            raise ImportError("simulated: no rdkit")
        return real_import(name, *a, **k)

    builtins.__import__ = no_rdkit
    try:
        import mlipfid.select as sel
        importlib.reload(sel)
        assert sel.charge_and_multiplicity("[NH4+]", [7, 1, 1, 1, 1])[0] == 1
        assert sel.charge_and_multiplicity("[O-]C=O", [6, 8, 8, 1])[0] == -1
        assert sel.charge_and_multiplicity("[Fe+2]", [26])[0] == 2
        assert sel.charge_and_multiplicity("C", [6, 1, 1, 1, 1])[0] == 0
    finally:
        builtins.__import__ = real_import
        importlib.reload(sel)


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


def test_missing_smiles_raises_instead_of_guessing_neutral():
    """No SMILES means no charge is readable, so it must not default to neutral.

    The old behaviour returned (0, 1) for '', None and whitespace. That is the same
    silent-neutral class of defect that ran 74 charged Folmsbee compounds as neutral
    and cost ORB 0.127 of accuracy on those pairs (0.668 -> 0.795). Measured on SPICE
    1.1.4: 19,238 molecules carry a SMILES and 0 are blank, so raising here cannot
    move any published number - and if a future archive has blanks, the build stops
    instead of quietly scoring ions as neutrals.

    A caller that has decided how to handle the gap must say so explicitly with
    on_missing="count", and the builder records that count in its manifest.
    """
    import pytest

    from mlipfid.select import MissingSmilesError

    for bad in ("", "   ", None):
        with pytest.raises(MissingSmilesError):
            charge_and_multiplicity(bad, [7, 1, 1, 1])

    # the explicit opt-out still returns a neutral, and only when asked for
    assert charge_and_multiplicity(None, [7, 1, 1, 1], on_missing="count") == (0, 1)

    # an UNPARSEABLE but non-empty SMILES is a different case: its bracket text
    # still carries the charge (F9), so it must not raise.
    assert charge_and_multiplicity("[NH4+]C((", [7, 1, 1, 1, 1, 6])[0] == 1
