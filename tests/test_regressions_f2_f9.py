"""Regression guards for two fixes the suite used to stay green without.

F2 - MMFF inter-fragment terms. RDKit's MMFF/UFF builders default to
ignoreInterfragInteractions=True, which drops every non-bonded term between
disconnected fragments. For a dimer or an ion pair that dropped term is the only
thing that changes between conformers, so with the default 255 MMFF pairs had an
exactly zero gap and the ion_pairs subset scored 0.000. The end-to-end test uses a
single-fragment molecule and cannot see the flag; reverting it left 57 tests green
(review R2, F-I). Here a two-fragment system is scored at two separations through
the real classical path in scripts/run_model.py, and the energies must differ.

F9 - silent neutral charge fallback. RDKit's MolFromSmiles returns None for an
unparseable SMILES instead of raising, so an ion with a malformed string used to
be run as a neutral. The fix falls back to the bracket-charge text parser; the
existing fallback test uses a neutral string, so reverting the fix left the suite
green (review R2, mutant C3).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlipfid.select import charge_and_multiplicity  # noqa: E402


def _run_model():
    spec = importlib.util.spec_from_file_location(
        "run_model", ROOT / "scripts" / "run_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# One rigid water monomer (Angstrom); the dimer is two copies, the second shifted
# along x. Monomer geometry is identical at every separation, so with the
# inter-fragment terms dropped the two energies would be identical.
_WATER = np.array([[0.0, 0.0, 0.0],        # O
                   [0.9572, 0.0, 0.0],     # H
                   [-0.2400, 0.9266, 0.0]])  # H


def _dimer(sep: float, cid: str) -> dict:
    second = _WATER + np.array([sep, 0.0, 0.0])
    pos = np.vstack([_WATER, second])
    return {"id": cid, "atomic_numbers": [8, 1, 1, 8, 1, 1],
            "positions": pos.round(6).tolist(), "charge": 0, "spin": 1}


def test_mmff_sees_the_interaction_between_two_fragments():
    rm = _run_model()
    near, far = _dimer(2.9, "near"), _dimer(6.0, "far")
    e_near, e_far = rm.energies_classical("rdkit_mmff", [near, far])
    assert np.isfinite(e_near) and np.isfinite(e_far), "bond perception failed"
    gap_kcal = (e_far - e_near) * rm.KCAL_PER_EV
    # Moving one water from 2.9 to 6.0 A changes the electrostatic and vdW
    # interaction by kcal/mol; with ignoreInterfragInteractions=True the gap is
    # exactly 0.0 because only intra-fragment terms remain.
    assert abs(gap_kcal) > 0.1, (
        f"MMFF gap between the two dimer separations is {gap_kcal:.6f} kcal/mol: "
        "inter-fragment terms are being ignored (the F2 bug)")


def test_uff_sees_the_interaction_between_two_fragments():
    rm = _run_model()
    near, far = _dimer(2.9, "near"), _dimer(6.0, "far")
    e_near, e_far = rm.energies_classical("rdkit_uff", [near, far])
    assert np.isfinite(e_near) and np.isfinite(e_far)
    assert abs((e_far - e_near) * rm.KCAL_PER_EV) > 0.01


@pytest.mark.parametrize("smiles,z,want", [
    ("[NH4+]C((", [7, 1, 1, 1, 1, 6], 1),     # the reviewer's original probe
    ("[O-]C(=O", [8, 6, 8, 1], -1),
    ("[NH3+]CC(=O)[O-]C(((", [7, 1, 1, 1, 6, 1, 1, 6, 8, 8, 6], 0),
    ("[Ca+2].[Cl-].[Cl-]((", [20, 17, 17], 0),
    ("[Mg++](((", [12], 2),
])
def test_unparseable_ion_smiles_keeps_its_charge(smiles, z, want):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    # Precondition: RDKit really cannot parse it, so this exercises the fallback.
    assert Chem.MolFromSmiles(smiles) is None
    charge, spin = charge_and_multiplicity(smiles, z)
    assert charge == want, (
        f"{smiles!r}: charge {charge}, expected {want} - an unparseable ion SMILES "
        "is being run as a neutral (the F9 bug)")
    assert spin == (2 if (sum(z) - want) % 2 else 1)
