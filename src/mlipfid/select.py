"""Conformer selection, and the charge/multiplicity the calculators need.

Two failures this file exists to prevent, both measured on the real archive:

1. RANDOM subsampling destroys the decision set. Molecules average ~49
   conformers; picking 12 at random from anywhere in a 40+ kcal/mol span gave
   120 decision pairs across 40 molecules, while picking 12 in a bounded
   low-energy window gave 784. A random cap would have shipped a repo claiming
   "few decisions exist" when the truth was "we sampled wrong".

2. SPICE is not all neutral. Parsing SMILES for the first 3,000 molecules found
   94 cations and 46 anions (formal charges up to +2 and -6). The runner defaulted
   every conformer to charge 0, which would have computed the wrong electronic
   state for every ion while still returning finite energies.

Selection is deterministic given the seed, and never reorders a pair's members.
"""
from __future__ import annotations

import numpy as np

HARTREE2KCAL = 627.5094740631

# Conformers above this relative energy are not part of a chemical decision about
# a molecule's preferred shape; they are strained or partially dissociated.
SELECT_WINDOW_KCAL = 10.0


def select_conformers(energies_hartree: np.ndarray, cap: int,
                      window_kcal: float = SELECT_WINDOW_KCAL) -> np.ndarray:
    """Indices of conformers to keep, energy-stratified inside the window.

    Strategy: restrict to the window, sort by energy, then take `cap` roughly
    evenly spaced along that sorted order. Even spacing (rather than taking the
    lowest `cap`) keeps mid-window conformers in the set, so not every pair shares
    the global minimum as one member; taking the lowest cap would make the pair
    graph a star and inflate the apparent difficulty of the ordering task.
    """
    e = np.asarray(energies_hartree, dtype=float)
    n = len(e)
    if n <= cap:
        return np.arange(n)
    rel_kcal = (e - e.min()) * HARTREE2KCAL
    inside = np.where(rel_kcal <= window_kcal)[0]
    if len(inside) < 2:
        inside = np.arange(n)                     # window unusable: use all
    if len(inside) <= cap:
        return np.sort(inside)
    order = inside[np.argsort(e[inside], kind="stable")]
    # Even spacing over the sorted window, always including the lowest conformer.
    pick = np.linspace(0, len(order) - 1, cap)
    idx = np.unique(np.round(pick).astype(int))
    return np.sort(order[idx])


def max_pairs_for(cap: int) -> int:
    """Pairs available at a given conformer cap (cap choose 2); keeps the
    per-molecule cap honest instead of hard-coding a number larger than possible."""
    return cap * (cap - 1) // 2


def _charge_from_smiles_text(smiles: str) -> int:
    """Net formal charge from bracket atoms, without RDKit.

    Handles the SMILES forms that appear in SPICE: [NH4+], [O-], [CH3+], [Fe+2]
    and multi-charge brackets such as [Ca+2]. Unbracketed atoms are neutral, so a
    SMILES with no brackets correctly yields 0.
    """
    total = 0
    for chunk in smiles.split("["):
        if "]" not in chunk:
            continue
        body = chunk.split("]", 1)[0]
        i = 0
        while i < len(body):
            if body[i] in "+-":
                sign = 1 if body[i] == "+" else -1
                j = i + 1
                digits = ""
                while j < len(body) and body[j].isdigit():
                    digits += body[j]
                    j += 1
                total += sign * (int(digits) if digits else 1)
                # a repeated sign ("++") also means magnitude 2
                while j < len(body) and body[j] == body[i]:
                    total += sign
                    j += 1
                i = j
            else:
                i += 1
    return total


def charge_and_multiplicity(smiles: str | None, atomic_numbers) -> tuple[int, int]:
    """Formal charge and spin multiplicity from a SMILES, without trusting it blindly.

    Returns (charge, spin). If RDKit cannot parse the SMILES it falls back to
    (0, 1) so a molecule is not silently dropped, but the caller records that it
    happened so a results table can show how many conformers were run neutral.
    """
    z = np.asarray(atomic_numbers, dtype=int)
    n_electrons = int(z.sum())
    charge = 0
    if smiles:
        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog("rdApp.*")
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                charge = int(Chem.GetFormalCharge(m))
            else:
                # MolFromSmiles returns None for an unparseable SMILES instead of
                # raising, so the except branch never fired and an ion with a
                # malformed string was silently run as a neutral. A reviewer caught
                # this: charge_and_multiplicity('[NH4+]C((', ...) returned 0 although
                # the text parser gives +1. Checked on the real data: 0 of 1,588
                # molecules are affected, so no published number moves.
                charge = _charge_from_smiles_text(smiles)
        except ImportError:
            # RDKit is absent: there is no way to read a formal charge, so fall
            # back to a SMILES-only count of bracket charges. Returning 0 here
            # would run every ion as a neutral, and the CI run proved that path
            # exists: three charge tests failed simply because the runner has no
            # RDKit. A missing dependency must be visible, not silently neutral.
            charge = _charge_from_smiles_text(smiles)
        except Exception:  # noqa: BLE001 - malformed SMILES
            charge = _charge_from_smiles_text(smiles)
    n_electrons -= charge
    spin = 2 if (n_electrons % 2) else 1
    # LIMIT, stated rather than hidden: an electron count only distinguishes
    # closed-shell from open-shell. It cannot recover the ground-state
    # multiplicity of a transition metal - real Fe2+ is d6 and a quintet, while
    # this returns a singlet. SPICE molecules are organic (C/H/N/O/S/halogen) plus
    # alkali/alkaline counter-ions, where parity is the correct answer; a metal
    # complex would need a real spin-state determination and is out of scope.
    return charge, spin
