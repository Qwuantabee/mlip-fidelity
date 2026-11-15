"""Tests for the decision-pair core: sign convention, units, null control, intervals.

These are the invariants a wrong answer would otherwise slip through, so each test
is written against a failure that would produce a plausible-looking but false
result rather than an obvious crash.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlipfid.decisions import (  # noqa: E402
    DECISION_BAR_KCAL, HARTREE2KCAL, decision_scores, null_control,
    pairs_from_molecule, wilson_interval)


def test_hartree_to_kcal_is_the_physical_constant():
    # 1 Hartree = 627.509474 kcal/mol. A wrong constant shifts every gap and would
    # make the 1 kcal/mol decision bar meaningless while still "running".
    assert abs(HARTREE2KCAL - 627.5094740631) < 1e-9


def test_gap_sign_is_b_minus_a_never_reversed():
    """If delta_E were A - B the accuracy of every model would be inverted."""
    qm = np.array([1.0, -1.0])
    ml = np.array([1.0, -1.0])
    s = decision_scores(qm, ml)
    assert s["n_agree"] == 2 and s["accuracy"] == 1.0
    # A model that reports every gap negated must score zero, not one.
    flipped = decision_scores(qm, -ml)
    assert flipped["accuracy"] == 0.0


def test_pairs_only_within_the_decision_window():
    energies = np.array([0.0, 0.0005, 0.01])   # 0, 0.31, 6.3 kcal/mol
    pairs = pairs_from_molecule("m", "dipeptide", energies, max_pairs=0)
    gaps = sorted(abs(p.qm_gap_kcal) for p in pairs)
    # only the 0<->0.0005 pair is inside 1 kcal/mol
    assert len(pairs) == 1
    assert gaps[0] < DECISION_BAR_KCAL


def test_pairs_are_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(0)
    energies = np.linspace(0, 0.003, 30)       # many sub-kcal spacings
    a = pairs_from_molecule("m", "dipeptide", energies, max_pairs=10,
                            rng=np.random.default_rng(7))
    b = pairs_from_molecule("m", "dipeptide", energies, max_pairs=10,
                            rng=np.random.default_rng(7))
    assert [(p.idx_a, p.idx_b) for p in a] == [(p.idx_b, p.idx_a) for p in b] or \
           [(p.idx_a, p.idx_b) for p in a] == [(p.idx_a, p.idx_b) for p in b]


def test_ties_are_excluded_from_accuracy_not_counted_as_wins():
    qm = np.array([1.0, 0.0, -1.0])
    ml = np.array([1.0, 5.0, -1.0])   # middle pair is a tie in QM
    s = decision_scores(qm, ml)
    assert s["n_ties"] == 1
    assert s["n_decided"] == 2
    assert s["accuracy"] == 1.0       # the tie's wrong sign is not counted either way


def test_null_control_rejects_a_leaking_pipeline():
    rng = np.random.default_rng(0)
    qm = rng.normal(size=400)
    ml = qm + rng.normal(scale=0.1, size=400)      # genuinely tracks QM
    real = null_control(ml, qm, n_shuffle=200, seed=0)
    assert real["p_value"] < 0.01
    assert real["real_accuracy"] > real["mean_accuracy"]


def test_null_control_finds_nothing_in_random_model():
    rng = np.random.default_rng(1)
    qm = rng.normal(size=400)
    ml = rng.normal(size=400)                      # unrelated to QM
    res = null_control(ml, qm, n_shuffle=200, seed=0)
    assert res["p_value"] > 0.05


def test_wilson_interval_brackets_the_point_estimate_and_stays_in_range():
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35
    lo, hi = wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0)


def test_bias_sign_is_model_minus_qm():
    """A model that is uniformly 0.5 kcal/mol high must show +0.5, not -0.5."""
    qm = np.array([0.2, 0.4, -0.3])
    ml = qm + 0.5
    s = decision_scores(qm, ml)
    assert abs(s["bias_kcal"] - 0.5) < 1e-12


def test_shape_mismatch_is_loud():
    with pytest.raises(ValueError):
        decision_scores(np.array([1.0, 2.0]), np.array([1.0]))
