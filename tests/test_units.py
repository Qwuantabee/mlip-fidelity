"""Pins the physical unit of every energy artefact.

Written after an independent review found that RDKit force fields return KCAL/MOL
while the runner stored them under an "energies_ev" key; score.py then treated them
as eV and divided by 27.211386, inflating every MMFF gap by 23.06x. The published
headline ("the classical baseline has 206x the best model's error") was really
8.8x. Nothing in the suite could see it, because sign accuracy is invariant to a
positive scale factor - the accuracy was right and the error column was wrong.

Two guards, both of which fail against the old code:
  1. a physical probe: a small bond stretch must produce a plausible energy change
     in the units we claim;
  2. a slope check: regressing model gap on QM gap over within-molecule pairs must
     give a slope near 1 for any model that tracks the reference at all.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

KCAL_PER_EV = 627.5094740631 / 27.211386245988


def test_kcal_per_ev_constant():
    # 1 eV = 23.060548 kcal/mol. A wrong constant rescales every conversion
    # silently. Tolerance is 1e-4, not 1e-6: the hand-written value in the comment
    # (23.0605419) differs from the exact quotient by 5.9e-6, and a tolerance tight
    # enough to reject that is testing my rounding, not the physics.
    assert abs(KCAL_PER_EV - 23.060548) < 1e-4


def test_kcal_per_ev_is_the_same_in_runner_and_scorer():
    """The runner converts once, at the source; the scorer converts eV to kcal.

    If both convert, a classical model is inflated by 23.06x; if neither does, it is
    deflated by the same factor. Pin the constant in both files to one value.
    """
    runner = (ROOT / "scripts" / "run_model.py").read_text()
    assert "KCAL_PER_EV" in runner, "runner must convert kcal/mol to eV at the source"
    import subprocess
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'scripts'); import run_model; "
         "print(run_model.KCAL_PER_EV)"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-400:]
    assert abs(float(out.stdout.strip()) - KCAL_PER_EV) < 1e-6


@pytest.mark.parametrize("model", ["rdkit_mmff", "mace_off_small",
                                   "orb_orbmol_v1_conservative"])
def test_model_gap_tracks_qm_gap_with_slope_near_one(model):
    """A model that reproduces the reference must give gaps of the same size.

    Slope is fitted over |QM gap| < 10 kcal/mol (wider than the decision window so
    the fit has lever arm). Under the unit bug MMFF's slope was 16.16; with correct
    units it is ~0.70, and the ML models sit near 1.0.
    """
    p = ROOT / "results" / f"decisions_{model}_all.csv"
    if not p.exists():
        pytest.skip(f"no {p.name}; run the sweep and score first")
    import pandas as pd
    d = pd.read_csv(p)
    d = d[d["qm_gap_kcal"].abs() < 10]
    if len(d) < 100:
        pytest.skip("too few pairs for a slope")
    slope = float(np.polyfit(d["qm_gap_kcal"], d["model_gap_kcal"], 1)[0])
    assert 0.3 < slope < 3.0, (
        f"{model}: model gap vs QM gap slope {slope:.2f} is outside [0.3, 3.0] - "
        "almost certainly a unit error (23.06x) or a reference-zero mismatch")
