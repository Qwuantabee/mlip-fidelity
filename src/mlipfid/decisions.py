"""Read SPICE, build decision pairs, and score model predictions against QM.

The unit of analysis is a DECISION PAIR: two conformers of the same molecule whose
reference quantum energies differ by less than a threshold, so that the question
"which of these is lower?" is a question a chemist actually asks. A model that
gets the ordering right has preserved the chemistry; a model whose energy MAE is
tiny but whose ordering is inverted has not.

Conventions (asserted in tests, not just documented):
  * delta_E is always E(B) - E(A), never the reverse.
  * Energies are Hartree in SPICE and are converted to kcal/mol with HARTREE2KCAL.
  * Pairs are (A, B) with A the lower-energy conformer by construction? NO - see
    build_pairs: we deliberately do NOT sort by energy, because a signed gap is
    what carries the information and sorting would hide sign errors.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HARTREE2KCAL = 627.5094740631
# A chemical decision is usually decided at ~1 kcal/mol: conformer preferences,
# tautomer ratios, binding-pose choices. Below this the ranking is a tie in
# practice; above it, being wrong has consequences.
DECISION_BAR_KCAL = 1.0


@dataclass(frozen=True)
class Pair:
    """One decision: does the model also say B is above/below A?"""

    molecule: str          # SPICE molecule key (e.g. "dipeptide:AC-Ala")
    subset: str            # SPICE subset the molecule came from
    idx_a: int             # conformer index inside the molecule
    idx_b: int
    qm_gap_kcal: float     # E_QM(B) - E_QM(A)

    @property
    def qm_sign(self) -> int:
        return int(np.sign(self.qm_gap_kcal))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used everywhere a decision accuracy is quoted, because a raw 17/20 looks
    identical to 850/1000 until you show the interval, and the whole point of
    this repo is not to overclaim from a small count.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pairs_from_molecule(
    molecule: str,
    subset: str,
    energies_hartree: np.ndarray,
    max_pairs: int = 40,
    rng: np.random.Generator | None = None,
) -> list[Pair]:
    """All near-degenerate conformer pairs of one molecule, deterministically."""
    energies_hartree = np.asarray(energies_hartree, dtype=float)
    n = len(energies_hartree)
    if n < 2:
        return []
    kcal = (energies_hartree - energies_hartree.min()) * HARTREE2KCAL
    out: list[Pair] = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = kcal[j] - kcal[i]
            # Both members must lie within the decision window of each other.
            # A pair spanning 40 kcal/mol is not a decision, it is a comparison
            # between two different species.
            if abs(gap) > DECISION_BAR_KCAL:
                continue
            out.append(Pair(molecule, subset, i, j, float(gap)))
    if max_pairs and len(out) > max_pairs:
        if rng is None:
            rng = np.random.default_rng(0)
        keep = rng.choice(len(out), size=max_pairs, replace=False)
        out = [out[k] for k in sorted(keep)]
    return out


def decision_scores(qm_gaps_kcal: np.ndarray, model_gaps_kcal: np.ndarray) -> dict:
    """Sign agreement and error statistics for one model on one pair set.

    'correct' is defined on the SIGN of the gap only when the QM gap is large
    enough to be a real decision; pairs closer than TIE_KCAL are reported as ties
    and excluded from accuracy, not silently counted as wins.
    """
    qm = np.asarray(qm_gaps_kcal, dtype=float)
    ml = np.asarray(model_gaps_kcal, dtype=float)
    if qm.shape != ml.shape:
        raise ValueError(f"shape mismatch: qm {qm.shape} vs model {ml.shape}")
    tie_kcal = 0.01
    decided = np.abs(qm) > tie_kcal
    n_decided = int(decided.sum())
    agree = int((np.sign(qm[decided]) == np.sign(ml[decided])).sum())
    err = np.abs(ml - qm) / DECISION_BAR_KCAL   # error in units of the decision bar
    lo, hi = wilson_interval(agree, n_decided)
    return {
        "n_pairs": int(len(qm)),
        "n_decided": n_decided,
        "n_ties": int(len(qm) - n_decided),
        "n_agree": agree,
        "accuracy": (agree / n_decided) if n_decided else float("nan"),
        "accuracy_lo": lo,
        "accuracy_hi": hi,
        # signed bias matters: a model can be accurate yet systematically low
        "bias_kcal": float(np.mean(ml - qm)),
        "median_abs_err_kcal": float(np.median(np.abs(ml - qm))),
        "p90_abs_err_kcal": float(np.percentile(np.abs(ml - qm), 90)),
        "median_err_in_bars": float(np.median(err)),
        "n_beyond_bar": int((np.abs(ml - qm) > DECISION_BAR_KCAL).sum()),
    }


def null_control(model_gaps: np.ndarray, qm_gaps: np.ndarray,
                 n_shuffle: int = 200, seed: int = 0) -> dict:
    """Shuffle which model gap is paired with which QM gap.

    If a model's accuracy survives shuffling, the pipeline is leaking the answer
    (or the pairs share a common mean) and the run must be thrown away.
    """
    rng = np.random.default_rng(seed)
    qm = np.asarray(qm_gaps, dtype=float)
    ml = np.asarray(model_gaps, dtype=float)
    tie_kcal = 0.01
    decided = np.abs(qm) > tie_kcal
    qm_d = qm[decided]
    ml_d = ml[decided]
    if len(qm_d) < 2:
        return {"n_shuffle": 0, "mean_accuracy": float("nan"),
                "p_value": float("nan")}
    real = float((np.sign(qm_d) == np.sign(ml_d)).mean())
    accs = []
    for _ in range(n_shuffle):
        perm = rng.permutation(len(ml_d))
        accs.append(float((np.sign(qm_d) == np.sign(ml_d[perm])).mean()))
    accs = np.asarray(accs)
    # one-sided: how often does a shuffle do at least as well as the real pairing
    p = float((np.sum(accs >= real) + 1) / (len(accs) + 1))
    return {"n_shuffle": n_shuffle, "real_accuracy": real,
            "mean_accuracy": float(accs.mean()),
            "sd_accuracy": float(accs.std()), "p_value": p}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
