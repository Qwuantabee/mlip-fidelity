"""The README's secondary numbers must equal the committed artefacts too.

tests/test_readme_claims.py guards the headline accuracy cell and the decided
counts. A mutation test (change one README number, re-run the suite) showed that
the interval bounds, tie counts, bio-slice accuracies, constant-guess baseline
and every held-out figure other than the bold accuracy could be edited without
any test failing. This file recomputes those from the decision CSVs with its own
arithmetic (it imports nothing from src/ or scripts/) and requires the README to
print the result.

It also pins the comparison the held-out section rests on: accuracy at equal gap
size, SPICE against Folmsbee, and the gap-reweighted Folmsbee accuracy.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODELS = {
    "orb": "orb_orbmol_v1_conservative",
    "mace": "mace_off_small",
    "mmff": "rdkit_mmff",
}
TIE_KCAL = 0.01
GAP_BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
MINUS = "−"      # the README prints a typographic minus


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def _signed(x: float, nd: int = 3) -> str:
    s = f"{abs(x):.{nd}f}"
    return ("+" if x >= 0 else MINUS) + s


@pytest.fixture(scope="module")
def readme() -> str:
    return " ".join((ROOT / "README.md").read_text().split())


@pytest.fixture(scope="module")
def spice() -> dict[str, pd.DataFrame]:
    out = {}
    for k, m in MODELS.items():
        d = pd.read_csv(RES / f"decisions_{m}_all.csv")
        d["ok"] = np.sign(d.qm_gap_kcal) == np.sign(d.model_gap_kcal)
        d["absgap"] = d.qm_gap_kcal.abs()
        out[k] = d
    return out


@pytest.fixture(scope="module")
def folmsbee() -> dict[str, pd.DataFrame]:
    out = {}
    for k, m in MODELS.items():
        d = pd.read_csv(RES / f"heldout_folmsbee_decisions_{m}.csv")
        d["ok"] = np.sign(d.ref_kcal) == np.sign(d.model_kcal)
        d["absgap"] = d.ref_kcal.abs()
        out[k] = d
    return out


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((RES / "pairs_manifest.json").read_text())


def _decided(d: pd.DataFrame) -> pd.DataFrame:
    return d[d.absgap > TIE_KCAL]


# ---------------------------------------------------------------- in-sample --

def test_interval_bounds_in_the_results_table(readme, spice):
    for k, d in spice.items():
        dec = _decided(d)
        lo, hi = _wilson(int(dec.ok.sum()), len(dec))
        cell = rf"{dec.ok.mean():.3f}\*{{0,2}} \({lo:.3f}–{hi:.3f}\)"
        assert re.search(cell, readme), f"{k}: table cell should read {cell}"


def test_lead_sentence_percentages(readme, spice):
    for k, d in spice.items():
        dec = _decided(d)
        assert f"{100 * dec.ok.mean():.1f}%" in readme
    dec = _decided(spice["orb"])
    assert f"{100 * (dec.qm_gap_kcal > 0).mean():.1f}%" in readme


def test_tie_and_unscorable_counts_are_reported(readme, spice, manifest):
    n_all = manifest["n_pairs"]
    ties = {k: len(d) - len(_decided(d)) for k, d in spice.items()}
    assert (f"{ties['orb']} pairs for ORB-omol and MACE-OFF23, {ties['mmff']} for MMFF94"
            in readme)
    assert ties["orb"] == ties["mace"]
    unscorable = {k: n_all - len(d) for k, d in spice.items()}
    assert (f"{unscorable['orb']} for ORB-omol, {unscorable['mace']} for MACE-OFF23, "
            f"{unscorable['mmff']} for MMFF94") in readme
    zero = {k: int((_decided(d).model_gap_kcal == 0).sum()) for k, d in spice.items()}
    assert f"{zero['orb']} ORB-omol pairs, {zero['mace']} MACE-OFF23" in readme
    assert zero["mmff"] == 0


def test_wrong_ordering_count_and_reversal_rate(readme, spice):
    dec = _decided(spice["orb"])
    assert (f"{len(dec) - int(dec.ok.sum()):,} wrong orderings out of {len(dec):,}"
            in readme)
    mace = _decided(spice["mace"])
    assert f"reverses {100 * (1 - mace.ok.mean()):.0f}%" in readme


def test_bio_slice_accuracies(readme):
    want = {}
    for k, m in MODELS.items():
        d = pd.read_csv(RES / f"decisions_{m}_bio.csv")
        d = d[d.qm_gap_kcal.abs() > TIE_KCAL]
        want[k] = (np.sign(d.qm_gap_kcal) == np.sign(d.model_gap_kcal)).mean()
    assert f"**{want['orb']:.3f}**, MACE" in readme
    assert f"**{want['mace']:.3f}**, MMFF94" in readme
    assert f"→ {want['mmff']:.3f}. See" in readme


def test_constant_guess_baseline(readme, spice):
    for k, d in spice.items():
        dec = _decided(d)
        const = (dec.qm_gap_kcal > 0).to_numpy()
        ok = dec.ok.to_numpy()
        assert f"{const.mean():.3f}" == "0.534"
        assert f"**{_signed(ok.mean() - const.mean())}**" in readme, k
        if k == "mmff":
            assert (f"{int((ok & ~const).sum()):,} pairs and loses on "
                    f"{int((~ok & const).sum()):,}") in readme
    assert "scores 0.534**" in readme


def test_multi_component_and_charged_counts(readme, spice, manifest):
    pairs = manifest["pairs"]
    multi = sum(p["subset"] in ("des370k", "ion_pairs") for p in pairs)
    assert (f"{multi:,} of {len(pairs):,} pairs ({100 * multi / len(pairs):.1f}%)"
            in readme)
    charge = {p["id_a"] + ">" + p["id_b"]: p["charge"] for p in pairs}
    dec = _decided(spice["mace"]).copy()
    dec["charge"] = (dec.id_a + ">" + dec.id_b).map(charge)
    ch, ne = dec[dec.charge != 0], dec[dec.charge == 0]
    assert f"{len(ch):,} of its {len(dec):,} decided pairs" in readme
    assert f"**{ch.ok.mean():.3f} there against {ne.ok.mean():.3f}" in readme
    assert f"{ne.ok.mean():.3f} on {len(ne):,} pairs" in readme


# ----------------------------------------------------------------- held out --

def test_heldout_ties_excluded_column_and_like_for_like_shifts(readme, spice, folmsbee):
    shifts = []
    for k in MODELS:
        f = _decided(folmsbee[k])
        assert f"{f.ok.mean():.3f} ({len(f):,})" in readme, k
        shifts.append(_signed(f.ok.mean() - _decided(spice[k]).ok.mean()))
    assert " / ".join(shifts) in readme
    n_ties = {k: len(folmsbee[k]) - len(_decided(folmsbee[k])) for k in MODELS}
    assert (f"{n_ties['orb']} ties excluded for the ML models, {n_ties['mmff']} for MMFF94"
            in readme)


def test_heldout_raw_shift_columns(readme, spice, folmsbee):
    """Folmsbee over every pair minus SPICE over decided pairs, as the table prints."""
    shift = {k: folmsbee[k].ok.mean() - _decided(spice[k]).ok.mean() for k in MODELS}
    for k in MODELS:
        assert f"| {_signed(shift[k])} |" in readme, k
    for k in ("orb", "mace"):
        assert f"| {_signed(shift[k] - shift['mmff'])} |" in readme, k


def _bin_acc(d: pd.DataFrame, lo: float, hi: float) -> tuple[float, int]:
    b = d[(d.absgap >= lo) & (d.absgap < hi)]
    return float(b.ok.mean()), len(b)


def test_gap_size_table(readme, spice, folmsbee):
    """Every cell of 'Accuracy at equal gap size', ties included as the script does."""
    for lo, hi in GAP_BINS:
        cells = []
        for k in MODELS:
            s, _ = _bin_acc(spice[k], lo, hi)
            f, _ = _bin_acc(folmsbee[k], lo, hi)
            cells.append(f"{s:.3f} → {f:.3f}")
        row = f"| {lo:g} – {hi if hi < 1 else '1.0'} | " + " | ".join(cells) + " |"
        assert row in readme, f"gap table row should read: {row}"


def test_every_model_scores_higher_on_folmsbee_in_every_gap_bin(spice, folmsbee):
    """The sentence the held-out conclusion rests on. If this stops being true the
    README's reading of the held-out test has to be rewritten, not just renumbered."""
    for k in MODELS:
        for lo, hi in GAP_BINS:
            s, _ = _bin_acc(spice[k], lo, hi)
            f, _ = _bin_acc(folmsbee[k], lo, hi)
            assert f > s, f"{k} [{lo},{hi}): Folmsbee {f:.3f} is not above SPICE {s:.3f}"


def test_small_gap_share_and_median_gap(readme, spice, folmsbee, manifest):
    _, n_f = _bin_acc(folmsbee["orb"], 0.0, 0.1)
    _, n_s = _bin_acc(spice["orb"], 0.0, 0.1)
    n_f_all, n_s_all = len(folmsbee["orb"]), manifest["n_pairs"]
    assert (f"{n_f:,} of Folmsbee's {n_f_all:,} pairs ({100 * n_f / n_f_all:.1f}%)"
            in readme)
    assert f"{n_s:,} of SPICE's {n_s_all:,} ({100 * n_s / n_s_all:.1f}%)" in readme
    med_f = folmsbee["orb"].absgap.median()
    med_s = spice["orb"].absgap.median()
    assert (f"median reference gap {med_f:.2f} kcal/mol against SPICE's {med_s:.2f}"
            in readme)


def _gap_matched(s: pd.DataFrame, f: pd.DataFrame) -> float:
    """Folmsbee accuracy with its gap bins reweighted to SPICE's, ties excluded."""
    s, f = _decided(s), _decided(f)
    total = 0.0
    for lo, hi in GAP_BINS:
        acc, _ = _bin_acc(f, lo, hi)
        _, n = _bin_acc(s, lo, hi)
        total += acc * n
    n_binned = sum(_bin_acc(s, lo, hi)[1] for lo, hi in GAP_BINS)
    assert n_binned == len(s), "a decided SPICE pair fell outside the gap bins"
    return total / n_binned


def test_gap_reweighted_folmsbee_accuracy(readme, spice, folmsbee):
    matched = {k: _gap_matched(spice[k], folmsbee[k]) for k in MODELS}
    delta = {k: matched[k] - _decided(spice[k]).ok.mean() for k in MODELS}
    assert (f"ORB-omol {matched['orb']:.3f}, MACE-OFF23 {matched['mace']:.3f} and "
            f"MMFF94 {matched['mmff']:.3f}") in readme
    assert " / ".join(_signed(delta[k]) for k in MODELS) in readme
    rel_orb = delta["orb"] - delta["mmff"]
    rel_mace = delta["mace"] - delta["mmff"]
    assert f"→ {_signed(rel_orb)}, MACE-OFF23" in readme
    assert f"→ {_signed(rel_mace)}, each relative" in readme
    margin = matched["orb"] - matched["mmff"]
    assert f"the Folmsbee margin is {100 * margin:.0f} points" in readme


def test_contamination_disclosure_precedes_the_results_table():
    """Two of three models trained on the evaluation set: that has to be read
    before the numbers, not after them."""
    text = (ROOT / "README.md").read_text()
    first_table = text.index("\n| model |")
    assert "in-sample" in text[:first_table]
    assert "trained on" in text[:first_table] or "trained\non" in text[:first_table]
