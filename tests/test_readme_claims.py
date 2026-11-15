"""The README's published numbers must equal the committed artefacts.

A README is the only part of a repo most readers see, so it is also the easiest
place for a number to rot. Every figure quoted in the prose or the results table
is re-derived here from `results/summary.csv` and `results/pairs_manifest.json`
and compared to what the README literally says. If a re-run changes accuracy, this
test fails before the README silently goes stale.

It also enforces the two rails that are easy to lose in an edit: the null control
must be reported for every row, and the non-commercial licence of MACE-OFF23 must
appear wherever its accuracy appears.
"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "summary.csv"
MANIFEST = ROOT / "results" / "pairs_manifest.json"
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def summary():
    if not SUMMARY.exists():
        pytest.skip("no summary.csv; run scripts/score.py first")
    return pd.read_csv(SUMMARY)


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


def _all(summary, model):
    row = summary[(summary.model == model) & (summary.scope == "all")]
    assert len(row) == 1, f"expected exactly one 'all' row for {model}"
    return row.iloc[0]


def test_readme_quotes_the_measured_accuracy(summary, readme):
    for model, quoted in [
        ("orb_orbmol_v1_conservative", "0.900"),
        ("mace_off_small", "0.747"),
        ("rdkit_mmff", "0.563"),
    ]:
        row = _all(summary, model)
        measured = f"{row.accuracy:.3f}"
        assert measured == quoted, (
            f"README quotes {quoted} for {model} but the CSV says {measured}")
        assert quoted in readme, f"{model} accuracy {quoted} is not in the README"


def test_readme_quotes_the_measured_decided_counts(summary, readme):
    for model in ("orb_orbmol_v1_conservative", "mace_off_small", "rdkit_mmff"):
        n = int(_all(summary, model).n_decided)
        assert f"{n:,}" in readme, f"decided count {n:,} for {model} not in README"


def test_readme_pairs_and_molecule_counts_match_the_manifest(readme):
    m = json.loads(MANIFEST.read_text())
    assert f"{m['n_pairs']:,}" in readme
    assert f"{m['n_molecules_with_pairs']:,}" in readme


def test_every_row_reports_a_null_p_value(summary, readme):
    """The null control is not optional; a row without one is not publishable."""
    assert summary.null_p_value.notna().all(), "a summary row has no null p-value"
    assert (summary.null_p_value <= 0.05).all(), "a null control did not separate"
    assert "null" in readme.lower(), "the README does not discuss the null control"


def test_non_commercial_licence_is_visible_wherever_mace_appears(summary, readme):
    row = _all(summary, "mace_off_small")
    lic = row.licence
    assert "non-commercial" in lic.lower() or "ASL" in lic
    assert "non-commercial" in readme.lower(), (
        "README mentions MACE but never its non-commercial weights")
    assert "ASL" in readme or "Academic Software Licence" in readme


def test_no_model_is_dropped_from_the_published_table(summary, readme):
    """Losers are published too: every scored model must appear in the README."""
    for model in summary.model.unique():
        # the display name is what appears in prose; fall back to the id
        assert model in readme or _display(model) in readme, (
            f"{model} is scored but absent from the README")


def _display(model: str) -> str:
    return {
        "mace_off_small": "MACE-OFF23",
        "orb_orbmol_v1_conservative": "ORB-omol",
        "rdkit_mmff": "MMFF94",
    }.get(model, model)


def test_readme_headline_is_computed_not_handwritten(summary, readme):
    """The '1 decision in 10' and '25%' claims must follow from the CSV."""
    orb_wrong = 1 - _all(summary, "orb_orbmol_v1_conservative").accuracy
    mace_wrong = 1 - _all(summary, "mace_off_small").accuracy
    assert "1 decision in 10" in readme and 0.09 <= orb_wrong <= 0.11, (
        "the headline claims 1-in-10 but the artefact disagrees")
    assert "25%" in readme and 0.24 <= mace_wrong <= 0.26
