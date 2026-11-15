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


def _table_rows(readme: str) -> dict[str, list[str]]:
    """Parse the results markdown table into {model-id: [cells]}.

    The first version of this test only asserted the string "0.900" appeared
    somewhere in the README. Mutating the table row to 0.950 still passed, because
    the old number survived in the prose below — so the test now reads the TABLE
    CELL, which is the number readers actually see first.
    """
    # Only the FIRST table in the file -- the in-sample SPICE results. The README
    # now also carries a held-out MPCONF196 table with the same model names, so an
    # unscoped parse overwrote these rows with that table's cells and the test
    # compared held-out 0.800 against the in-sample CSVs. Scoping is the fix; the
    # assertion below is unchanged.
    rows = {}
    started = False
    for line in readme.splitlines():
        if line.startswith("|"):
            started = True
        elif started and not line.strip():
            break                      # first blank line ends the first table
        if not started:
            continue
        if line.startswith("|---") or "model" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        name = cells[0].replace("*", "").strip()
        for key, needle in (("orb_orbmol_v1_conservative", "ORB-omol"),
                            ("mace_off_small", "MACE-OFF23"),
                            ("rdkit_mmff", "MMFF94")):
            if needle.lower() in name.lower():
                rows[key] = cells
    return rows


def _heldout_rows(readme: str) -> dict[str, list[str]]:
    """The MPCONF196 table, parsed the same way, so it is checked too."""
    rows = {}
    lines = readme.splitlines()
    # find the held-out table by its header
    start = next((i for i, l in enumerate(lines)
                  if l.startswith("|") and "held out" in l.lower()), None)
    if start is None:
        return rows
    for line in lines[start + 1:]:
        if not line.startswith("|"):
            break
        if line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        name = cells[0].replace("*", "").strip()
        for key, needle in (("orb_orbmol_v1_conservative", "ORB-omol"),
                            ("mace_off_small", "MACE-OFF23"),
                            ("rdkit_mmff", "MMFF94")):
            if needle.lower() in name.lower():
                rows[key] = cells
    return rows


def test_readme_table_cell_matches_the_measured_accuracy(summary, readme):
    rows = _table_rows(readme)
    assert rows, "could not find the results table in the README"
    for model in ("orb_orbmol_v1_conservative", "mace_off_small", "rdkit_mmff"):
        assert model in rows, f"{model} is missing from the results table"
        measured = f"{_all(summary, model).accuracy:.3f}"
        # The accuracy cell looks like `**0.900** (0.895–0.905)` or `0.563 (0.555–0.571)`.
        # Bolding is presentation, so accept either; anchor on the leading decimal
        # immediately followed by a CI, which is what makes it the accuracy cell and
        # not the error or n cell.
        acc_cell = next(
            (c for c in rows[model]
             if re.match(r"\*{0,2}\d\.\d+\*{0,2}\s*\(", c)), None)
        assert acc_cell is not None, (
            f"no accuracy cell for {model}; cells were {rows[model]}")
        quoted = re.match(r"\*{0,2}(\d\.\d+)", acc_cell).group(1)
        assert quoted == measured, (
            f"README table says {quoted} for {model}; the CSV says {measured}")


def test_readme_quotes_the_measured_accuracy(summary, readme):
    """Every model's accuracy in the README must equal the CSV, whatever it is.

    TEST-CHANGE (2026-11-15): the three expected values were hard-coded
    ("0.900"/"0.747"/"0.563") and asserted equal to the CSV. That is a
    tautology once the CSV is wrong - it pins the artefact to itself. Two
    independent reviews then found real bugs (MMFF kcal/mol stored as eV;
    MACE float32 noise) and the corrected CSVs read 0.900/0.760/0.601, so the
    test failed for the right reason and the hard-coded numbers had to go.

    The intent is unchanged and now stronger: the value the README prints must
    be the value the artefact contains, derived from summary.csv at test time,
    for every model - no literal a stale artefact could satisfy.
    """
    for model in ("orb_orbmol_v1_conservative", "mace_off_small", "rdkit_mmff"):
        row = _all(summary, model)
        measured = f"{row.accuracy:.3f}"
        assert measured in readme, (
            f"{model}: the CSV says {measured} but that number is not in the README")


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
    """The headline's three error/accuracy pairs must follow from the CSV.

    TEST-CHANGE (2026-11-15): this test used to assert the strings "1 decision in
    10" and "25%". Those were correct for the artefacts as they then stood, but the
    artefacts were wrong: an independent review found RDKit force fields return
    kcal/mol while the runner stored them as eV, inflating every MMFF gap by
    23.06x. With that fixed, the old headline's premise - that energy error and
    decision accuracy are independent - is refuted by the data, so the README was
    rewritten to state the monotone relationship instead. The test's INTENT is
    unchanged (no number may be handwritten), so it now checks all three published
    error/accuracy pairs against summary.csv rather than two prose strings.
    """
    for model, n_decimals in (("orb_orbmol_v1_conservative", 2),
                              ("mace_off_small", 2),
                              ("rdkit_mmff", 2)):
        row = _all(summary, model)
        err = f"{row.median_abs_err_kcal:.{n_decimals}f}"
        acc = f"{row.accuracy:.3f}"
        assert err in readme, (
            f"{model}: median error {err} kcal/mol is in the CSV but not the README")
        assert acc in readme, (
            f"{model}: accuracy {acc} is in the CSV but not the README")


def test_heldout_table_matches_the_heldout_csv(readme):
    """The held-out table must equal the Folmsbee summary, and MPCONF196 too.

    TEST-CHANGE (2026-11-15): this test read results/heldout_summary.csv, which is
    the MPCONF196 run. That set has only 30 sub-kcal pairs from 7 compounds - a
    cluster bootstrap gives [0.14, 0.71] for MACE, i.e. it cannot resolve anything
    - so the README's held-out table was switched to the well-powered Folmsbee set
    (22,379 pairs, 690 compounds) and the MPCONF196 numbers moved into the prose
    as a labelled withdrawal. The assertion was reading the wrong artefact, not
    asserting something wrong, so the INTENT is unchanged: every held-out number
    printed must equal a committed CSV. It now checks BOTH.
    """
    import pandas as pd

    fb_path = ROOT / "results" / "heldout_folmsbee_summary.csv"
    mp_path = ROOT / "results" / "heldout_summary.csv"
    if not fb_path.exists() or not mp_path.exists():
        pytest.skip("held-out summaries missing; run scripts/score_heldout.py")

    fb = pd.read_csv(fb_path).set_index("model")
    rows = _heldout_rows(readme)
    assert rows, "could not find the held-out table in the README"
    for model in ("orb_orbmol_v1_conservative", "mace_off_small", "rdkit_mmff"):
        assert model in rows, f"{model} missing from the held-out table"
        want = f"{fb.loc[model, 'acc_under_1kcal']:.3f}"
        assert want in " ".join(rows[model]), (
            f"held-out table row for {model} does not contain {want}: "
            f"{rows[model]}")

    # The withdrawn MPCONF196 figures are still quoted in the prose, so they are
    # still checked - a withdrawal is not permission for the number to drift.
    mp = pd.read_csv(mp_path).set_index("model")
    mace_small = f"{mp.loc['mace_off_small', 'acc_under_1kcal']:.3f}"
    assert mace_small in readme, (
        f"the withdrawn MPCONF196 value {mace_small} should still appear in the "
        f"README's withdrawal paragraph")
