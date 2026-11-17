"""No script may hard-code a path on the author's machine.

`scripts/loop_step.py` once printed commands under `/home/ubuntu/...`; it now
reads MLIP_DATA and MLIP_PYTHON instead. The README check in
test_readme_repro_commands.py only covers the shell steps the README runs, so
this test scans every Python script.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
FORBIDDEN = ("/home/", "/mnt/c")


def test_scripts_were_found():
    assert len(SCRIPTS) >= 10, SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_script_carries_no_author_machine_path(path):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        for bad in FORBIDDEN:
            assert bad not in line, f"{path.name}:{n} hard-codes {bad!r}: {line.strip()}"


def test_loop_step_paths_come_from_the_environment(monkeypatch, tmp_path):
    import importlib.util
    import sys

    def load():
        spec = importlib.util.spec_from_file_location(
            "loop_step_under_test", ROOT / "scripts" / "loop_step.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    monkeypatch.delenv("MLIP_DATA", raising=False)
    monkeypatch.delenv("MLIP_PYTHON", raising=False)
    m = load()
    assert m.DATA == ROOT / "data" / "_external"
    assert m.PY == sys.executable

    monkeypatch.setenv("MLIP_DATA", str(tmp_path))
    monkeypatch.setenv("MLIP_PYTHON", "/opt/py/bin/python")
    m = load()
    assert m.DATA == tmp_path
    assert m.SPICE == tmp_path / "SPICE-1.1.4.hdf5"
    assert m.PY == "/opt/py/bin/python"
