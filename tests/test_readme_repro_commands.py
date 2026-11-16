"""The README's reproduction block must be runnable as written.

Review R2 (2026-11-16) ran README step 4 verbatim: `python scripts/sweep_all.sh`
is a bash script handed to the Python interpreter, so it died with a SyntaxError
(exit 1); run as bash it then cd'd into the author's home directory. Nothing in the
suite read that block, so the broken command shipped. These tests parse the block
and check each command's interpreter, target file, syntax and portability. They do
not run the sweep itself (that needs a GPU and the model venvs).
"""
import re
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _repro_commands() -> list[list[str]]:
    text = (ROOT / "README.md").read_text()
    sec = text.split("## Reproducing it", 1)
    assert len(sec) == 2, "README lost its 'Reproducing it' section"
    m = re.search(r"```bash\n(.*?)```", sec[1], re.S)
    assert m, "no bash block in 'Reproducing it'"
    block = m.group(1).replace("\\\n", " ")
    cmds = []
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            cmds.append(shlex.split(line))
    return cmds


COMMANDS = _repro_commands()
SCRIPTS = [c for c in COMMANDS if c[0] in ("python", "bash")]


def test_the_block_was_parsed():
    assert len(SCRIPTS) >= 6, COMMANDS


@pytest.mark.parametrize("cmd", SCRIPTS, ids=[" ".join(c[:2]) for c in SCRIPTS])
def test_each_command_uses_the_interpreter_its_file_needs(cmd):
    interp, target = cmd[0], cmd[1]
    path = ROOT / target
    assert path.exists(), f"README runs {target}, which does not exist"
    want = {".py": "python", ".sh": "bash"}[path.suffix]
    assert interp == want, (
        f"README runs '{interp} {target}', but a {path.suffix} file needs {want}")


SHELL = sorted({ROOT / c[1] for c in SCRIPTS if c[1].endswith(".sh")})


@pytest.mark.parametrize("path", SHELL, ids=[p.name for p in SHELL])
def test_shell_steps_parse_and_carry_no_author_paths(path):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    body = path.read_text()
    assert "/home/" not in body, f"{path.name} hard-codes an author-machine path"
