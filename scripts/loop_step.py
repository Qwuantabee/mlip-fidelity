#!/usr/bin/env python3
"""The mlip-fidelity loop's single decided-on-each-fire script.

Reads LOOP_STATE.json, works out what the next unfinished step is from what is
actually on disk, does that step if it is cheap, and writes the new state. The
model turn only handles judgement calls (choosing subsets, writing prose); all the
mechanical "is the file there yet" logic lives here so a firing turn cannot
drift or double-do work.

Run: python3 scripts/loop_step.py [--advance-on-complete]

Exit code 0 always; the JSON on stdout is the state the turn should act on.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "LOOP_STATE.json"
DATA = Path("/home/ubuntu/biotech-portfolio/_data")
SPICE = DATA / "SPICE-1.1.4.hdf5"
PY = "/home/ubuntu/.openclaw/workspace/venv/bin/python"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"step": 1, "notes": []}


def save(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def unit_active(name: str) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", name],
                       capture_output=True, text=True)
    return r.stdout.strip() == "active"


SWEEP_UNITS = ("sweep-mace", "sweep-orb", "sweep-fairchem", "sweep-all")


def running_sweeps() -> list[str]:
    """Sweep units currently alive.

    This exists because the loop fires every 15 minutes while a full sweep takes
    hours. Without a check that reads DISK (not the ledger), each fire sees step 4
    unfinished and launches another sweep: four duplicate runs were observed on
    2026-11-15, competing for one GPU and deleting each other's partial results. A
    fire that finds a live unit must not start another.
    """
    return [u for u in SWEEP_UNITS if unit_active(u)]


def spice_complete() -> bool:
    if not SPICE.exists():
        return False
    log = DATA / "par_fetch.log"
    if not log.exists():
        return False
    txt = log.read_text()
    return "OK " in txt and "bytes == Content-Length" in txt


def detect_step(st: dict) -> tuple[int, str]:
    """Ground truth wins over the recorded step: never trust a stale ledger."""
    if not spice_complete():
        return 1, "DOWNLOAD"
    if not (ROOT / "results" / "spice_inventory.json").exists():
        return 2, "INVENTORY"
    if not (ROOT / "results" / "pairs_manifest.json").exists():
        return 3, "PAIRS"
    raw = list((ROOT / "results").glob("raw_*.json"))
    if len(raw) < 1:
        return 4, "SWEEP"
    if not (ROOT / "results" / "summary.csv").exists():
        return 5, "SCORE"
    figs = list((ROOT / "figures").glob("*.png"))
    if len(figs) < 6:
        return 6, "FIGURES"
    if not (ROOT / "README.md").exists():
        return 7, "README"
    if not (ROOT / ".github" / "workflows" / "tests.yml").exists():
        return 8, "CI"
    return 9, "PUSH"


def main() -> int:
    st = load()
    step, name = detect_step(st)
    st["step"] = step
    st["step_name"] = name
    st["checked_at"] = now()

    out = {"step": step, "step_name": name, "spice_complete": spice_complete(),
           "download_active": unit_active("spice-par"), "notes": []}

    # Cheap, mechanical actions this script is allowed to take on its own.
    if name == "DOWNLOAD":
        pct = None
        log = DATA / "par_fetch.log"
        if log.exists():
            lines = [l for l in log.read_text().splitlines() if "%" in l]
            if lines:
                pct = lines[-1]
        out["download_progress"] = pct
        out["next"] = ("wait for the unit; it re-arms itself. If it died, restart "
                       "with: systemctl --user reset-failed spice-par && "
                       "systemd-run --user --collect --unit spice-par "
                       "--working-directory=/home/ubuntu/biotech-portfolio/_data "
                       "/home/ubuntu/biotech-portfolio/_data/fetch_par.sh")
    elif name == "INVENTORY":
        out["next"] = (f"{PY} scripts/spice_inventory.py "
                       "--archive /home/ubuntu/biotech-portfolio/_data/SPICE-1.1.4.hdf5")
    elif name == "PAIRS":
        out["next"] = (f"{PY} scripts/build_pairs.py "
                       "--archive /home/ubuntu/biotech-portfolio/_data/SPICE-1.1.4.hdf5")
    elif name == "SWEEP":
        live = running_sweeps()
        out["sweeps_active"] = live
        out["do_not_relaunch"] = bool(live)
        if live:
            out["next"] = (f"a sweep is ALREADY RUNNING ({', '.join(live)}); do NOT "
                           "start another. Check the raw_*.json files, then end "
                           "this fire.")
        else:
            out["next"] = (f"{PY} scripts/run_sweep.py --models mace_off_small "
                           "orb_orbmol_v1_conservative (extend the list as venvs "
                           "prove out)")
    elif name == "SCORE":
        out["next"] = f"{PY} scripts/score.py"
    elif name == "FIGURES":
        out["next"] = f"{PY} scripts/make_figures.py"
    elif name == "README":
        out["next"] = "write README.md from results/summary.csv (headline scatter first)"
    elif name == "CI":
        out["next"] = "write .github/workflows/tests.yml and push once"
    save(st)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
