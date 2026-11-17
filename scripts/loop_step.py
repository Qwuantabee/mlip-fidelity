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
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "LOOP_STATE.json"
# Where the SPICE archive and its fetch script live, and which interpreter the
# printed commands use. Both come from the environment so no machine path is
# baked in: MLIP_DATA (default data/_external under the repo) and MLIP_PYTHON
# (default: the interpreter running this script).
DATA = Path(os.environ.get("MLIP_DATA") or ROOT / "data" / "_external")
SPICE = DATA / "SPICE-1.1.4.hdf5"
PY = os.environ.get("MLIP_PYTHON") or sys.executable


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



def orb_coverage() -> float:
    """Fraction of exported conformers the merged ORB artefact actually covers.

    Existence is not completeness: a merge over 10 of 39 chunks writes a valid
    raw_orb file holding a quarter of the conformers, which is exactly how the
    withdrawn 0.893 got published. Coverage is the only honest test.
    """
    raw = ROOT / "results" / "raw_orb_orbmol_v1_conservative.json"
    conf = ROOT / "results" / "conformers.jsonl"
    if not conf.exists():
        return 1.0
    n_conf = sum(1 for _ in open(conf))
    if not raw.exists():
        return 0.0
    try:
        n_ok = json.loads(raw.read_text())["n_ok"]
    except Exception:  # noqa: BLE001
        return 0.0
    return n_ok / max(n_conf, 1)


def is_shipped(st: dict) -> bool:
    """True once the ledger records the project as shipped.

    Without this the loop had no terminal state: detect_step() kept returning
    9 PUSH forever, so every fire re-ran the whole verification sweep and burned
    its 1800 s budget until cron killed it ('timed out, last phase
    model-call-started'). A long-horizon loop must know when to stop, or it
    becomes a job that fails on schedule.
    """
    status = str(st.get("status", "")).upper()
    return status.startswith("COMPLETE") or status.startswith("SHIPPED")


def detect_step(st: dict) -> tuple[int, str]:
    """Ground truth wins over the recorded step: never trust a stale ledger."""
    if is_shipped(st):
        return 99, "DONE"
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
    # A finished-looking repo is not finished while a model's sweep is unmerged.
    # This must be checked before the "all files exist" verdict: the first version
    # reported PUSH while ORB's re-run was still in flight.
    if orb_coverage() < 0.95:
        return 45, "FINALIZE_PENDING"
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

    if name == "DONE":
        # Exit immediately and cheaply: nothing to do, no model call needed. The
        # automation payload also disables the job when it sees this, so a shipped
        # project cannot keep firing.
        out["next"] = "none - project shipped; disable this automation"
        out["shipped"] = True
        save(st)
        print(json.dumps(out, indent=2))
        return 0

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
                       f"--working-directory={DATA} "
                       f"{DATA}/fetch_par.sh")
    elif name == "INVENTORY":
        out["next"] = (f"{PY} scripts/spice_inventory.py "
                       f"--archive {SPICE}")
    elif name == "PAIRS":
        out["next"] = (f"{PY} scripts/build_pairs.py "
                       f"--archive {SPICE}")
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
    elif name == "FINALIZE_PENDING":
        out["next"] = (f"{PY} scripts/finalize_orb.py   # merges ORB chunks and "
                       "refuses to publish unless the raw covers >=95% of the "
                       "conformers exported")
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
