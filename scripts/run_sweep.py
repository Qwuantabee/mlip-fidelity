#!/usr/bin/env python3
"""Score every installed model over the committed conformer set.

Design constraints that came from real failures:

* One subprocess per model, in that model's own venv (`envs/<name>`). Installing
  fairchem-core upgrades e3nn and silently breaks MACE checkpoint loading, so a
  shared environment is not an option.
* The conformer set is read from `results/conformers.jsonl`, which is committed.
  Every model therefore sees byte-identical geometry, and a model added later is
  comparable to one scored months earlier.
* A model that fails to build or crashes mid-way does not abort the sweep: its
  failure is recorded with the error text, because "model X would not run" is
  itself a result the README must be able to state.
* Nothing here interprets the chemistry. It writes raw energies; the decision
  scorer does the rest, so a scoring bug cannot hide inside the runner.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# model name -> venv directory under envs/ (each may pin conflicting deps)
MODELS = {
    "mace_off_small": "mace",
    "mace_off_medium": "mace",
    "mace_mp_small": "mace",
    "orb_orbmol_v1_conservative": "orb",
    "orb_orb_v3_conservative_omol": "orb",
    "orb_v2": "orb",
    "rdkit_mmff": None,         # shared venv: rdkit is not torch-dependent
    "rdkit_uff": None,
}


def acquire_run_lock(path: Path, blocking: bool = False):
    """Take an exclusive lock on `results/`, or return None if another run holds it.

    Mutual exclusion used to live only in scripts/sweep_all.sh. A stale
    /tmp/orb2.sh launcher called this script DIRECTLY with --chunk 400 while a
    --chunk 1500 sweep was mid-run; each wrote its own merged raw_<model>.json
    from a partial view, so the published orb file held 400 of 15232 conformers.
    The lock must belong to the process that writes results, not to a wrapper.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(fh.fileno(), flags)
    except OSError:
        fh.close()
        return None
    fh.write(str(__import__("os").getpid()))
    fh.flush()
    return fh


def chunk_bounds(n: int, chunk: int) -> list[tuple[int, int]]:
    """Half-open [lo, hi) slices that tile 0..n exactly once."""
    if n <= 0:
        return []
    step = chunk if chunk and chunk > 0 else n
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def chunk_path(out_dir: Path, model: str, lo: int, hi: int) -> Path:
    """Cache filename keyed by the SLICE, never by the chunk ordinal.

    Naming chunks by index (`chunk_<model>_005.json`) meant two sweeps with
    different --chunk sizes collided on the same slot: a 600-conformer chunk was
    resumed as if it were a 1500-conformer slice, producing 1200 duplicated and
    6232 missing conformers in a merged file that still reported "ok".
    """
    return out_dir / f"chunk_{model}_{lo:06d}_{hi:06d}.json"


def chunk_input_hash(model: str, recs: list[dict]) -> str:
    """sha256 of everything a model sees for one chunk, plus the model name.

    Per conformer: id, atomic_numbers, positions, charge, spin, serialised as
    canonical JSON (sorted keys, no whitespace). charge/spin default to 0/1, as
    run_model.py does. A chunk cached under the same ids but computed from
    different geometry (a review scaled one conformer by 1.15 and still got
    "(cached)" with a stale 1.887 eV instead of 14.858 eV) no longer matches.
    """
    body = {"model": model, "conformers": [
        {"id": r["id"], "atomic_numbers": r["atomic_numbers"],
         "positions": r["positions"], "charge": int(r.get("charge", 0)),
         "spin": int(r.get("spin", 1))} for r in recs]}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def chunk_older_than(path: Path, conformers_path: Path | None) -> bool:
    """True if the chunk file predates the conformers.jsonl it was computed from."""
    if conformers_path is None or not Path(conformers_path).exists():
        return False
    return path.stat().st_mtime < Path(conformers_path).stat().st_mtime


def load_cached_chunk(path: Path, want_ids: list[str], recs: list[dict] | None = None,
                      model: str | None = None,
                      conformers_path: Path | None = None) -> dict | None:
    """Return the cached chunk only if it holds exactly the requested slice.

    ids alone are not enough. When `recs` (the current slice) is given, the
    chunk must carry an input_sha256 equal to chunk_input_hash(model, recs) and
    the same model name; a chunk without a hash cannot prove what it was
    computed from and is recomputed. When `conformers_path` is given, a chunk
    older than that file is stale and recomputed. main() always passes both;
    callers that pass neither get the legacy ids-only check.
    """
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not saved.get("ok"):
        return None
    if list(saved.get("ids", [])) != list(want_ids):
        print(f"   ignoring stale cache {path.name}: ids are not this slice",
              flush=True)
        return None
    if len(saved.get("energies_ev", [])) != len(want_ids):
        print(f"   ignoring cache {path.name}: {len(saved.get('energies_ev', []))} "
              f"energies for {len(want_ids)} ids", flush=True)
        return None
    if chunk_older_than(path, conformers_path):
        print(f"   ignoring stale cache {path.name}: older than "
              f"{Path(conformers_path).name}", flush=True)
        return None
    if recs is not None:
        if model is not None and saved.get("model") != model:
            print(f"   ignoring cache {path.name}: model {saved.get('model')!r} "
                  f"is not {model!r}", flush=True)
            return None
        got = saved.get("input_sha256")
        if not got:
            print(f"   ignoring cache {path.name}: no input hash, cannot verify "
                  f"the geometry it was computed from", flush=True)
            return None
        if got != chunk_input_hash(model or saved.get("model", ""), recs):
            print(f"   ignoring stale cache {path.name}: inputs changed since it "
                  f"was computed (input hash mismatch)", flush=True)
            return None
    return saved


def chunk_failed_count(chunk: dict) -> int:
    """True number of un-scored conformers in one chunk.

    `failures` is truncated to 10 entries per chunk to keep payloads small, so its
    length is NOT the failure count. Summing those truncated lengths made a run
    with 319 NaN energies report n_failed=70. Prefer the chunk's own `n_failed`,
    and fall back to the list length only for old chunks without the field.
    """
    if "n_failed" in chunk and chunk["n_failed"] is not None:
        return int(chunk["n_failed"])
    return len(chunk.get("failures", []))


def check_merged_ids(ids: list[str]) -> None:
    """A merged result must score each conformer at most once."""
    if len(set(ids)) != len(ids):
        dupes = len(ids) - len(set(ids))
        raise ValueError(f"merged ids contain {dupes} duplicate conformer(s)")


def venv_python(venv: str | None) -> str:
    if venv is None:
        # sys.executable, not a hard-coded venv path: a fresh clone on another
        # machine has no /home/ubuntu, and a reviewer could not rebuild the
        # environment because of it.
        import sys as _sys
        return _sys.executable
    p = ROOT / "envs" / venv / "bin" / "python"
    if not p.exists():
        raise SystemExit(f"missing venv {p}; build it first")
    return str(p)


def run_one(model: str, job: Path, out_dir: Path, device: str,
            limit: int | None) -> dict:
    venv = MODELS.get(model)
    if model not in MODELS:
        return {"ok": False, "model": model, "error": "unknown model"}
    py = venv_python(venv)
    cmd = [py, str(ROOT / "scripts" / "run_model.py"),
           "--model", model, "--input", str(job), "--device", device]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
    dt = time.time() - t0
    raw = p.stdout.strip().splitlines()
    payload = None
    for line in reversed(raw):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        return {"ok": False, "model": model,
                "error": f"no JSON from runner (rc={p.returncode})",
                "stderr_tail": p.stderr[-1200:], "seconds": dt}
    payload["wall_seconds"] = dt
    payload["runner_rc"] = p.returncode
    # NOT raw_<model>.json: that name means "this model is fully swept", and a
    # single chunk written there made a crashed sweep look complete with ok: true
    # (a code reviewer demonstrated it). The merged file is written once, by
    # main(), after every chunk exists.
    payload["partial"] = True
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--conformers", default=str(ROOT / "results" / "conformers.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N conformers (0 = all)")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="conformers per subprocess; bounds peak memory (a single "
                         "15k-conformer job plus torch OOM-killed this box)")
    ap.add_argument("--lock", default=None,
                    help="lock file guarding results/ (default results/.sweep.lock)")
    args = ap.parse_args()

    models = args.models or list(MODELS)
    conf_path = Path(args.conformers)
    if not conf_path.exists():
        raise SystemExit(f"missing {conf_path}; run scripts/build_pairs.py first")
    recs = [json.loads(line) for line in conf_path.read_text().splitlines() if line.strip()]
    if args.limit:
        # A --limit smoke run scores a PREFIX of the conformers; the merged file it
        # writes covers only that prefix. Writing it under raw_<model>.json made a
        # 400-conformer smoke test silently replace a 15,232-conformer production
        # artefact (a reviewer demonstrated the clobber).
        recs = recs[:args.limit]
        print(f"SMOKE RUN: scoring {len(recs)} of the exported conformers; the "
              f"merged file will be written as raw_<model>.smoke.json and is NOT "
              f"a production artefact", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(args.lock) if args.lock else out_dir / ".sweep.lock"
    lock = acquire_run_lock(lock_path, blocking=False)
    if lock is None:
        print(f"another sweep holds {lock_path}; refusing to start a duplicate",
              flush=True)
        return 75
    print(f"{len(recs)} conformers, models: {models}, "
          f"chunks of {args.chunk}", flush=True)

    # Chunked execution. A single job file with 15k conformers plus torch inside the
    # runner exceeded this 19 GB machine and was OOM-killed (rc=137) after an
    # earlier identical run had succeeded, so the memory ceiling is not
    # reproducible at full size. Each chunk is a separate subprocess, merged after.
    bounds = chunk_bounds(len(recs), args.chunk)

    summary = {}
    for model in models:
        ids_all: list[str] = []
        energies_all: list[float] = []
        failures: list[dict] = []
        n_failed_all = 0
        model_meta: dict = {}
        wall = 0.0
        ok = True
        err = ""
        for ci, (lo, hi) in enumerate(bounds):
            # Persist each chunk the moment it finishes. An earlier full run was
            # OOM-killed at chunk 2 and lost ALL of chunk 1's work because results
            # were only merged at the end; a crash must cost one chunk, not the run.
            want_ids = [c["id"] for c in recs[lo:hi]]
            part_raw = chunk_path(out_dir, model, lo, hi)
            saved = load_cached_chunk(part_raw, want_ids, recs=recs[lo:hi],
                                      model=model, conformers_path=conf_path)
            if saved is not None:
                ids_all += saved["ids"]
                energies_all += saved["energies_ev"]
                failures += saved.get("failures", [])
                n_failed_all += chunk_failed_count(saved)
                model_meta = saved.get("meta", {})
                wall += saved.get("wall_seconds", 0.0)
                print(f"-- {model} chunk {ci+1}/{len(bounds)} (cached)", flush=True)
                continue
            part = out_dir / f"_job_{model}_{ci}.json"
            part.write_text(json.dumps({"conformers": recs[lo:hi]}))
            print(f"-- {model} chunk {ci+1}/{len(bounds)} "
                  f"({hi-lo} conformers) ...", flush=True)
            res = run_one(model, part, out_dir, args.device, None)
            part.unlink(missing_ok=True)
            if not res.get("ok"):
                ok = False
                err = res.get("error", "failed")
                print(f"   FAILED: {err[:200]}", flush=True)
                break
            if (list(res.get("ids", [])) != want_ids
                    or len(res.get("energies_ev", [])) != len(want_ids)):
                ok = False
                err = (f"runner returned {len(res.get('ids', []))} ids / "
                       f"{len(res.get('energies_ev', []))} energies for a "
                       f"{len(want_ids)}-conformer slice")
                print(f"   FAILED: {err}", flush=True)
                break
            res["model"] = model
            res["input_sha256"] = chunk_input_hash(model, recs[lo:hi])
            part_raw.write_text(json.dumps(res))
            ids_all += res["ids"]
            energies_all += res["energies_ev"]
            failures += res.get("failures", [])
            n_failed_all += chunk_failed_count(res)
            model_meta = res.get("meta", {})
            wall += res.get("wall_seconds", 0.0)
        if not ok:
            summary[model] = {"ok": False, "model": model, "error": err}
            continue
        check_merged_ids(ids_all)
        merged = {"ok": True, "model": model, "meta": model_meta,
                  "ids": ids_all, "energies_ev": energies_all,
                  "n_ok": int(sum(1 for e in energies_all if e == e)),
                  "n_failed": n_failed_all, "failures": failures[:10],
                  "wall_seconds": wall}
        suffix = ".smoke.json" if args.limit else ".json"
        (out_dir / f"raw_{model}{suffix}").write_text(json.dumps(merged) + "\n")
        print(f"   ok  n={merged['n_ok']} failed={merged['n_failed']} "
              f"{wall:.1f}s total", flush=True)
        summary[model] = {k: v for k, v in merged.items()
                          if k not in ("ids", "energies_ev", "failures")}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ok = [m for m, s in summary.items() if s.get("ok")]
    print(f"\nran {len(ok)}/{len(models)}: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
