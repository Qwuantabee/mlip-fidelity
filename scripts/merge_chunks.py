#!/usr/bin/env python3
"""Merge persisted per-chunk raw results into one model file, and say what is missing.

Refuses (exit 1, no raw_<model>.json written) when:
  * a chunk is not ok, or its len(energies_ev) != len(ids);
  * a chunk is older than the conformers.jsonl it covers (stale geometry);
  * a chunk carries an input_sha256 that does not match the CURRENT
    conformers.jsonl for its ids (chunks without the hash are guarded by the
    mtime check only);
  * a conformer id appears in more than one chunk with disagreeing energies
    (|dE| > 1e-9 eV, or NaN in one and finite in the other) - the first
    occurrence used to win silently, and a stale 999.0 beat a fresh 1.0;
  * the merged ids are not exactly the id set of conformers.jsonl.
"""
import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOL_EV = 1e-9


def _sweep():
    spec = importlib.util.spec_from_file_location(
        "_run_sweep_for_merge", ROOT / "scripts" / "run_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _disagree(a, b) -> bool:
    na, nb = a != a, b != b
    if na or nb:
        return na != nb
    return abs(a - b) > TOL_EV


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", default=str(ROOT / "results"))
    ap.add_argument("--conformers", default=None,
                    help="conformer set the chunks must cover "
                         "(default <dir>/conformers.jsonl, else results/conformers.jsonl)")
    a = ap.parse_args()
    d = Path(a.dir)
    conf_path = Path(a.conformers) if a.conformers else d / "conformers.jsonl"
    if not a.conformers and not conf_path.exists():
        conf_path = ROOT / "results" / "conformers.jsonl"
    if not conf_path.exists():
        raise SystemExit(f"missing {conf_path}; cannot check coverage - refusing to merge")
    recs = [json.loads(line) for line in conf_path.read_text().splitlines() if line.strip()]
    by_id = {r["id"]: r for r in recs}
    sweep = _sweep()

    # Match ONLY the range-named chunks (chunk_<model>_<6d>_<6d>.json). A legacy
    # ordinal name like chunk_<model>_000.json sorts BEFORE every range file
    # ('.' < '0') and silently supplied stale energies.
    pat = re.compile(rf"^chunk_{re.escape(a.model)}_\d{{6}}_\d{{6}}\.json$")
    parts = sorted(p for p in d.glob(f"chunk_{a.model}_*.json") if pat.match(p.name))
    if not parts:
        raise SystemExit(f"no chunk files for {a.model}")

    # Chunk boundaries are not fixed - a re-run with a different --chunk writes
    # ranges that OVERLAP earlier files. Overlapping ids are deduplicated, but
    # only when every copy agrees; a disagreement means one chunk is stale.
    problems: list[str] = []
    ids, ens, fails, meta, wall = [], [], [], {}, 0.0
    seen: dict[str, int] = {}
    dupes, disagreements = 0, []
    for p in parts:
        r = json.loads(p.read_text())
        if not r.get("ok"):
            problems.append(f"{p.name}: chunk is not ok")
            continue
        cids, cens = list(r.get("ids", [])), list(r.get("energies_ev", []))
        if len(cids) != len(cens):
            problems.append(f"{p.name}: {len(cids)} ids but {len(cens)} energies")
            continue
        if sweep.chunk_older_than(p, conf_path):
            problems.append(f"{p.name}: older than {conf_path.name} (stale geometry)")
            continue
        if r.get("input_sha256"):
            unknown = [c for c in cids if c not in by_id]
            if unknown:
                problems.append(f"{p.name}: {len(unknown)} ids not in {conf_path.name}")
                continue
            want = sweep.chunk_input_hash(r.get("model", a.model), [by_id[c] for c in cids])
            if r.get("model", a.model) != a.model or r["input_sha256"] != want:
                problems.append(f"{p.name}: input hash does not match current "
                                f"{conf_path.name} (computed from other inputs)")
                continue
        for cid, e in zip(cids, cens):
            e = float("nan") if e is None else e
            if cid in seen:
                dupes += 1
                if _disagree(ens[seen[cid]], e):
                    disagreements.append(f"{cid}: {ens[seen[cid]]!r} vs {e!r} ({p.name})")
                continue
            seen[cid] = len(ids)
            ids.append(cid)
            ens.append(e)
        fails += r.get("failures", [])
        meta = r.get("meta", meta)
        wall += r.get("wall_seconds", 0.0)

    if disagreements:
        problems.append(f"{len(disagreements)} duplicated conformer(s) with disagreeing "
                        f"energies, e.g. " + "; ".join(disagreements[:3]))
    missing = set(by_id) - set(ids)
    extra = set(ids) - set(by_id)
    if missing or extra:
        problems.append(f"coverage: {len(missing)} conformers of {len(by_id)} missing, "
                        f"{len(extra)} not in {conf_path.name}")
    if len(ids) != len(ens):
        problems.append(f"ids/energies length mismatch ({len(ids)} vs {len(ens)})")
    if problems:
        for msg in problems:
            print(f"{a.model}: {msg}", file=sys.stderr)
        print(f"{a.model}: refusing to write raw_{a.model}.json", file=sys.stderr)
        return 1

    # n_failed is the NaN count, not len(failures): run_model keeps only the first 10
    # failure dicts, so len(failures) reported 10 for a model with 319 NaN conformers
    # and the published summary inherited that wrong number.
    nan_count = int(sum(1 for e in ens if e != e))
    merged = {"ok": True, "model": a.model, "meta": meta, "ids": ids,
              "energies_ev": ens, "n_ok": len(ids) - nan_count,
              "n_failed": nan_count, "failures": fails[:10], "wall_seconds": wall,
              "n_chunks_merged": len(parts)}
    (d / f"raw_{a.model}.json").write_text(json.dumps(merged) + "\n")
    print(f"{a.model}: merged {len(parts)} chunks ({dupes} agreeing duplicate "
          f"conformers dropped), n_ok={merged['n_ok']} failed={merged['n_failed']} "
          f"wall={wall:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
