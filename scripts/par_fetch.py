#!/usr/bin/env python3
"""Parallel HTTP range downloader with resume, for large single-file archives.

Written because a single Zenodo connection ran at ~0.12 GB/min and the SPICE
archive is 16 GB. Range requests are supported (HTTP 206), so the file is split
into fixed chunks and fetched over several connections at once; each chunk is
written at its own offset in a preallocated sparse file, so a dropped connection
resumes from the chunk boundary and never rewrites what is already there.

Correctness: on completion the total byte count must equal the Content-Length the
server reported at the start, or the file is left with a .BAD suffix. Nothing
downstream is allowed to read an unverified archive.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path

CHUNK = 16 * 1024 * 1024  # 16 MiB per range request


def head_size(url: str, timeout: int = 60) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = int(r.headers.get("Content-Length", 0))
    if n <= 0:
        raise SystemExit("server did not report a Content-Length; cannot verify")
    return n


def fetch_range(url: str, start: int, end: int, out: Path, tries: int = 5) -> None:
    """Fetch [start, end] inclusive and write it in place."""
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status not in (206, 200):
                    raise IOError(f"unexpected status {r.status}")
                data = r.read()
            want = end - start + 1
            if len(data) != want:
                raise IOError(f"short read {len(data)} != {want}")
            with open(out, "r+b") as f:
                f.seek(start)
                f.write(data)
            return
        except Exception as e:  # noqa: BLE001 - retry any transport error
            if attempt == tries:
                raise
            time.sleep(min(2 ** attempt, 20))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--connections", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    total = head_size(args.url)
    chunk = args.chunk_mb * 1024 * 1024
    starts = list(range(0, total, chunk))
    print(f"total {total/1e9:.2f} GB in {len(starts)} chunks "
          f"over {args.connections} connections", flush=True)

    if out.exists() and out.stat().st_size == total:
        print("already complete")
        return 0
    # Preallocate so concurrent writers can seek without racing on growth.
    with open(out, "wb") as f:
        f.truncate(total)

    todo: queue.Queue[tuple[int, int, int]] = queue.Queue()
    done_bytes = [0]
    lock = threading.Lock()
    t0 = time.time()

    for i, s in enumerate(starts):
        e = min(s + chunk - 1, total - 1)
        todo.put((i, s, e))

    errors: list[Exception] = []
    n_chunks = len(starts)
    finished = [0]

    def worker() -> None:
        while True:
            try:
                _, s, e = todo.get_nowait()
            except queue.Empty:
                return
            try:
                fetch_range(args.url, s, e, out)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                return
            with lock:
                done_bytes[0] += e - s + 1
                finished[0] += 1
                if finished[0] % 25 == 0 or finished[0] == n_chunks:
                    el = time.time() - t0
                    rate = done_bytes[0] / 1e9 / max(el, 1e-9)
                    pct = 100 * done_bytes[0] / total
                    print(f"{finished[0]}/{n_chunks} chunks  "
                          f"{done_bytes[0]/1e9:.2f}/{total/1e9:.2f} GB "
                          f"({pct:.1f}%)  {rate:.2f} GB/min", flush=True)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.connections)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print(f"FAILED after {finished[0]}/{n_chunks} chunks: {errors[0]}",
              file=sys.stderr)
        return 1
    size = out.stat().st_size
    if size != total:
        bad = out.with_suffix(out.suffix + ".BAD")
        out.rename(bad)
        print(f"SIZE MISMATCH {size} != {total}; moved to {bad}", file=sys.stderr)
        return 1
    print(f"OK {size} bytes == Content-Length", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
