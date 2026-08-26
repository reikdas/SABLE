#!/usr/bin/env python3
"""
Download every SuiteSparse matrix used in Section 5 of the SABLE paper into
the shared ssgetpy cache (~/.ssgetpy on Linux/macOS), which is exactly where
SABLE/bench_suitesparse.py, SABLE/find-submatrices/find_matrices.py, and
SABLE/find-submatrices/find_vdia.py already look first before downloading
anything themselves.

This is baked into the Docker image (run once at `docker build` time), so a
container built from the image already has every matrix on disk -- no
internet access is needed at benchmark time. It is also safe to re-run by
hand (inside or outside the container): already-cached matrices are skipped.

Usage:
    python3 download_matrices.py                # download the full 78-matrix union
    python3 download_matrices.py --set vbr_csr   # download only the 55 VBR+CSR matrices
    python3 download_matrices.py --set fukaya    # download only the 4 Fukaya matrices
    python3 download_matrices.py --set vdia_only # download only the 19 VDIA-only matrices
    python3 download_matrices.py --retries 5     # increase per-matrix retry count

NOTE ON SPEED: ssgetpy 1.0rc2's own Matrix.download() hard-codes
`time.sleep(0.1)` after every 4096-byte chunk it writes -- an artificial
client-side rate limit of ~40 KB/s per file, regardless of actual network
speed (confirmed: a plain `curl` to the same URL got ~2.7 MB/s, ~68x
faster). We do NOT call Matrix.download()/ssgetpy.fetch() at all -- both of
them re-`search()` internally and call the slow method on the fresh result,
so monkey-patching the class method did not reliably take effect. Instead we
use get_matrix_info() (pure local-index lookup, no network) purely for
name/group/URL resolution and Matrix.localpath() (pure path computation) for
where the file belongs, then fetch the tar.gz ourselves with a plain
`requests` stream (no sleep, 1 MiB chunks) and hand it to ssgetpy's own
(unmodified, correct) `ssgetpy.bundle.extract()` -- so the on-disk cache
layout is byte-for-byte identical to what ssgetpy would have produced, and
get_matrix_info()/localpath() and the rest of the pipeline see no
difference.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import requests
from ssgetpy import bundle as ssgetpy_bundle

SABLE_ROOT = pathlib.Path(__file__).resolve().parent
MATRICES_JSON = SABLE_ROOT / "matrices.json"

# find_matrices.py wraps ssgetpy's search() with the exact name-resolution
# logic the rest of the pipeline expects (see get_matrix_info).
sys.path.insert(0, str(SABLE_ROOT / "find-submatrices"))
from find_matrices import get_matrix_info  # noqa: E402


def _destpath_for(info, dest, fmt):
    """Where this matrix's files belong. With --dest we mirror ssgetpy's own
    <root>/<format>/<group>/ layout so the resulting directory can be dropped
    straight in as ~/.ssgetpy; without it we use ssgetpy's default root."""
    if dest is None:
        return None
    return os.path.join(dest, fmt, info.group)


def fast_download_matrix(matrix_name, fmt="MM", dest=None):
    """Equivalent of find_matrices.download_matrix(), without ssgetpy's
    artificial per-chunk sleep. Returns (matrix_path, matrix_info) or
    (None, None), matching download_matrix()'s contract."""
    info = get_matrix_info(matrix_name)
    if info is None:
        return None, None

    localdestpath, localdest = info.localpath(
        format=fmt, destpath=_destpath_for(info, dest, fmt), extract=True
    )
    matrix_path = os.path.join(localdestpath, f"{info.name}.mtx")
    if os.path.exists(matrix_path):
        return matrix_path, info

    destpath = os.path.dirname(localdest)
    os.makedirs(destpath, exist_ok=True)
    response = requests.get(info.url(fmt), stream=True, timeout=120)
    response.raise_for_status()
    with open(localdest, "wb") as outfile:
        for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MiB, no sleep
            outfile.write(chunk)
    if fmt in ("MM", "RB"):
        ssgetpy_bundle.extract(localdest)

    if not os.path.exists(matrix_path):
        return None, None
    return matrix_path, info


def load_matrix_names(which):
    data = json.loads(MATRICES_JSON.read_text())
    sets = ["vbr_csr", "vdia_only", "fukaya"] if which == "all" else [which]
    names = []
    seen = set()
    for key in sets:
        for entry in data[key]["matrices"]:
            if entry["name"] not in seen:
                seen.add(entry["name"])
                names.append(entry["name"])
    return names


def already_cached(matrix_name, dest=None):
    info = get_matrix_info(matrix_name)
    if info is None:
        return False, None
    cache_subdir, _ = info.localpath(
        format="MM", destpath=_destpath_for(info, dest, "MM"), extract=True
    )
    mtx_path = pathlib.Path(cache_subdir) / f"{info.name}.mtx"
    return mtx_path.exists(), mtx_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--set",
        dest="which",
        default="all",
        choices=["all", "vbr_csr", "vdia_only", "fukaya"],
        help="Which matrix subset to download (default: all -- the 78-matrix union).",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per matrix on download failure (default: 3).")
    parser.add_argument(
        "--dest",
        default=None,
        help="Directory to populate instead of the shared ssgetpy cache. Mirrors "
             "ssgetpy's <root>/MM/<Group>/ layout, so the result can ship as the "
             "artifact's matrices/ folder and be COPYed straight to ~/.ssgetpy.",
    )
    args = parser.parse_args()

    dest = os.path.abspath(args.dest) if args.dest else None
    names = load_matrix_names(args.which)
    print(f"Downloading {len(names)} matrices (set={args.which}) into "
          f"{dest or 'the shared ssgetpy cache'} ...\n")

    ok, skipped, failed = [], [], []
    for i, name in enumerate(names, 1):
        cached, mtx_path = already_cached(name, dest)
        if cached:
            print(f"[{i}/{len(names)}] {name}: already cached at {mtx_path}")
            skipped.append(name)
            continue

        print(f"[{i}/{len(names)}] {name}: downloading ...")
        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                matrix_path, matrix_info = fast_download_matrix(name, dest=dest)
                if matrix_path is not None:
                    print(f"    -> {matrix_path}")
                    ok.append(name)
                    break
                last_err = "download_matrix returned None (matrix not found or download failed)"
            except Exception as exc:  # noqa: BLE001 - report and retry
                last_err = repr(exc)
            print(f"    attempt {attempt}/{args.retries} failed: {last_err}")
            if attempt < args.retries:
                time.sleep(2 * attempt)
        else:
            failed.append((name, last_err))

    print("\n=== Summary ===")
    print(f"Downloaded : {len(ok)}")
    print(f"Skipped    : {len(skipped)} (already cached)")
    print(f"Failed     : {len(failed)}")
    if failed:
        for name, err in failed:
            print(f"  - {name}: {err}")
        print(
            "\nSome matrices failed to download. Re-run this script (already-cached matrices are\n"
            "skipped automatically) once network access is available, or download the failed\n"
            "matrices by hand from https://sparse.tamu.edu/ and place them at\n"
            "~/.ssgetpy/MM/<Group>/<Name>/<Name>.mtx"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
