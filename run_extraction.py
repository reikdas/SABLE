#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import sys
import time

FILEPATH = pathlib.Path(__file__).resolve().parent
FIND_SUBMATRICES = FILEPATH / "find-submatrices"
PARTITIONER = FIND_SUBMATRICES / "build" / "partition_matrix"
BLOCKS_OUT = FIND_SUBMATRICES / "results"
BANDS_OUT = FIND_SUBMATRICES / "results_bands_075"

# The extraction thresholds used throughout the evaluation.
BLOCK_MIN_DENSITY = 0.5
BLOCK_MIN_AREA = 2500
BAND_MIN_DENSITY = 0.75

sys.path.insert(0, str(FIND_SUBMATRICES))


def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def resolve_matrices(args, parser):
    """The matrix names to process, from --matrix-set or the command line."""
    from download_matrices import load_matrix_names

    if args.matrices and args.matrix_set:
        parser.error("--matrix-set and an explicit matrix list are "
                     "mutually exclusive")
    if args.matrices:
        return list(args.matrices)
    which = "all" if args.matrix_set in (None, "paper") else args.matrix_set
    return load_matrix_names(which)


def matrix_path(name):
    """Path to the cached .mtx, downloading only if it is not already there.

    The artifact ships every evaluated matrix in the ssgetpy cache, so this
    is a lookup rather than a download.
    """
    from download_matrices import fast_download_matrix

    path, _info = fast_download_matrix(name)
    return path


def extract_blocks(names, force, min_density, min_area):
    """Run the block partitioner over `names`, writing one YAML each."""
    if not PARTITIONER.exists():
        sys.exit(f"error: {PARTITIONER} not found. Build it first:\n"
                 "    bash build_native.sh")
    BLOCKS_OUT.mkdir(parents=True, exist_ok=True)

    done = failed = skipped = 0
    for i, name in enumerate(names, 1):
        out = BLOCKS_OUT / f"{name}.yaml"
        if out.exists() and not force:
            skipped += 1
            continue
        path = matrix_path(name)
        if path is None:
            print(f"[{i}/{len(names)}] {name}: matrix not available, skipping")
            failed += 1
            continue
        print(f"[{i}/{len(names)}] {name}: block search...", flush=True)
        start = time.time()
        proc = subprocess.run(
            [str(PARTITIONER), path,
             "--min-density", str(min_density),
             "--min-area", str(min_area),
             "--output", str(out)],
        )
        if proc.returncode != 0:
            print(f"    partition_matrix exited {proc.returncode}")
            failed += 1
            continue
        print(f"    done in {_fmt(time.time() - start)} -> {out.name}")
        done += 1
    print(f"\nBlocks: {done} extracted, {skipped} already present, {failed} failed.")
    return failed


def extract_bands(names, force, min_density):
    """Run the band extractor over `names`, writing one YAML each."""
    from find_vdia import process_matrix, save_results

    BANDS_OUT.mkdir(parents=True, exist_ok=True)

    done = failed = skipped = 0
    for i, name in enumerate(names, 1):
        out = BANDS_OUT / f"{name}.yaml"
        if out.exists() and not force:
            skipped += 1
            continue
        print(f"[{i}/{len(names)}] {name}: band search...", flush=True)
        start = time.time()
        resolved, result = process_matrix(name, verbose=False,
                                          min_density=min_density)
        if result is None:
            print("    failed to load matrix")
            failed += 1
            continue
        save_results(resolved, result, output_dir=str(BANDS_OUT))
        print(f"    {len(result['bands'])} band(s) in "
              f"{_fmt(time.time() - start)} -> {resolved}.yaml")
        done += 1
    print(f"\nBands: {done} extracted, {skipped} already present, {failed} failed.")
    return failed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrices", nargs="*",
                        help="Matrix names to process (default: --matrix-set paper).")
    parser.add_argument("--matrix-set",
                        choices=("paper", "vbr_csr", "vdia_only", "fukaya"),
                        default=None,
                        help="A named matrix set from matrices.json; 'paper' "
                             "(the default) is the 78 the evaluation reports.")
    parser.add_argument("--stage", choices=("both", "blocks", "bands"),
                        default="both",
                        help="Which extractor to run (default: both).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract matrices that already have a YAML.")
    parser.add_argument("--min-density", type=float, default=BLOCK_MIN_DENSITY,
                        help=f"Block minimum density (default: {BLOCK_MIN_DENSITY}).")
    parser.add_argument("--min-area", type=int, default=BLOCK_MIN_AREA,
                        help=f"Block minimum area (default: {BLOCK_MIN_AREA}).")
    parser.add_argument("--band-density", type=float, default=BAND_MIN_DENSITY,
                        help=f"Band minimum density (default: {BAND_MIN_DENSITY}).")
    parser.add_argument("--list", action="store_true",
                        help="Print the matrices that would be processed and exit.")
    args = parser.parse_args()

    names = resolve_matrices(args, parser)
    if args.list:
        print("\n".join(names))
        return 0
    print(f"{len(names)} matrices, stage = {args.stage}\n")

    failed = 0
    if args.stage in ("both", "blocks"):
        failed += extract_blocks(names, args.force, args.min_density,
                                 args.min_area)
    if args.stage in ("both", "bands"):
        if args.stage == "both":
            print()
        failed += extract_bands(names, args.force, args.band_density)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
