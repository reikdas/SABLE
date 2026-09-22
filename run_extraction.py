#!/usr/bin/env python3
"""Re-derive the extraction the evaluation used, or search new matrices.

The image ships the extraction the figures read: dense blocks in
find-submatrices/results/<matrix>.yaml and diagonal bands in
find-submatrices/results_bands_075/<matrix>.yaml. Not every matrix has both.
The evaluation ran the block partitioner on the VBR+CSR set and on the band
set, and the band finder on the band set and the four Fukaya matrices
(STAGE_SETS below). The Fukaya matrices, compared as VDIA+CSR only, were never
block-extracted. With no matrix names this script covers exactly that, stage
by stage, and skips every extraction that is already present unless --force is
given, so on a fresh image it reports that there is nothing to do. A matrix
named on the command line is searched in both stages whether or not the
evaluation used it.

How long a search takes: the block partitioner is exhaustive and stops at
--block-timeout (four hours, the evaluation's budget), writing the blocks it
found by then; it prints nothing while it searches. The band finder has no
budget and runs to completion, printing each band as it accepts it: seconds on
a small matrix, 3.5 hours on circuit5M_dc.
"""

import argparse
import pathlib
import subprocess
import sys
import time

FILEPATH = pathlib.Path(__file__).resolve().parent
FIND_SUBMATRICES = FILEPATH / "find-submatrices"
PARTITIONER = FIND_SUBMATRICES / "build" / "partition_matrix"
STAGE_DIRS = {
    "blocks": FIND_SUBMATRICES / "results",
    "bands": FIND_SUBMATRICES / "results_bands_075",
}

# The extraction thresholds used throughout the evaluation.
BLOCK_MIN_DENSITY = 0.5
BLOCK_MIN_AREA = 2500
BLOCK_TIMEOUT_SECONDS = 4.0 * 60.0 * 60.0
BAND_MIN_DENSITY = 0.75

# The sets of matrices.json each stage covered in the evaluation: blocks for
# Figure 5 and, on the band set, for the composition of Figure 7; bands for
# Figures 6 and 7. vdia_in_vbr_csr is the part of vbr_csr that is in the band
# set, so a request for vbr_csr band-extracts those five and no others.
STAGE_SETS = {
    "blocks": ("vbr_csr", "vdia_only"),
    "bands": ("vdia_only", "vdia_in_vbr_csr", "fukaya"),
}
PART_OF = {"vdia_in_vbr_csr": "vbr_csr"}
MATRIX_SETS = ("paper", "vbr_csr", "vdia_only", "fukaya")

sys.path.insert(0, str(FIND_SUBMATRICES))


def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def stage_sets(matrix_set, stage):
    """The matrices.json sets a request covers in one stage."""
    if matrix_set == "paper":
        return STAGE_SETS[stage]
    return tuple(s for s in STAGE_SETS[stage] if s == matrix_set or PART_OF.get(s) == matrix_set)


def plan(args, stage):
    """(matrices, already extracted, to search, where the list came from) for one stage."""
    from download_matrices import load_matrix_names

    if args.matrices:
        names = list(args.matrices)
        origin = "the command line"
    else:
        sets = stage_sets(args.matrix_set or "paper", stage)
        names = []
        for s in sets:
            names.extend(n for n in load_matrix_names(s) if n not in names)
        origin = ", ".join(sets)
    present = [n for n in names if (STAGE_DIRS[stage] / f"{n}.yaml").exists()]
    todo = names if args.force else [n for n in names if n not in present]
    return names, present, todo, origin


def describe(stage, names, present, todo, origin, matrix_set, force):
    where = STAGE_DIRS[stage].relative_to(FILEPATH)
    if not names:
        print(f"{stage.capitalize()} ({where}): no matrix of the {matrix_set} set was "
              f"{stage[:-1]}-extracted for the evaluation; name a matrix to search it.")
        return
    noun = "matrix" if len(names) == 1 else "matrices"
    print(f"{stage.capitalize()} ({where}): {len(names)} {noun} from {origin}; "
          f"{len(present)} already extracted, {len(todo)} to search"
          + (", re-extracting those present (--force)" if force and present else ""))


def matrix_path(name):
    """Path to the cached .mtx, downloading only if it is not already there.

    The artifact ships every evaluated matrix in the ssgetpy cache, so this
    is a lookup rather than a download.
    """
    from download_matrices import fast_download_matrix

    path, _info = fast_download_matrix(name)
    return path


def matrix_shape(path):
    """(rows, cols, nnz) from the Matrix Market header, or None."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("%"):
                    continue
                fields = line.split()
                return tuple(int(v) for v in fields[:3]) if len(fields) >= 3 else None
    except (OSError, ValueError):
        return None
    return None


def matrix_label(name, path):
    shape = matrix_shape(path) if path else None
    if shape is None:
        return name
    rows, cols, nnz = shape
    return f"{name} ({rows:,} x {cols:,}, {nnz:,} nonzeros)"


def extract_blocks(names, min_density, min_area, timeout_seconds):
    """Run the block partitioner over `names`, writing one YAML each."""
    if not PARTITIONER.exists():
        sys.exit(f"error: {PARTITIONER} not found. Build it first:\n"
                 "    bash build_native.sh")
    out_dir = STAGE_DIRS["blocks"]
    out_dir.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    for i, name in enumerate(names, 1):
        out = out_dir / f"{name}.yaml"
        path = matrix_path(name)
        if path is None:
            print(f"[{i}/{len(names)}] {name}: matrix not available, skipping")
            failed += 1
            continue
        print(f"[{i}/{len(names)}] {matrix_label(name, path)}: block search, budget "
              f"{_fmt(timeout_seconds)}. The partitioner prints nothing until it finishes or "
              "the budget runs out, then writes the blocks it found.", flush=True)
        start = time.time()
        proc = subprocess.run(
            [str(PARTITIONER), path,
             "--min-density", str(min_density),
             "--min-area", str(min_area),
             "--timeout-seconds", str(timeout_seconds),
             "--output", str(out)],
        )
        if proc.returncode != 0:
            print(f"    partition_matrix exited {proc.returncode}")
            failed += 1
            continue
        print(f"    done in {_fmt(time.time() - start)} -> {out.relative_to(FILEPATH)}", flush=True)
        done += 1
    print(f"\nBlocks: {done} extracted, {failed} failed.")
    return failed


def extract_bands(names, min_density):
    """Run the band extractor over `names`, writing one YAML each."""
    from find_vdia import process_matrix, save_results

    out_dir = STAGE_DIRS["bands"]
    out_dir.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    for i, name in enumerate(names, 1):
        path = matrix_path(name)
        print(f"[{i}/{len(names)}] {matrix_label(name, path)}: band search. It has no time budget "
              "and prints each band as it accepts it: seconds on a small matrix, 3.5 h on "
              "circuit5M_dc.", flush=True)
        start = time.time()
        resolved, result = process_matrix(name, verbose=True, min_density=min_density)
        if result is None:
            print("    failed to load matrix")
            failed += 1
            continue
        save_results(resolved, result, output_dir=str(out_dir))
        print(f"    {len(result['bands'])} band(s) in {_fmt(time.time() - start)} "
              f"-> {out_dir.relative_to(FILEPATH)}/{resolved}.yaml", flush=True)
        done += 1
    print(f"\nBands: {done} extracted, {failed} failed.")
    return failed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrices", nargs="*",
                        help="Matrix names to search in both stages, whether or not the evaluation "
                             "used them (default: the evaluation's own extraction, see --matrix-set).")
    parser.add_argument("--matrix-set", choices=MATRIX_SETS, default=None,
                        help="A named matrix set from matrices.json; 'paper' (the default) is the 78 "
                             "the evaluation reports. Each stage covers the part of the set the "
                             "evaluation extracted.")
    parser.add_argument("--stage", choices=("both", "blocks", "bands"), default="both",
                        help="Which extractor to run (default: both).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract matrices that already have a YAML.")
    parser.add_argument("--min-density", type=float, default=BLOCK_MIN_DENSITY,
                        help=f"Block minimum density (default: {BLOCK_MIN_DENSITY}).")
    parser.add_argument("--min-area", type=int, default=BLOCK_MIN_AREA,
                        help=f"Block minimum area (default: {BLOCK_MIN_AREA}).")
    parser.add_argument("--block-timeout", type=float, default=BLOCK_TIMEOUT_SECONDS, metavar="SECONDS",
                        help="Stop the block search on one matrix after this long and keep the blocks "
                             f"found by then (default: {int(BLOCK_TIMEOUT_SECONDS)}, the evaluation's "
                             "four hours).")
    parser.add_argument("--band-density", type=float, default=BAND_MIN_DENSITY,
                        help=f"Band minimum density (default: {BAND_MIN_DENSITY}).")
    parser.add_argument("--list", action="store_true",
                        help="Print what each stage would cover and exit.")
    args = parser.parse_args()
    if args.matrices and args.matrix_set:
        parser.error("--matrix-set and an explicit matrix list are mutually exclusive")

    stages = ("blocks", "bands") if args.stage == "both" else (args.stage,)
    plans = {stage: plan(args, stage) for stage in stages}
    for stage in stages:
        describe(stage, *plans[stage], args.matrix_set or "paper", args.force)

    if args.list:
        for stage in stages:
            names, _present, todo, _origin = plans[stage]
            for name in names:
                print(f"  {stage}: {name}" + ("" if name in todo else "  (already extracted)"))
        return 0
    if not any(plans[stage][2] for stage in stages):
        if any(plans[stage][0] for stage in stages):
            print("\nNothing to do: every extraction requested is already present. --force "
                  "re-derives them (the block search is exhaustive: up to --block-timeout per "
                  "matrix); naming a matrix searches it.")
        return 0

    failed = 0
    if "blocks" in stages and plans["blocks"][2]:
        print()
        failed += extract_blocks(plans["blocks"][2], args.min_density, args.min_area, args.block_timeout)
    if "bands" in stages and plans["bands"][2]:
        print()
        failed += extract_bands(plans["bands"][2], args.band_density)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
