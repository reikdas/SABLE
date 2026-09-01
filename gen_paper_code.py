#!/usr/bin/env python3

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time

FILEPATH = pathlib.Path(__file__).resolve().parent
DRIVER = FILEPATH / "bench_suitesparse.py"

# Both honour the same environment variables bench_suitesparse.py reads, so
# a redirected codegen or dense-tensor directory is cleared where it really
# is rather than where it would be by default.
CODEGEN_ROOT = pathlib.Path(os.environ.get("SABLE_CODEGEN_DIR") or FILEPATH)
STAGED_DATA_DIR = CODEGEN_ROOT / "Generated_Staged_Data"
DENSE_TENSOR_DIR = pathlib.Path(
    os.environ.get("SABLE_DENSE_TENSOR_DIR") or (FILEPATH / "Generated_dense_tensors")
)

MATRIX_SETS = ("paper", "vbr_csr", "vdia_only", "fukaya")


def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def resolve_matrices(args, parser):
    """The matrix names to generate for, from --matrix-set or the command line."""
    from download_matrices import load_matrix_names

    if args.matrices and args.matrix_set:
        parser.error("--matrix-set and an explicit matrix list are "
                     "mutually exclusive")
    if args.matrices:
        return list(args.matrices)
    which = "all" if args.matrix_set in (None, "paper") else args.matrix_set
    return load_matrix_names(which)


def generated_sources():
    """Every .c generated so far, across all kernel configurations."""
    return sorted(CODEGEN_ROOT.glob("Generated_Sp*/**/*.c"))


def drop_bulk_data():
    """Remove the staged data and dense right-hand sides."""
    for directory in (STAGED_DATA_DIR, DENSE_TENSOR_DIR):
        shutil.rmtree(directory, ignore_errors=True)


def generate(name, extra_args):
    """Generate every configuration's C for one matrix. True if it succeeded."""
    command = [sys.executable, str(DRIVER), "--codegen-only", name] + extra_args
    result = subprocess.run(command, cwd=str(FILEPATH))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("matrices", nargs="*", help="Matrix names to generate for.")
    parser.add_argument("--matrix-set", choices=MATRIX_SETS,
                        help="Generate for a named set from matrices.json instead "
                             "of an explicit list. Defaults to 'paper', the 78 "
                             "matrices the evaluation reports.")
    parser.add_argument("--keep-bulk-data", action="store_true",
                        help="Keep Generated_Staged_Data/ and "
                             "Generated_dense_tensors/ instead of clearing them "
                             "after each matrix. The generated C is only runnable "
                             "with them present, but they are large.")
    args, extra_args = parser.parse_known_args()

    if not DRIVER.exists():
        sys.exit(f"error: {DRIVER} not found")

    names = resolve_matrices(args, parser)
    print(f"Generating C for {len(names)} matrices")
    if extra_args:
        print(f"Forwarding to {DRIVER.name}: {' '.join(extra_args)}")

    started = time.time()
    failed = []
    for index, name in enumerate(names, start=1):
        print(f"\n=== [{index}/{len(names)}] {name} ===")
        if not generate(name, extra_args):
            print(f"  {name}: code generation failed")
            failed.append(name)
        if not args.keep_bulk_data:
            drop_bulk_data()

    sources = generated_sources()
    print("\n" + "=" * 60)
    print(f"Generated {len(sources)} C files from {len(names)} matrices "
          f"in {_fmt(time.time() - started)}")
    if not args.keep_bulk_data:
        print("Staged data and dense right-hand sides cleared; re-run this "
              "script to recreate them before compiling.")
    if failed:
        print(f"Failed for {len(failed)}: {', '.join(failed)}")
    if not sources:
        sys.exit("error: no C was generated")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
