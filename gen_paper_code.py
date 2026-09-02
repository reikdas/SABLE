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
COMPLETED_DIR = CODEGEN_ROOT / "Generated_Code_Completed"

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


def matrix_sources(name):
    """Every .c generated for one matrix, across all kernel configurations."""
    return sorted(CODEGEN_ROOT.glob(f"Generated_Sp*/**/{name}.c"))


def completion_stamp(name):
    return COMPLETED_DIR / f"{name}.done"


def already_generated(name):
    """True if an earlier run finished this matrix and its C is still on disk.

    Deliberately a stamp rather than a bare "does any .c exist" test. A matrix
    whose generation died part-way -- the case a resumed run exists to recover
    from -- leaves behind the C of the configurations it did reach, so an
    existence test would skip exactly the matrices that need redoing. The
    second half of the check catches generated C wiped out from under a
    surviving stamp.
    """
    return completion_stamp(name).exists() and bool(matrix_sources(name))


def mark_generated(name):
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    completion_stamp(name).touch()


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
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip matrices an earlier run finished rather than "
                             "regenerating them, for resuming a run that died "
                             "part-way. Off by default: a plain run rewrites "
                             "every matrix's C. Completion is recorded under "
                             "Generated_Code_Completed/ either way.")
    parser.add_argument("--allow-failures", action="store_true",
                        help="Exit 0 even when some matrices fail, so a build is "
                             "not sunk by the handful the machine has not got the "
                             "memory to generate. They are still reported, and "
                             "still have no completion stamp, so a later "
                             "--skip-existing run retries exactly them. Exiting "
                             "non-zero because nothing generated at all is not "
                             "suppressed.")
    parser.add_argument("--keep-bulk-data", action="store_true",
                        help="Keep Generated_Staged_Data/ and "
                             "Generated_dense_tensors/ instead of clearing them "
                             "after each matrix. The generated C is only runnable "
                             "with them present, but they are large.")
    # Arguments for bench_suitesparse.py come after "--" rather than being
    # sniffed out of the command line. parse_known_args lets the greedy
    # nargs="*" positional swallow an unknown option's value, so
    # "--matrix-set fukaya --operation spmv" would take spmv for a matrix name
    # and forward a valueless --operation.
    argv = sys.argv[1:]
    if "--" in argv:
        separator = argv.index("--")
        own_args, extra_args = argv[:separator], argv[separator + 1:]
    else:
        own_args, extra_args = argv, []
    args = parser.parse_args(own_args)

    if not DRIVER.exists():
        sys.exit(f"error: {DRIVER} not found")

    names = resolve_matrices(args, parser)
    print(f"Generating C for {len(names)} matrices")
    if extra_args:
        print(f"Forwarding to {DRIVER.name}: {' '.join(extra_args)}")

    started = time.time()
    failed = []
    skipped = []
    for index, name in enumerate(names, start=1):
        if args.skip_existing and already_generated(name):
            print(f"\n=== [{index}/{len(names)}] {name}: already generated, "
                  f"skipping ===")
            skipped.append(name)
            continue
        print(f"\n=== [{index}/{len(names)}] {name} ===")
        # Drop any stamp from an earlier run first, so a matrix that used to
        # generate cleanly and no longer does is not left marked as done.
        completion_stamp(name).unlink(missing_ok=True)
        if generate(name, extra_args):
            mark_generated(name)
        else:
            print(f"  {name}: code generation failed")
            failed.append(name)
        if not args.keep_bulk_data:
            drop_bulk_data()

    sources = generated_sources()
    print("\n" + "=" * 60)
    if skipped:
        print(f"Generated C for {len(names) - len(skipped)} matrices "
              f"({len(skipped)} already done, skipped) in "
              f"{_fmt(time.time() - started)}; {len(sources)} C files on disk")
    else:
        print(f"Generated {len(sources)} C files from {len(names)} matrices "
              f"in {_fmt(time.time() - started)}")
    if not args.keep_bulk_data:
        print("Staged data and dense right-hand sides cleared; re-run this "
              "script to recreate them before compiling.")
    if failed:
        print(f"Failed for {len(failed)}: {', '.join(failed)}")
        if args.allow_failures:
            print("Continuing anyway (--allow-failures); those matrices have no "
                  "generated C. Re-run with --skip-existing to retry just them.")
    if not sources:
        sys.exit("error: no C was generated")
    sys.exit(1 if failed and not args.allow_failures else 0)


if __name__ == "__main__":
    main()
