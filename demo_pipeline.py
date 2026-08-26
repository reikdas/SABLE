#!/usr/bin/env python3
"""
End-to-end walkthrough of the SABLE pipeline on a single matrix, printing
every intermediate artifact the paper's evaluation produces internally:

  1. Band (VDIA) extraction  -> discovered diagonal-band regions + timing
  2. Block (VBR) extraction  -> discovered dense blocks + timing (via a Plan)
  3. The .sabledata file     -> the staged data SABLE's generated C reads at runtime
  4. The generated C code    -> the compiled kernel source
  5. Compile + run           -> timings for the generated SpMV program

Requires Linux + gcc + MKL + a built find-submatrices/build/partition_matrix
binary (steps 2-5); run this inside the Docker container described in the
paper's Artifact Appendix. Step 1 (pure Python) and matrix loading work anywhere SABLE's
Python dependencies are installed.

Usage:
    python3 demo_pipeline.py                 # uses heart1 (group Norris)
    python3 demo_pipeline.py cari --group Meszaros
"""
import argparse
import os
import pathlib
import sys
import tempfile

SABLE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SABLE_DIR))
sys.path.insert(0, str(SABLE_DIR / "find-submatrices"))


def get_matrix_path(name):
    """Return the local .mtx path for `name`, downloading it if needed."""
    from find_matrices import download_matrix, get_matrix_info

    info = get_matrix_info(name)
    if info is None:
        sys.exit(f"Matrix {name!r} not found via ssgetpy search.")
    cache_subdir, _ = info.localpath(format="MM", extract=True)
    mtx_path = os.path.join(cache_subdir, f"{info.name}.mtx")
    if os.path.exists(mtx_path):
        return mtx_path
    print(f"{name} not cached yet; downloading ...")
    mtx_path, _ = download_matrix(name)
    if mtx_path is None:
        sys.exit(
            f"Failed to download {name!r}. Run download_matrices.py first, "
            "or check network access."
        )
    return mtx_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrix_name", nargs="?", default="heart1")
    parser.add_argument("--artifact-dir", default=None, help="Where to write .sabledata/.c/binary (default: a fresh temp dir)")
    args = parser.parse_args()

    mtx_path = get_matrix_path(args.matrix_name)
    print(f"Using matrix: {mtx_path}\n")

    # These imports need only numpy/scipy/pyyaml -- safe on any platform.
    from sable import Matrix

    matrix = Matrix(mtx_path, name=args.matrix_name)
    print(f"Loaded {args.matrix_name}: {matrix.nrows} x {matrix.ncols}, {matrix.nnz} nonzeros\n")

    # -----------------------------------------------------------------
    # Step 1: standalone band (VDIA) extraction, for inspection only.
    # Pure Python (calls find-submatrices/find_vdia.py in-process, no
    # compiled binary needed) -- this step works on any platform.
    # -----------------------------------------------------------------
    print("=== Step 1: band (VDIA) extraction (inspection only) ===")
    try:
        from sable.extractors.band_extractor import BandExtractor

        band_probe = BandExtractor(min_density=0.75, verbose=True)  # 0.75 = paper's evaluation density
        _vdia_fmt, _residual = band_probe.extract(matrix)
        n_bands = len(band_probe.last_timing["bands"]) if band_probe.last_timing else 0
        print(f"\nBands found: {n_bands}")
        if band_probe.last_timing:
            for band in band_probe.last_timing["bands"]:
                print(f"  {band}")
    except Exception as exc:  # noqa: BLE001
        print(f"Band extraction failed: {exc!r}")
    print()

    # -----------------------------------------------------------------
    # Steps 2-5 need the compiled find-submatrices partition_matrix binary
    # (block extraction), gcc, and MKL -- i.e. a Linux environment with the
    # native toolchain built (see build_native.sh). Run inside the
    # Docker container; on an unsupported platform this section prints a
    # clear message and exits instead of a raw traceback.
    # -----------------------------------------------------------------
    try:
        import numpy as np

        from sable import Plan
        from sable.extractors import BlockDetector, CSRConvertor
        from sable.kernels import MixedVBRSpmv, NaiveCSRSpmv
        from sable.tensor import DenseInput
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Could not import the SABLE frontend/kernels ({exc!r}). "
            "Steps 2-5 require the Python dependencies in SABLE/requirements.txt "
            "to be installed. Run this inside the Docker container described in "
            "the paper's Artifact Appendix."
        )

    artifact_dir = args.artifact_dir or tempfile.mkdtemp(prefix="sable_demo_")
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"=== Step 2: block (VBR) extraction + CSR residual (artifact_dir={artifact_dir}) ===")

    rhs_path = os.path.join(artifact_dir, "rhs.vector")
    np.savetxt(rhs_path, np.random.default_rng(0).random(matrix.ncols))

    plan = Plan(matrix, artifact_dir=artifact_dir)
    plan.rhs(DenseInput.vector(rhs_path, matrix.ncols))  # DenseInput.vector(path, size)

    detector = BlockDetector(min_density=0.5, min_area=2500, threads=4)
    try:
        vbr = plan.extract(detector)
    except (RuntimeError, FileNotFoundError) as exc:
        sys.exit(
            f"Block extraction failed ({exc!r}).\n"
            "This needs the compiled find-submatrices/build/partition_matrix binary "
            "(built by build_native.sh, or built lazily here on first use if "
            "cmake/a C++ compiler are available) -- run this inside the Docker "
            "container described in the paper's Artifact Appendix."
        )
    print(f"Block-extraction timing/metadata: {detector.last_timing}\n")

    csr = plan.extract(CSRConvertor())
    plan.dispatch(vbr, MixedVBRSpmv())
    plan.dispatch(csr, NaiveCSRSpmv())

    # --- Step 3+4: compile -> writes both the .sabledata file and the .c file ---
    print("=== Step 3+4: compiling plan -> .sabledata + generated C code ===")
    executor = plan.compile(filename=args.matrix_name, bench=5)
    print(f".sabledata file : {executor.data_path}")
    print(f"Generated C file: {executor.c_path}")
    print(f"\n--- first 40 lines of {executor.c_path} ---")
    with open(executor.c_path) as f:
        for _ in range(40):
            line = f.readline()
            if not line:
                break
            print(line, end="")
    print("--- (truncated; open the file directly to see the rest) ---\n")

    # --- Step 5: build (gcc) + run, producing timings ---
    print("=== Step 5: compile (gcc) + run ===")
    try:
        executor.build()
        output = executor.run()
        print(output)
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Build/run failed ({exc!r}). This needs gcc and Intel MKL "
            "(MKLROOT set) -- run this inside the Docker container described in "
            "the paper's Artifact Appendix."
        )

    print(
        "\nTo regenerate a full results JSON file (many matrices/kernels at once) "
        "instead of this one-matrix demo, see the paper's Artifact Appendix "
        "'Generating a results file' section, e.g.:\n"
        "  cd SABLE && python3 bench_suitesparse.py --operation spmv "
        "--vbr-kernels blockmixed --csr-kernels naive "
        f"{args.matrix_name}"
    )


if __name__ == "__main__":
    main()
