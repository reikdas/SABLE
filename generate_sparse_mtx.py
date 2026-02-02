#!/usr/bin/env python3
"""
Generate Matrix Market (.mtx) files from the CSR sparse parts of VBR-C split files.

Reads VBR-C files from Generated_VBR_split/ and writes corresponding .mtx files
to Generated_sparse_mtx/. Only processes split variants (not sparse variants).

Usage:
    python3 generate_sparse_mtx.py                    # process all split VBR-C files
    python3 generate_sparse_mtx.py matrix1 matrix2    # process specific matrices
"""

import argparse
import os
import pathlib
import sys

from scipy.sparse import csr_array
from scipy.io import mmwrite

from utils.fileio import read_vbrc

FILEPATH = pathlib.Path(__file__).resolve().parent
GENERATED_VBR_SPLIT_DIR = FILEPATH / "Generated_VBR_split"
GENERATED_SPARSE_MTX_DIR = FILEPATH / "Generated_sparse_mtx"


def vbrc_sparse_to_mtx(vbrc_path: str, mtx_path: str) -> None:
    """Read a VBR-C file and write its CSR sparse part as a Matrix Market file.

    Args:
        vbrc_path: Path to the input .vbrc file.
        mtx_path:  Path to the output .mtx file.
    """
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbrc_path)

    if len(csr_val) == 0 or len(indptr) == 0:
        print(f"  Skipping {os.path.basename(vbrc_path)}: no sparse CSR part (ublocks empty)")
        return

    nrows = rpntr[-1]
    ncols = cpntr[-1]

    mat = csr_array((csr_val, indices, indptr), shape=(nrows, ncols))

    os.makedirs(os.path.dirname(mtx_path), exist_ok=True)
    mmwrite(mtx_path, mat)
    print(f"  Wrote {mtx_path}  (nnz={mat.nnz}, shape={mat.shape})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate .mtx files from the sparse CSR parts of VBR-C split files."
    )
    parser.add_argument(
        "matrices", nargs="*",
        help="Matrix names to process (without extension). If omitted, all .vbrc files "
             "in Generated_VBR_split/ are processed.",
    )
    parser.add_argument(
        "--input-dir", type=str, default=str(GENERATED_VBR_SPLIT_DIR),
        help=f"Input directory with .vbrc files (default: {GENERATED_VBR_SPLIT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(GENERATED_SPARSE_MTX_DIR),
        help=f"Output directory for .mtx files (default: {GENERATED_SPARSE_MTX_DIR})",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("Run benchmarks first to generate VBR-C split files.")
        sys.exit(1)

    # Collect .vbrc files to process
    if args.matrices:
        vbrc_files = []
        for name in args.matrices:
            path = os.path.join(input_dir, f"{name}.vbrc")
            if os.path.isfile(path):
                vbrc_files.append(path)
            else:
                print(f"Warning: {path} not found, skipping.")
    else:
        vbrc_files = sorted(
            str(p) for p in pathlib.Path(input_dir).glob("*.vbrc")
        )

    if not vbrc_files:
        print("No .vbrc files found to process.")
        sys.exit(0)

    print(f"Processing {len(vbrc_files)} VBR-C file(s) from {input_dir}")
    for vbrc_path in vbrc_files:
        name = pathlib.Path(vbrc_path).stem
        mtx_path = os.path.join(output_dir, f"{name}.mtx")
        vbrc_sparse_to_mtx(vbrc_path, mtx_path)

    print("Done.")


if __name__ == "__main__":
    main()
