#!/usr/bin/env python3
"""
Benchmark script for SABLE sparse matrix-matrix multiplication (SpMM).

This script:
1. Reads YAML files from find-submatrices/results/ to get dense block information
2. Downloads matrices from SuiteSparse
3. Converts to VBRC format
4. Generates C code for SpMM
5. Compiles and runs the generated code
6. Collects timing information and writes to JSON files
7. Cleans up downloaded matrix files after processing
"""

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import scipy
from scipy.io import mmread
from scipy.sparse import csc_matrix

# Add find-submatrices to path for importing
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "find-submatrices"))
from find_matrices import get_matrix_paths, cleanup_matrix_files, get_matrix_info
from ssgetpy import fetch

from src.codegen import gen_single_threaded_spmm
from src.consts import CFLAGS
from utils.convert_real_to_vbr import convert_sparse_to_vbrc_with_blocks, _write_vbrc_file, analyze_dense_blocks
from utils.fileio import parse_yaml_blocks, write_dense_matrix
from utils.utils import (
    estimate_total_speedup,
    estimate_dense_speedup,
)

# Import benchmark utilities from SpMV benchmark
from bench_suitesparse_split_timings_c import (
    compile_c_program,
    eval_single_file_split_timings,
)

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

# Configuration
COMPILE_TIMEOUT = 60 * 60 * 4
DEFAULT_BENCH_ITERATIONS = 50

# Directories
SUITESPARSE_DIR = FILEPATH / "Suitesparse"
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"
GENERATED_VBR_SPLIT_DIR = FILEPATH / "Generated_VBR_split"
GENERATED_VBR_SPARSE_DIR = FILEPATH / "Generated_VBR_Sparse"
GENERATED_SPMM_DIR = FILEPATH / "Generated_SpMV_C_split"
GENERATED_SPMM_SPARSE_DIR = FILEPATH / "Generated_SpMM_C_sparse"


def download_matrix_from_suitesparse(matrix_name: str) -> Optional[Tuple[str, Any, Optional[str], Optional[str]]]:
    """
    Download a matrix from SuiteSparse and return the path to the .mtx file, matrix_info, and cleanup paths.
    Returns None if the download fails.
    
    Returns:
        Tuple of (matrix_path, matrix_info, tar_path, matrix_subdir) for cleanup
    """
    # Ensure Suitesparse directory exists
    SUITESPARSE_DIR.mkdir(exist_ok=True)
    
    # Set environment variable to control download location
    original_ssgetpy_dir = os.environ.get('SSGETPY_DIR')
    try:
        os.environ['SSGETPY_DIR'] = str(SUITESPARSE_DIR)
        matrix_info = get_matrix_info(matrix_name)
        if matrix_info is None:
            return None
        
        # Download the matrix
        fetch(matrix_name)
        
        # Get paths - but check if file exists in our target directory first
        tar_path, tar_dir, matrix_subdir, matrix_path = get_matrix_paths(matrix_info)
        
        # If the file doesn't exist at the expected location, try to find it in SUITESPARSE_DIR
        if not os.path.exists(matrix_path):
            # Try to find it in SUITESPARSE_DIR
            expected_path = SUITESPARSE_DIR / matrix_info.name / f"{matrix_info.name}.mtx"
            if expected_path.exists():
                matrix_path = str(expected_path)
                matrix_subdir = str(expected_path.parent)
                tar_dir = str(SUITESPARSE_DIR)
                # Look for tar file in SUITESPARSE_DIR
                tar_candidates = list(SUITESPARSE_DIR.glob(f"{matrix_info.name}*.tar.gz"))
                tar_path = str(tar_candidates[0]) if tar_candidates else None
            else:
                print(f"Error: Matrix file not found at {matrix_path} or {expected_path}")
                return None
        
        return matrix_path, matrix_info, tar_path, matrix_subdir
    finally:
        # Restore original environment variable
        if original_ssgetpy_dir is not None:
            os.environ['SSGETPY_DIR'] = original_ssgetpy_dir
        elif 'SSGETPY_DIR' in os.environ:
            del os.environ['SSGETPY_DIR']


def get_available_matrices() -> List[str]:
    """Get list of matrices with YAML files in find-submatrices/results/."""
    yaml_files = list(RESULTS_DIR.glob("*.yaml"))
    return [f.stem for f in yaml_files]


def convert_and_prepare_vbrc(
    matrix_name: str,
    dense_block_coords: List[Tuple[int, int, int, int]],
    mat: scipy.sparse.spmatrix,
    matrix_nnz: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert matrix to VBRC format (both split and sparse versions).
    This is done once and shared between benchmarks.
    
    Returns:
        Tuple of (split_vbrc_data, sparse_vbrc_data) dictionaries containing VBRC data
    """
    vbr_dir = str(GENERATED_VBR_SPLIT_DIR / matrix_name)
    sparse_vbr_dir = str(GENERATED_VBR_SPARSE_DIR / matrix_name)
    
    # Ensure directories exist
    os.makedirs(vbr_dir, exist_ok=True)
    os.makedirs(sparse_vbr_dir, exist_ok=True)
    
    # Convert to VBRC format with dense blocks
    print(f"  Converting to VBRC format (split)...")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(mat, dense_block_coords)
    
    # Write VBRC file
    _write_vbrc_file(matrix_name, vbr_dir, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    
    # Analyze dense blocks
    dense_blocks = analyze_dense_blocks(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks)
    
    # Write dense matrix (512 columns for SpMM)
    write_dense_matrix(1.0, cpntr[-1], 512)
    
    if len(val) == 0 and len(dense_block_coords) > 0:
        print(f"  Warning: No dense blocks found after conversion")
    
    split_data = {
        "val": val, "indx": indx, "bindx": bindx, "rpntr": rpntr, "cpntr": cpntr,
        "bpntrb": bpntrb, "bpntre": bpntre, "ublocks": ublocks,
        "indptr": indptr, "indices": indices, "csr_val": csr_val,
        "dense_blocks": dense_blocks, "vbr_dir": vbr_dir
    }
    
    # Convert to fully sparse VBRC format (no dense blocks)
    print(f"  Converting to VBRC format (fully sparse)...")
    val_sparse, indx_sparse, bindx_sparse, rpntr_sparse, cpntr_sparse, \
        bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, \
        indices_sparse, csr_val_sparse = convert_sparse_to_vbrc_with_blocks(mat, [])
    
    # Write sparse VBRC file
    _write_vbrc_file(matrix_name, sparse_vbr_dir, val_sparse, indx_sparse, bindx_sparse, 
                     rpntr_sparse, cpntr_sparse, bpntrb_sparse, bpntre_sparse, 
                     ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse)
    
    # Assert that the sparse variant has no dense blocks
    assert len(val_sparse) == 0, f"Expected fully sparse variant for {matrix_name}, but found {len(val_sparse)} dense blocks"
    
    sparse_data = {
        "val": val_sparse, "indx": indx_sparse, "bindx": bindx_sparse, 
        "rpntr": rpntr_sparse, "cpntr": cpntr_sparse,
        "bpntrb": bpntrb_sparse, "bpntre": bpntre_sparse, "ublocks": ublocks_sparse,
        "indptr": indptr_sparse, "indices": indices_sparse, "csr_val": csr_val_sparse,
        "vbr_dir": sparse_vbr_dir
    }
    
    return split_data, sparse_data


def process_and_benchmark_matrix(
    matrix_name: str,
    split_vbrc_data: Dict[str, Any],
    sparse_vbrc_data: Dict[str, Any],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    bench_iterations: int,
) -> Optional[Dict[str, Any]]:
    """
    Run SpMM benchmarks using pre-converted VBRC data.
    
    Args:
        matrix_name: Name of the matrix
        split_vbrc_data: Pre-converted VBRC data with dense blocks
        sparse_vbrc_data: Pre-converted VBRC data without dense blocks
        matrix_rows, matrix_cols, matrix_nnz: Matrix dimensions
        bench_iterations: Number of benchmark iterations
    
    Returns:
        Dictionary with benchmark results
    """
    codegen_dir = str(GENERATED_SPMM_DIR)
    sparse_codegen_dir = str(GENERATED_SPMM_SPARSE_DIR)
    
    # Ensure directories exist
    os.makedirs(codegen_dir, exist_ok=True)
    os.makedirs(sparse_codegen_dir, exist_ok=True)
    
    # Extract VBRC data
    val = split_vbrc_data["val"]
    indx = split_vbrc_data["indx"]
    bindx = split_vbrc_data["bindx"]
    rpntr = split_vbrc_data["rpntr"]
    cpntr = split_vbrc_data["cpntr"]
    bpntrb = split_vbrc_data["bpntrb"]
    bpntre = split_vbrc_data["bpntre"]
    ublocks = split_vbrc_data["ublocks"]
    indptr = split_vbrc_data["indptr"]
    indices = split_vbrc_data["indices"]
    csr_val = split_vbrc_data["csr_val"]
    dense_blocks = split_vbrc_data["dense_blocks"]
    vbr_dir = split_vbrc_data["vbr_dir"]
    
    val_sparse = sparse_vbrc_data["val"]
    indx_sparse = sparse_vbrc_data["indx"]
    bindx_sparse = sparse_vbrc_data["bindx"]
    rpntr_sparse = sparse_vbrc_data["rpntr"]
    cpntr_sparse = sparse_vbrc_data["cpntr"]
    bpntrb_sparse = sparse_vbrc_data["bpntrb"]
    bpntre_sparse = sparse_vbrc_data["bpntre"]
    ublocks_sparse = sparse_vbrc_data["ublocks"]
    indptr_sparse = sparse_vbrc_data["indptr"]
    indices_sparse = sparse_vbrc_data["indices"]
    csr_val_sparse = sparse_vbrc_data["csr_val"]
    sparse_vbr_dir = sparse_vbrc_data["vbr_dir"]
    
    # Generate C code for split version
    print(f"  Generating C code (split)...")
    gen_single_threaded_spmm(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre,
        ublocks, indptr, indices, csr_val,
        codegen_dir, matrix_name, vbr_dir, bench=bench_iterations
    )
    
    # Evaluate split version (compile + run)
    print(f"  Evaluating split version...")
    avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_split_ns = \
        eval_single_file_split_timings(matrix_name, codegen_dir, bench_iterations)
    
    # Generate C code for sparse version
    print(f"  Generating C code (fully sparse)...")
    gen_single_threaded_spmm(
        val_sparse, indx_sparse, bindx_sparse,
        rpntr_sparse, cpntr_sparse,
        bpntrb_sparse, bpntre_sparse,
        ublocks_sparse, indptr_sparse,
        indices_sparse, csr_val_sparse,
        sparse_codegen_dir, matrix_name,
        sparse_vbr_dir, bench=bench_iterations
    )
    
    # Evaluate fully sparse version (compile + run)
    print(f"  Evaluating fully sparse version...")
    sparse_avg_sparse_time, _, _, compile_time_sparse_ns = eval_single_file_split_timings(
        matrix_name, sparse_codegen_dir, bench_iterations
    )
    
    # Calculate percentages
    total_time = avg_sparse_time + avg_dense_time
    sparse_percentage = (avg_sparse_time / total_time * 100) if total_time > 0 else 0
    dense_percentage = (avg_dense_time / total_time * 100) if total_time > 0 else 0
    speedup = (sparse_avg_sparse_time / total_time) if total_time > 0 else 0
    
    # Calculate nnz statistics
    dense_all = sum(block.get("rows", 0) * block.get("cols", 0) for block in dense_blocks)
    dense_nnz = sum(block.get("nnz", 0) for block in dense_blocks)
    sparse_nnz = matrix_nnz - dense_nnz
    extra_zeros = dense_all - dense_nnz
    dense_nnz_perc = (dense_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
    sparse_nnz_perc = (sparse_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
    
    density_calculation = matrix_nnz / (matrix_rows * matrix_cols) if matrix_rows * matrix_cols > 0 else 0
    
    # Build result
    matrix_result = {
        "matrix_name": matrix_name,
        "matrix_dimensions": {
            "rows": matrix_rows,
            "cols": matrix_cols,
            "nnz": matrix_nnz,
            # Rounded to 3 decimal places
            "density": round(density_calculation, 3),
        },
        "timing": {
            # Rounded to 2 decimal places
            "sparse_time_ns": round(avg_sparse_time, 2),
            "dense_time_ns": round(avg_dense_time, 2),
            "total_time_ns": round(total_time, 2),
            "sparse_percentage": round(sparse_percentage, 3),
            "dense_percentage": round(dense_percentage, 3),
            "fully_sparse_time": sparse_avg_sparse_time,
            "speedup": speedup,
            "expected_sparse_time_ns": ((100-dense_nnz_perc)/100)*sparse_avg_sparse_time,
            "dense_if_sparse_time_ns": sparse_avg_sparse_time - ((100-dense_nnz_perc)/100)*sparse_avg_sparse_time,
            # Compilation times (calling gcc) - stored in seconds
            "compile_time_split_s": compile_time_split_ns / 1e9 if compile_time_split_ns else 0.0,
            "compile_time_sparse_s": compile_time_sparse_ns / 1e9 if compile_time_sparse_ns else 0.0,
        },
        "nnz": {
            "sparse_nnz": sparse_nnz,
            "dense_all": dense_all,
            "dense_nnz": dense_nnz,
            "extra_zeros": extra_zeros,
            # Rounded to 2 decimal places
            "dense_nnz_perc": round(dense_nnz_perc, 2),
            "sparse_nnz_perc": round(sparse_nnz_perc, 2),
        },
        "individual_dense_block_timings": {}
    }
    
    # Add individual dense block timings and merge with dense block analysis
    for block_id, block_time in avg_individual_block_times.items():
        # Find corresponding dense block info (block_id is 1-indexed, dense_blocks is 0-indexed)
        block_info = dense_blocks[block_id - 1] if block_id - 1 < len(dense_blocks) else {}
        
        block_nnz = block_info.get("nnz", 0)
        dense_nnz_sum = sum(block.get("nnz", 0) for block in dense_blocks)
        matrix_result["individual_dense_block_timings"][f"block_{block_id}"] = {
            # Rounded to 2 decimal places
            "time_ns": round(block_time, 2),
            "percentage_of_total_time": round((block_time / total_time * 100), 2) if total_time > 0 else 0,
            "percentage_of_dense_time": round((block_time / avg_dense_time * 100), 2) if avg_dense_time > 0 else 0,
            # Rounded to 3 decimal places
            "percentage_of_total_nnz": round((block_nnz / matrix_nnz * 100), 3) if matrix_nnz > 0 else 0,
            "percentage_of_dense_nnz": round((block_nnz / dense_nnz_sum * 100), 3) if dense_nnz_sum > 0 else 0,
            "rows": block_info.get("rows", 0),
            "cols": block_info.get("cols", 0),
            # Rounded to 3 decimal places
            "density_percent": round(block_info.get("density_percent", 0), 3),
            "nnz": block_nnz,
            "predicted_speedup": 0
        }
    
    # Calculate estimated speedup based on individual block characteristics
    estimated_speedup = estimate_total_speedup(matrix_result["individual_dense_block_timings"])
    estimated_dense_speedup = estimate_dense_speedup(matrix_result["individual_dense_block_timings"])
    
    # Add the estimated speedup to the timing section
    matrix_result["timing"]["expected_speedup"] = estimated_speedup
    matrix_result["timing"]["expected_dense_speedup"] = estimated_dense_speedup
    
    return matrix_result


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(
        description="Benchmark SABLE sparse matrix-matrix multiplication (SpMM)",
        epilog="Examples:\n"
               "  %(prog)s eris1176 bloweybl heart1\n"
               "  %(prog)s --matrices eris1176 bloweybl\n"
               "  %(prog)s  # processes all available matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "matrices",
        nargs="*",
        help="Matrix names to benchmark (positional arguments). If not provided, all available matrices are processed."
    )
    parser.add_argument("--matrices", dest="matrices_flag", nargs="*", metavar="MATRIX",
                        help="Matrix names to benchmark (alternative to positional arguments)")
    parser.add_argument("--bench", type=int, default=DEFAULT_BENCH_ITERATIONS, 
                        help=f"Number of benchmark iterations (default: {DEFAULT_BENCH_ITERATIONS})")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results (default: results)")
    
    args = parser.parse_args()
    
    # Get matrices to process - merge positional arguments and --matrices flag, or use all
    matrices = args.matrices or args.matrices_flag
    if not matrices:
        matrices = get_available_matrices()
    
    print(f"Will process {len(matrices)} matrices")
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Ensure required directories exist
    SUITESPARSE_DIR.mkdir(exist_ok=True)
    GENERATED_VBR_SPLIT_DIR.mkdir(exist_ok=True)
    GENERATED_VBR_SPARSE_DIR.mkdir(exist_ok=True)
    
    all_results = []
    
    for matrix_name in matrices:
        yaml_path = RESULTS_DIR / f"{matrix_name}.yaml"
        
        if not yaml_path.exists():
            print(f"Warning: YAML file not found for {matrix_name}, skipping")
            continue
        
        print(f"\nProcessing {matrix_name}...")
        
        # Download matrix from SuiteSparse
        print(f"  Downloading matrix from SuiteSparse...")
        download_result = download_matrix_from_suitesparse(matrix_name)
        if download_result is None:
            print(f"  Failed to download {matrix_name}, skipping")
            continue
        
        mtx_path, matrix_info, tar_path, matrix_subdir = download_result
        
        try:
            # Load the matrix
            print(f"  Loading matrix from {mtx_path}...")
            A = mmread(mtx_path)
            A = csc_matrix(A, copy=False)
            
            matrix_rows = A.shape[0]
            matrix_cols = A.shape[1]
            matrix_nnz = A.nnz
            
            print(f"  Matrix shape: {matrix_rows} x {matrix_cols}, NNZ: {matrix_nnz}")
            
            # Parse dense blocks from YAML
            print(f"  Parsing dense blocks from {yaml_path}...")
            dense_block_coords = parse_yaml_blocks(str(yaml_path))
            print(f"  Found {len(dense_block_coords)} dense blocks")
            
            # Convert to VBRC format
            print(f"\n  === Converting to VBRC format ===")
            split_vbrc_data, sparse_vbrc_data = convert_and_prepare_vbrc(
                matrix_name, dense_block_coords, A, matrix_nnz
            )
            dense_blocks = split_vbrc_data["dense_blocks"]
            
            # Benchmark SpMM
            print(f"\n  === Running SpMM benchmark ===")
            result = process_and_benchmark_matrix(
                matrix_name, split_vbrc_data, sparse_vbrc_data,
                matrix_rows, matrix_cols, matrix_nnz,
                args.bench
            )
            if result:
                # Update or append result
                existing_idx = next((i for i, r in enumerate(all_results) if r["matrix_name"] == matrix_name), None)
                if existing_idx is not None:
                    all_results[existing_idx] = result
                    print(f"  Updated existing result for {matrix_name}")
                else:
                    all_results.append(result)
                    print(f"  Added new result for {matrix_name}")
                
                # Write intermediate results
                output_file = output_dir / "sable_spmm.json"
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"  Results written to {output_file}")
            
            print(f"\nCompleted processing {matrix_name}")
            
        except Exception as e:
            print(f"  Error processing {matrix_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up downloaded matrix files
            if 'matrix_info' in locals() and matrix_info is not None:
                print(f"  Cleaning up downloaded files for {matrix_name}...")
                if tar_path is not None or matrix_subdir is not None:
                    cleanup_matrix_files(tar_path, matrix_subdir)
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    
    if all_results:
        print(f"\nSpMM Results ({len(all_results)} matrices):")
        for result in all_results:
            num_dense_blocks = len(result['individual_dense_block_timings'])
            print(f"  {result['matrix_name']}: {num_dense_blocks} dense blocks, "
                  f"sparse: {result['timing']['sparse_time_ns']:.0f}ns, "
                  f"dense: {result['timing']['dense_time_ns']:.0f}ns, "
                  f"speedup: {result['timing']['speedup']:.3f}x")
    
    print(f"\nResults written to {output_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())
