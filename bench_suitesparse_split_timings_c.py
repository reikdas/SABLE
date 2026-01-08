#!/usr/bin/env python3
"""
Benchmark script for SABLE sparse matrix-vector multiplication.

This script:
1. Reads YAML files from find-submatrices/results/ to get dense block information
2. Downloads matrices from SuiteSparse
3. Converts to VBRC format
4. Generates C code for SpV8 and MKL versions
5. Compiles and runs the generated code
6. Collects timing information and writes to JSON files
7. Cleans up downloaded matrix files after processing
"""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import scipy
from scipy.io import mmread
from scipy.sparse import csc_matrix

# Add find-submatrices to path for importing
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "find-submatrices"))
from find_matrices import get_matrix_paths, cleanup_matrix_files, get_matrix_info
from ssgetpy import fetch

from src.codegen import gen_single_threaded_spmv_spv8, gen_single_threaded_spmv_mkl
from src.consts import CFLAGS, MKL_FLAGS
from utils.convert_real_to_vbr import convert_sparse_to_vbrc_with_blocks, _write_vbrc_file, analyze_dense_blocks
from utils.fileio import parse_yaml_blocks, write_dense_vector
from utils.utils import remove_outliers_deciles, set_ulimit

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

# Configuration
COMPILE_TIMEOUT = 60 * 60 * 4
DEFAULT_BENCH_ITERATIONS = 100

# Directories
SUITESPARSE_DIR = FILEPATH / "Suitesparse"
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"
GENERATED_VBR_SPLIT_DIR = FILEPATH / "Generated_VBR_split"
GENERATED_VBR_SPARSE_DIR = FILEPATH / "Generated_VBR_Sparse"
GENERATED_SPMV_SPV8_DIR = FILEPATH / "Generated_SpMV_C_split"
GENERATED_SPMV_MKL_DIR = FILEPATH / "Generated_SpMV_C_mkl"
GENERATED_SPMV_SPARSE_SPV8_DIR = FILEPATH / "Generated_SpMV_C_sparse"
GENERATED_SPMV_SPARSE_MKL_DIR = FILEPATH / "Generated_SpMV_C_sparse_mkl"


def compile_c_program(c_file_path: str, output_dir: str, use_mkl: bool = False) -> Optional[Tuple[str, float]]:
    """
    Compile the C program using gcc and return the executable path and compilation time.

    Returns:
        Tuple of (executable_path, compile_time_ns) or None if compilation fails
    """
    import time
    c_file = os.path.basename(c_file_path)
    output_name = os.path.splitext(c_file)[0]
    output_path = os.path.join(output_dir, output_name)
    
    # Compile with gcc, including MKL flags if needed
    print(f"  Compiling generated C code: {c_file} (output: {output_path})")
    if use_mkl:
        compile_cmd = ["gcc", c_file_path, "-o", output_path] + CFLAGS + MKL_FLAGS
    else:
        compile_cmd = ["gcc", c_file_path, "-o", output_path] + CFLAGS

    try:
        start_time = time.time_ns()
        result = subprocess.run(compile_cmd, cwd=output_dir, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
        compile_time_ns = time.time_ns() - start_time
        if result.returncode != 0:
            print(f"Compilation failed for {c_file}: {result.stderr}")
            return None
        print(f"  Finished compiling {c_file}. Starting benchmark runs...")
        return output_path, compile_time_ns
    except subprocess.TimeoutExpired:
        print(f"Compilation timeout for {c_file}")
        return None
    except Exception as e:
        print(f"Compilation error for {c_file}: {e}")
        return None


def eval_single_file_split_timings(
    fname: str,
    codegen_dir: str,
    bench_freq: int,
    extract_indiv_blocks: bool = True
) -> Tuple[float, float, Dict[int, float], float]:
    """
    Run the compiled benchmark and parse timing results.

    Returns:
        Tuple of (avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_ns)
    """
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)

    sparse_times = []
    dense_times = []
    individual_dense_block_times = {}  # Dictionary to store individual block timings
    
    # First, compile the C program
    c_file_path = os.path.join(codegen_dir, f"{fname}.c")
    
    # Determine if this is MKL based on directory name
    use_mkl = "mkl" in codegen_dir.lower()
    compile_result = compile_c_program(c_file_path, codegen_dir, use_mkl=use_mkl)

    if compile_result is None:
        print(f"Failed to compile {fname}, skipping evaluation")
        return 0, 0, {}, 0.0

    executable_path, compile_time_ns = compile_result
    
    print(f"  Executing benchmark binary: {executable_path}")
    for _ in range(bench_freq):
        try:
            output = subprocess.check_output(
                ["taskset", "-a", "-c", ",".join([str(x) for x in cpu_affinity]), executable_path],
                cwd=codegen_dir,
                preexec_fn=set_ulimit
            ).decode("utf-8").split("\n")
        except subprocess.CalledProcessError as e:
            print(f"Error running {fname}: {e}")
            continue
        
        # Skip warning lines if present
        start_idx = 0
        if len(output) > 0 and "warning" in output[0].lower():
            start_idx = 1
        
        # Extract Sparse and Dense timing from output
        if len(output[start_idx:]) >= 2:
            sparse_line = output[start_idx:][0]
            dense_line = output[start_idx:][1]
            
            # Handle sparse times (may be empty)
            sparse_values = []
            if sparse_line.startswith('Sparse: '):
                sparse_content = sparse_line[8:].strip()  # Remove 'Sparse: ' prefix
                if sparse_content:  # Only parse if there's content
                    sparse_values = [float(x.strip()) for x in sparse_content.rstrip(',').split(',') if x.strip()]
            
            # Handle dense times
            dense_values = []
            dense_match = re.search(r'Dense: (.+)', dense_line)
            if dense_match:
                dense_content = dense_match.group(1).strip().rstrip(',')
                if dense_content:
                    dense_values = [float(x.strip()) for x in dense_content.split(',') if x.strip()]
            
            # Add times if we found any
            if sparse_values or dense_values:
                sparse_times.extend(sparse_values)
                dense_times.extend(dense_values)
            
            # Extract individual dense block timings
            if extract_indiv_blocks:
                for line in output[start_idx:]:
                    dense_block_match = re.search(r'Dense Block (\d+): (.+)', line)
                    if dense_block_match:
                        block_id = int(dense_block_match.group(1))
                        block_times_str = dense_block_match.group(2).strip().rstrip(',')
                        if block_times_str:
                            block_times = [float(x.strip()) for x in block_times_str.split(',') if x.strip()]
                            
                            if block_id not in individual_dense_block_times:
                                individual_dense_block_times[block_id] = []
                            individual_dense_block_times[block_id].extend(block_times)
    
    # Remove outliers and calculate averages
    sparse_times = remove_outliers_deciles(sparse_times)
    dense_times = remove_outliers_deciles(dense_times)
    
    avg_sparse_time = statistics.mean(sparse_times) if sparse_times else 0
    avg_dense_time = statistics.mean(dense_times) if dense_times else 0
    
    # Calculate averages for individual dense blocks
    avg_individual_block_times = {}
    if extract_indiv_blocks:
        for block_id, times in individual_dense_block_times.items():
            times_clean = remove_outliers_deciles(times)
            avg_individual_block_times[block_id] = statistics.mean(times_clean) if times_clean else 0

    return avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_ns


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
    This is done once and shared between SpV8 and MKL benchmarks.
    
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
    
    # Write dense vector
    write_dense_vector(1.0, cpntr[-1])
    
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
    use_mkl: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Run benchmarks for either SpV8 or MKL using pre-converted VBRC data.
    
    Args:
        matrix_name: Name of the matrix
        split_vbrc_data: Pre-converted VBRC data with dense blocks
        sparse_vbrc_data: Pre-converted VBRC data without dense blocks
        matrix_rows, matrix_cols, matrix_nnz: Matrix dimensions
        bench_iterations: Number of benchmark iterations
        use_mkl: Whether to use MKL (True) or SpV8 (False)
    
    Returns:
        Dictionary with benchmark results
    """
    variant_name = "mkl" if use_mkl else "spv8"
    
    # Select directories and functions based on variant
    if use_mkl:
        codegen_dir = str(GENERATED_SPMV_MKL_DIR)
        sparse_codegen_dir = str(GENERATED_SPMV_SPARSE_MKL_DIR)
        gen_function = gen_single_threaded_spmv_mkl
    else:
        codegen_dir = str(GENERATED_SPMV_SPV8_DIR)
        sparse_codegen_dir = str(GENERATED_SPMV_SPARSE_SPV8_DIR)
        gen_function = gen_single_threaded_spmv_spv8
    
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
    
    # Generate C code for split version (codegen time)
    print(f"  [{variant_name}] Generating C code (split)...")
    codegen_time_split_ns = gen_function(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre,
        ublocks, indptr, indices, csr_val,
        codegen_dir, matrix_name, vbr_dir, bench=bench_iterations
    )
    
    # Evaluate split version (compile + run)
    print(f"  [{variant_name}] Evaluating split version...")
    avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_split_ns = \
        eval_single_file_split_timings(matrix_name, codegen_dir, bench_iterations)
    
    # Generate C code for sparse version (codegen time)
    print(f"  [{variant_name}] Generating C code (fully sparse)...")
    codegen_time_sparse_ns = gen_function(
        val_sparse, indx_sparse, bindx_sparse,
        rpntr_sparse, cpntr_sparse,
        bpntrb_sparse, bpntre_sparse,
        ublocks_sparse, indptr_sparse,
        indices_sparse, csr_val_sparse,
        sparse_codegen_dir, matrix_name,
        sparse_vbr_dir, bench=bench_iterations
    )
    
    # Evaluate fully sparse version (compile + run)
    print(f"  [{variant_name}] Evaluating fully sparse version...")
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
        }
    
    return matrix_result


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(
        description="Benchmark SABLE sparse matrix-vector multiplication",
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
    parser.add_argument("--skip-spv8", action="store_true", help="Skip SpV8 benchmarks")
    parser.add_argument("--skip-mkl", action="store_true", help="Skip MKL benchmarks")
    
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
    
    spv8_results = []
    mkl_results = []
    
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
            
            # Convert to VBRC format once (shared between SpV8 and MKL)
            print(f"\n  === Converting to VBRC format ===")
            split_vbrc_data, sparse_vbrc_data = convert_and_prepare_vbrc(
                matrix_name, dense_block_coords, A, matrix_nnz
            )
            dense_blocks = split_vbrc_data["dense_blocks"]
            
            # Benchmark with SpV8
            if not args.skip_spv8:
                print(f"\n  === Running SpV8 benchmark ===")
                spv8_result = process_and_benchmark_matrix(
                    matrix_name, split_vbrc_data, sparse_vbrc_data,
                    matrix_rows, matrix_cols, matrix_nnz,
                    args.bench, use_mkl=False
                )
                if spv8_result:
                    # Update or append result
                    existing_idx = next((i for i, r in enumerate(spv8_results) if r["matrix_name"] == matrix_name), None)
                    if existing_idx is not None:
                        spv8_results[existing_idx] = spv8_result
                        print(f"  [spv8] Updated existing result for {matrix_name}")
                    else:
                        spv8_results.append(spv8_result)
                        print(f"  [spv8] Added new result for {matrix_name}")
                    
                    # Write intermediate results
                    spv8_output = output_dir / "sable_spmv_spv8.json"
                    with open(spv8_output, 'w') as f:
                        json.dump(spv8_results, f, indent=2)
                    print(f"  [spv8] Results written to {spv8_output}")
            
            # Benchmark with MKL
            if not args.skip_mkl:
                print(f"\n  === Running MKL benchmark ===")
                mkl_result = process_and_benchmark_matrix(
                    matrix_name, split_vbrc_data, sparse_vbrc_data,
                    matrix_rows, matrix_cols, matrix_nnz,
                    args.bench, use_mkl=True
                )
                if mkl_result:
                    # Update or append result
                    existing_idx = next((i for i, r in enumerate(mkl_results) if r["matrix_name"] == matrix_name), None)
                    if existing_idx is not None:
                        mkl_results[existing_idx] = mkl_result
                        print(f"  [mkl] Updated existing result for {matrix_name}")
                    else:
                        mkl_results.append(mkl_result)
                        print(f"  [mkl] Added new result for {matrix_name}")
                    
                    # Write intermediate results
                    mkl_output = output_dir / "sable_spmv_mkl.json"
                    with open(mkl_output, 'w') as f:
                        json.dump(mkl_results, f, indent=2)
                    print(f"  [mkl] Results written to {mkl_output}")
            
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
    
    if spv8_results:
        print(f"\nSpV8 Results ({len(spv8_results)} matrices):")
        for result in spv8_results:
            num_dense_blocks = len(result['individual_dense_block_timings'])
            print(f"  {result['matrix_name']}: {num_dense_blocks} dense blocks, "
                  f"sparse: {result['timing']['sparse_time_ns']:.0f}ns, "
                  f"dense: {result['timing']['dense_time_ns']:.0f}ns, "
                  f"speedup: {result['timing']['speedup']:.3f}x")
    
    if mkl_results:
        print(f"\nMKL Results ({len(mkl_results)} matrices):")
        for result in mkl_results:
            num_dense_blocks = len(result['individual_dense_block_timings'])
            print(f"  {result['matrix_name']}: {num_dense_blocks} dense blocks, "
                  f"sparse: {result['timing']['sparse_time_ns']:.0f}ns, "
                  f"dense: {result['timing']['dense_time_ns']:.0f}ns, "
                  f"speedup: {result['timing']['speedup']:.3f}x")
    
    print(f"\nResults written to {output_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())
