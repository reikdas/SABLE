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

from src.codegen import (
    gen_single_threaded_spmm_naive_naive,
    gen_single_threaded_spmm_naive_spreg,
    gen_single_threaded_spmm_mkl_naive,
    gen_single_threaded_spmm_mkl_spreg,
)
from src.consts import CFLAGS
from utils.convert_real_to_vbr import convert_sparse_to_vbrc_with_blocks, _write_vbrc_file, analyze_dense_blocks
from utils.fileio import parse_yaml_blocks, write_dense_matrix
from utils.utils import set_ulimit

# Import benchmark utilities from SpMV benchmark
from bench_suitesparse_split_timings_c import (
    eval_single_file_split_timings,
)

# Import compilation function - we'll override it for spreg
import subprocess
import time
from src.consts import CFLAGS, MKL_FLAGS

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

# Configuration
COMPILE_TIMEOUT = 60 * 60 * 4
DEFAULT_BENCH_ITERATIONS = 10

# Directories
SUITESPARSE_DIR = FILEPATH / "Suitesparse"
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"
GENERATED_VBR_SPLIT_DIR = FILEPATH / "Generated_VBR_split"
GENERATED_VBR_SPARSE_DIR = FILEPATH / "Generated_VBR_Sparse"
GENERATED_SPMM_NAIVE_NAIVE_DIR = FILEPATH / "Generated_SpMM_C_naive_naive"
GENERATED_SPMM_NAIVE_SPREG_DIR = FILEPATH / "Generated_SpMM_C_naive_spreg"
GENERATED_SPMM_MKL_NAIVE_DIR = FILEPATH / "Generated_SpMM_C_mkl_naive"
GENERATED_SPMM_MKL_SPREG_DIR = FILEPATH / "Generated_SpMM_C_mkl_spreg"

# Paths for sparse-register-tiling
SPREG_BASE = FILEPATH / "sparse-register-tiling" / "spmm_nano_kernels"
SPREG_INCLUDE_DIR = SPREG_BASE / "include"
SPREG_SRC_DIR = SPREG_BASE / "src"
SPREG_WRAPPER_CPP = SPREG_SRC_DIR / "spmm_spreg_wrapper.cpp"


def eval_single_file_split_timings_spreg(
    fname: str,
    codegen_dir: str,
    bench_freq: int,
    extract_indiv_blocks: bool = True
) -> Tuple[float, float, Dict[int, float], float]:
    """
    Run the compiled benchmark and parse timing results for spreg version.
    Uses custom compile function that links with sparse-register-tiling.

    Returns:
        Tuple of (avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_ns)
    """
    from bench_suitesparse_split_timings_c import remove_outliers_deciles
    import statistics
    import re
    
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)

    sparse_times = []
    dense_times = []
    individual_dense_block_times = {}  # Dictionary to store individual block timings
    
    # First, compile the C program using spreg compile function
    c_file_path = os.path.join(codegen_dir, f"{fname}.c")
    compile_result = compile_c_program_spreg(c_file_path, codegen_dir)

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
        
        # Extract Sparse, Dense, and individual block timings by searching ALL lines
        # (not assuming specific positions, since output may have extra lines)
        for line in output:
            # Handle sparse times
            if line.startswith('Sparse: '):
                sparse_content = line[8:].strip()  # Remove 'Sparse: ' prefix
                if sparse_content:
                    sparse_values = [float(x.strip()) for x in sparse_content.rstrip(',').split(',') if x.strip()]
                    sparse_times.extend(sparse_values)
            
            # Handle aggregate dense times (line starts with "Dense: " but NOT "Dense Block")
            elif line.startswith('Dense: ') and not line.startswith('Dense Block'):
                dense_content = line[7:].strip().rstrip(',')  # Remove 'Dense: ' prefix
                if dense_content:
                    dense_values = [float(x.strip()) for x in dense_content.split(',') if x.strip()]
                    dense_times.extend(dense_values)
            
            # Extract individual dense block timings
            elif extract_indiv_blocks:
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


def get_spreg_source_files():
    """
    Get all the source files needed for sparse-register-tiling compilation.
    Includes core source files and generated factory files for AVX512.
    """
    import glob
    
    # Core source files from src/
    core_src_files = [
        str(SPREG_SRC_DIR / "cake_block_dims.cpp"),
        str(SPREG_SRC_DIR / "ExecutorFactory.cpp"),
        str(SPREG_SRC_DIR / "kernel_mapping.cpp"),
        str(SPREG_SRC_DIR / "mapping_io.cpp"),
        str(SPREG_SRC_DIR / "packing.cpp"),
        str(SPREG_SRC_DIR / "utils" / "misc.cpp"),
        str(SPREG_WRAPPER_CPP),
    ]
    
    # Generated source files (top-level)
    generated_dir = SPREG_BASE / "generated"
    generated_files = glob.glob(str(generated_dir / "*.cpp"))
    
    # Generated AVX512 factory files
    avx512_factory_files = glob.glob(str(generated_dir / "AVX512" / "factories" / "*" / "*.cpp"))
    
    return core_src_files + generated_files + avx512_factory_files


def get_spreg_include_dirs():
    """
    Get all include directories needed for sparse-register-tiling.
    """
    return [
        str(SPREG_INCLUDE_DIR),
        str(SPREG_BASE / "third_party" / "version2" / "vectorclass"),
        str(SPREG_BASE / "third_party" / "rte"),
        str(SPREG_BASE / "generated"),
        str(SPREG_BASE / "generated" / "AVX512" / "include"),
    ]


# Cache compiled object files to speed up subsequent builds
_spreg_obj_cache = {}
_spreg_obj_cache_dir = None


def compile_c_program_spreg(c_file_path: str, output_dir: str) -> Optional[Tuple[str, float]]:
    """
    Compile the C program that uses sparse-register-tiling wrapper.
    Uses gcc to compile C code, g++ to compile C++ wrapper, then g++ to link.
    Caches compiled object files for the spreg library to speed up subsequent builds.
    
    Returns:
        Tuple of (executable_path, compile_time_ns) or None if compilation fails
    """
    global _spreg_obj_cache, _spreg_obj_cache_dir
    
    c_file = os.path.basename(c_file_path)
    output_name = os.path.splitext(c_file)[0]
    output_path = os.path.join(output_dir, output_name)
    
    # Get source files and include directories
    spreg_src_files = get_spreg_source_files()
    include_dirs = get_spreg_include_dirs()
    
    # Build include flags
    include_flags = []
    for inc_dir in include_dirs:
        include_flags.extend(["-I", inc_dir])
    
    # Common C/C++ flags
    common_flags = ["-O3", "-march=native", "-fopenmp"]
    
    # C++ specific flags - ENABLE_AVX512 is critical for the factory registration
    cpp_flags = ["-std=c++17", "-DRTE_CACHE_LINE_SIZE=64", "-DENABLE_AVX512", "-mavx512f"]
    
    # C flags from CFLAGS (filter out incompatible ones)
    c_flags = []
    for flag in CFLAGS:
        if flag.startswith("-I") or flag.startswith("-L"):
            c_flags.append(flag)
        elif flag in ["-O3", "-march=native", "-funroll-all-loops", 
                      "-mprefer-vector-width=512", "-mavx", "-ffast-math"]:
            c_flags.append(flag)
        elif not flag.endswith(".o"):  # Skip object files that are C-specific
            c_flags.append(flag)
    
    print(f"  Compiling generated C code: {c_file} (output: {output_path})")
    
    # Create object file cache directory
    obj_cache_dir = FILEPATH / ".spreg_obj_cache"
    obj_cache_dir.mkdir(exist_ok=True)
    
    try:
        start_time = time.time_ns()
        
        # Compile C file with gcc to object file
        c_obj = os.path.join(output_dir, f"{output_name}_c.o")
        gcc_cmd = ["gcc", "-c", c_file_path, "-o", c_obj] + c_flags + include_flags
        result = subprocess.run(gcc_cmd, cwd=output_dir, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
        if result.returncode != 0:
            print(f"Compilation failed for {c_file} (C compilation): {result.stderr}")
            return None
        
        # Compile C++ files with g++ to object files (with caching)
        cpp_objs = []
        
        # Check if cache needs rebuilding
        need_rebuild_cache = _spreg_obj_cache_dir != str(obj_cache_dir) or not _spreg_obj_cache
        
        for cpp_file in spreg_src_files:
            if not os.path.exists(cpp_file):
                print(f"Warning: C++ source file not found: {cpp_file}")
                continue
            
            # Use cache directory for spreg library objects
            cpp_basename = os.path.basename(cpp_file).replace(".cpp", ".o")
            # Make unique name based on path to avoid collisions
            cpp_hash = str(hash(cpp_file))[-8:]
            cpp_obj_name = f"{cpp_hash}_{cpp_basename}"
            cpp_obj = str(obj_cache_dir / cpp_obj_name)
            cpp_objs.append(cpp_obj)
            
            # Check if object file exists and is newer than source
            if not need_rebuild_cache and os.path.exists(cpp_obj):
                src_mtime = os.path.getmtime(cpp_file)
                obj_mtime = os.path.getmtime(cpp_obj)
                if obj_mtime > src_mtime:
                    continue  # Use cached object
            
            # Compile the C++ file
            gpp_cmd = ["g++", "-c", cpp_file, "-o", cpp_obj] + common_flags + cpp_flags + include_flags
            result = subprocess.run(gpp_cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            if result.returncode != 0:
                print(f"Compilation failed for {os.path.basename(cpp_file)}: {result.stderr}")
                return None
        
        _spreg_obj_cache_dir = str(obj_cache_dir)
        
        # Link everything with g++
        link_cmd = ["g++", "-o", output_path, c_obj] + cpp_objs + common_flags + ["-fopenmp", "-lstdc++fs", "-lpthread"]
        result = subprocess.run(link_cmd, cwd=output_dir, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
        compile_time_ns = time.time_ns() - start_time
        if result.returncode != 0:
            print(f"Compilation failed for {c_file} (linking): {result.stderr}")
            return None
        
        print(f"  Finished compiling {c_file}. Starting benchmark runs...")
        return output_path, compile_time_ns
    except subprocess.TimeoutExpired:
        print(f"Compilation timeout for {c_file}")
        return None
    except Exception as e:
        print(f"Compilation error for {c_file}: {e}")
        return None


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


def process_and_benchmark_matrix_naive_naive(
    matrix_name: str,
    split_vbrc_data: Dict[str, Any],
    sparse_vbrc_data: Dict[str, Any],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    bench_iterations: int,
) -> Optional[Dict[str, Any]]:
    """
    Run SpMM benchmarks using naive dense + naive sparse implementation.

    Args:
        matrix_name: Name of the matrix
        split_vbrc_data: Pre-converted VBRC data with dense blocks
        sparse_vbrc_data: Pre-converted VBRC data without dense blocks
        matrix_rows, matrix_cols, matrix_nnz: Matrix dimensions
        bench_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    codegen_dir_split = str(GENERATED_SPMM_NAIVE_NAIVE_DIR / "split")
    codegen_dir_sparse = str(GENERATED_SPMM_NAIVE_NAIVE_DIR / "sparse")
    
    # Ensure directories exist
    os.makedirs(codegen_dir_split, exist_ok=True)
    os.makedirs(codegen_dir_sparse, exist_ok=True)
    
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
    print(f"  [naive_naive] Generating C code (split)...")
    gen_single_threaded_spmm_naive_naive(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre,
        ublocks, indptr, indices, csr_val,
        codegen_dir_split, matrix_name, vbr_dir, bench=bench_iterations
    )

    # Evaluate split version (compile + run)
    print(f"  [naive_naive] Evaluating split version...")
    avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_split_ns = \
        eval_single_file_split_timings(matrix_name, codegen_dir_split, bench_iterations)

    # Generate C code for sparse version
    print(f"  [naive_naive] Generating C code (fully sparse)...")
    gen_single_threaded_spmm_naive_naive(
        val_sparse, indx_sparse, bindx_sparse,
        rpntr_sparse, cpntr_sparse,
        bpntrb_sparse, bpntre_sparse,
        ublocks_sparse, indptr_sparse,
        indices_sparse, csr_val_sparse,
        codegen_dir_sparse, matrix_name,
        sparse_vbr_dir, bench=bench_iterations
    )

    # Evaluate fully sparse version (compile + run)
    print(f"  [naive_naive] Evaluating fully sparse version...")
    sparse_avg_sparse_time, _, _, compile_time_sparse_ns = eval_single_file_split_timings(
        matrix_name, codegen_dir_sparse, bench_iterations
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
            "speedup": round(speedup, 3),
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
            "nnz": block_nnz
        }
    
    return matrix_result


def process_and_benchmark_matrix_mkl_naive(
    matrix_name: str,
    split_vbrc_data: Dict[str, Any],
    sparse_vbrc_data: Dict[str, Any],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    bench_iterations: int,
) -> Optional[Dict[str, Any]]:
    """
    Run SpMM benchmarks using MKL dense + naive sparse implementation.

    Args:
        matrix_name: Name of the matrix
        split_vbrc_data: Pre-converted VBRC data with dense blocks
        sparse_vbrc_data: Pre-converted VBRC data without dense blocks
        matrix_rows, matrix_cols, matrix_nnz: Matrix dimensions
        bench_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    codegen_dir_split = str(GENERATED_SPMM_MKL_NAIVE_DIR / "split")
    codegen_dir_sparse = str(GENERATED_SPMM_MKL_NAIVE_DIR / "sparse")

    # Ensure directories exist
    os.makedirs(codegen_dir_split, exist_ok=True)
    os.makedirs(codegen_dir_sparse, exist_ok=True)

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
    print(f"  [mkl_naive] Generating C code (split)...")
    gen_single_threaded_spmm_mkl_naive(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre,
        ublocks, indptr, indices, csr_val,
        codegen_dir_split, matrix_name, vbr_dir, bench=bench_iterations
    )

    # Evaluate split version (compile + run) - MKL flags auto-detected from directory name
    print(f"  [mkl_naive] Evaluating split version...")
    avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_split_ns = \
        eval_single_file_split_timings(matrix_name, codegen_dir_split, bench_iterations)

    # Generate C code for sparse version
    print(f"  [mkl_naive] Generating C code (fully sparse)...")
    gen_single_threaded_spmm_mkl_naive(
        val_sparse, indx_sparse, bindx_sparse,
        rpntr_sparse, cpntr_sparse,
        bpntrb_sparse, bpntre_sparse,
        ublocks_sparse, indptr_sparse,
        indices_sparse, csr_val_sparse,
        codegen_dir_sparse, matrix_name,
        sparse_vbr_dir, bench=bench_iterations
    )

    # Evaluate fully sparse version (compile + run) - MKL flags auto-detected from directory name
    print(f"  [mkl_naive] Evaluating fully sparse version...")
    sparse_avg_sparse_time, _, _, compile_time_sparse_ns = eval_single_file_split_timings(
        matrix_name, codegen_dir_sparse, bench_iterations
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
            "speedup": round(speedup, 3),
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
            "nnz": block_nnz
        }

    return matrix_result


def process_and_benchmark_matrix_naive_spreg(
    matrix_name: str,
    split_vbrc_data: Dict[str, Any],
    sparse_vbrc_data: Dict[str, Any],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    bench_iterations: int,
) -> Optional[Dict[str, Any]]:
    """
    Run SpMM benchmarks using naive dense + sparse-register-tiling implementation.

    Args:
        matrix_name: Name of the matrix
        split_vbrc_data: Pre-converted VBRC data with dense blocks
        sparse_vbrc_data: Pre-converted VBRC data without dense blocks
        matrix_rows, matrix_cols, matrix_nnz: Matrix dimensions
        bench_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    codegen_dir_split = str(GENERATED_SPMM_NAIVE_SPREG_DIR / "split")
    codegen_dir_sparse = str(GENERATED_SPMM_NAIVE_SPREG_DIR / "sparse")
    
    # Ensure directories exist
    os.makedirs(codegen_dir_split, exist_ok=True)
    os.makedirs(codegen_dir_sparse, exist_ok=True)
    
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
    print(f"  [naive_spreg] Generating C code (split)...")
    gen_single_threaded_spmm_naive_spreg(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre,
        ublocks, indptr, indices, csr_val,
        codegen_dir_split, matrix_name, vbr_dir, bench=bench_iterations
    )

    # Evaluate split version (compile + run)
    print(f"  [naive_spreg] Evaluating split version...")
    avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_split_ns = \
        eval_single_file_split_timings_spreg(matrix_name, codegen_dir_split, bench_iterations)

    # Generate C code for sparse version
    print(f"  [naive_spreg] Generating C code (fully sparse)...")
    gen_single_threaded_spmm_naive_spreg(
        val_sparse, indx_sparse, bindx_sparse,
        rpntr_sparse, cpntr_sparse,
        bpntrb_sparse, bpntre_sparse,
        ublocks_sparse, indptr_sparse,
        indices_sparse, csr_val_sparse,
        codegen_dir_sparse, matrix_name,
        sparse_vbr_dir, bench=bench_iterations
    )

    # Evaluate fully sparse version (compile + run)
    print(f"  [naive_spreg] Evaluating fully sparse version...")
    sparse_avg_sparse_time, _, _, compile_time_sparse_ns = eval_single_file_split_timings_spreg(
        matrix_name, codegen_dir_sparse, bench_iterations
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
            "speedup": round(speedup, 3),
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
            "nnz": block_nnz
        }
    
    return matrix_result


def eval_single_file_split_timings_mkl_spreg(
    fname: str,
    codegen_dir: str,
    bench_freq: int,
    extract_indiv_blocks: bool = True
) -> Tuple[float, float, Dict[int, float], float]:
    """
    Run the compiled benchmark and parse timing results for mkl_spreg version.
    Uses custom compile function that links with both MKL and sparse-register-tiling.

    Returns:
        Tuple of (avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_ns)
    """
    from bench_suitesparse_split_timings_c import remove_outliers_deciles
    import statistics
    import re

    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)

    sparse_times = []
    dense_times = []
    individual_dense_block_times = {}

    # First, compile the C program using mkl_spreg compile function
    c_file_path = os.path.join(codegen_dir, f"{fname}.c")
    compile_result = compile_c_program_mkl_spreg(c_file_path, codegen_dir)

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

        for line in output:
            if line.startswith('Sparse: '):
                sparse_content = line[8:].strip()
                if sparse_content:
                    sparse_values = [float(x.strip()) for x in sparse_content.rstrip(',').split(',') if x.strip()]
                    sparse_times.extend(sparse_values)

            elif line.startswith('Dense: ') and not line.startswith('Dense Block'):
                dense_content = line[7:].strip().rstrip(',')
                if dense_content:
                    dense_values = [float(x.strip()) for x in dense_content.split(',') if x.strip()]
                    dense_times.extend(dense_values)

            elif extract_indiv_blocks:
                dense_block_match = re.search(r'Dense Block (\d+): (.+)', line)
                if dense_block_match:
                    block_id = int(dense_block_match.group(1))
                    block_times_str = dense_block_match.group(2).strip().rstrip(',')
                    if block_times_str:
                        block_times = [float(x.strip()) for x in block_times_str.split(',') if x.strip()]
                        if block_id not in individual_dense_block_times:
                            individual_dense_block_times[block_id] = []
                        individual_dense_block_times[block_id].extend(block_times)

    sparse_times = remove_outliers_deciles(sparse_times)
    dense_times = remove_outliers_deciles(dense_times)

    avg_sparse_time = statistics.mean(sparse_times) if sparse_times else 0
    avg_dense_time = statistics.mean(dense_times) if dense_times else 0

    avg_individual_block_times = {}
    if extract_indiv_blocks:
        for block_id, times in individual_dense_block_times.items():
            times_clean = remove_outliers_deciles(times)
            avg_individual_block_times[block_id] = statistics.mean(times_clean) if times_clean else 0

    return avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_ns


def compile_c_program_mkl_spreg(c_file_path: str, output_dir: str) -> Optional[Tuple[str, float]]:
    """
    Compile the C program that uses both MKL and sparse-register-tiling.
    Uses gcc to compile C code, g++ to compile C++ wrapper, then g++ to link with MKL.
    """
    global _spreg_obj_cache, _spreg_obj_cache_dir

    c_file = os.path.basename(c_file_path)
    output_name = os.path.splitext(c_file)[0]
    output_path = os.path.join(output_dir, output_name)

    spreg_src_files = get_spreg_source_files()
    include_dirs = get_spreg_include_dirs()

    include_flags = []
    for inc_dir in include_dirs:
        include_flags.extend(["-I", inc_dir])

    common_flags = ["-O3", "-march=native", "-fopenmp"]
    cpp_flags = ["-std=c++17", "-DRTE_CACHE_LINE_SIZE=64", "-DENABLE_AVX512", "-mavx512f"]

    # C flags including MKL
    c_flags = []
    for flag in CFLAGS:
        if flag.startswith("-I") or flag.startswith("-L"):
            c_flags.append(flag)
        elif flag in ["-O3", "-march=native", "-funroll-all-loops",
                      "-mprefer-vector-width=512", "-mavx", "-ffast-math"]:
            c_flags.append(flag)
        elif not flag.endswith(".o"):
            c_flags.append(flag)
    # Add MKL include flags
    for flag in MKL_FLAGS:
        if flag.startswith("-I"):
            c_flags.append(flag)

    print(f"  Compiling generated C code: {c_file} (output: {output_path})")

    obj_cache_dir = FILEPATH / ".spreg_obj_cache"
    obj_cache_dir.mkdir(exist_ok=True)

    try:
        start_time = time.time_ns()

        # Compile C file with gcc to object file
        c_obj = os.path.join(output_dir, f"{output_name}_c.o")
        gcc_cmd = ["gcc", "-c", c_file_path, "-o", c_obj] + c_flags + include_flags
        result = subprocess.run(gcc_cmd, cwd=output_dir, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
        if result.returncode != 0:
            print(f"Compilation failed for {c_file} (C compilation): {result.stderr}")
            return None

        # Compile C++ files with g++ to object files (with caching)
        cpp_objs = []
        need_rebuild_cache = _spreg_obj_cache_dir != str(obj_cache_dir) or not _spreg_obj_cache

        for cpp_file in spreg_src_files:
            if not os.path.exists(cpp_file):
                print(f"Warning: C++ source file not found: {cpp_file}")
                continue

            cpp_basename = os.path.basename(cpp_file).replace(".cpp", ".o")
            cpp_hash = str(hash(cpp_file))[-8:]
            cpp_obj_name = f"{cpp_hash}_{cpp_basename}"
            cpp_obj = str(obj_cache_dir / cpp_obj_name)
            cpp_objs.append(cpp_obj)

            if not need_rebuild_cache and os.path.exists(cpp_obj):
                src_mtime = os.path.getmtime(cpp_file)
                obj_mtime = os.path.getmtime(cpp_obj)
                if obj_mtime > src_mtime:
                    continue

            gpp_cmd = ["g++", "-c", cpp_file, "-o", cpp_obj] + common_flags + cpp_flags + include_flags
            result = subprocess.run(gpp_cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            if result.returncode != 0:
                print(f"Compilation failed for {os.path.basename(cpp_file)}: {result.stderr}")
                return None

        _spreg_obj_cache_dir = str(obj_cache_dir)

        # Link everything with g++ including MKL flags
        mkl_link_flags = [flag for flag in MKL_FLAGS if flag.startswith("-L") or flag.startswith("-l") or flag.startswith("-Wl")]
        link_cmd = ["g++", "-o", output_path, c_obj] + cpp_objs + common_flags + ["-fopenmp", "-lstdc++fs", "-lpthread"] + mkl_link_flags
        result = subprocess.run(link_cmd, cwd=output_dir, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
        compile_time_ns = time.time_ns() - start_time
        if result.returncode != 0:
            print(f"Compilation failed for {c_file} (linking): {result.stderr}")
            return None

        print(f"  Finished compiling {c_file}. Starting benchmark runs...")
        return output_path, compile_time_ns
    except subprocess.TimeoutExpired:
        print(f"Compilation timeout for {c_file}")
        return None
    except Exception as e:
        print(f"Compilation error for {c_file}: {e}")
        return None


def process_and_benchmark_matrix_mkl_spreg(
    matrix_name: str,
    split_vbrc_data: Dict[str, Any],
    sparse_vbrc_data: Dict[str, Any],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    bench_iterations: int,
) -> Optional[Dict[str, Any]]:
    """
    Run SpMM benchmarks using MKL dense + sparse-register-tiling implementation.

    Args:
        matrix_name: Name of the matrix
        split_vbrc_data: Pre-converted VBRC data with dense blocks
        sparse_vbrc_data: Pre-converted VBRC data without dense blocks
        matrix_rows, matrix_cols, matrix_nnz: Matrix dimensions
        bench_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    codegen_dir_split = str(GENERATED_SPMM_MKL_SPREG_DIR / "split")
    codegen_dir_sparse = str(GENERATED_SPMM_MKL_SPREG_DIR / "sparse")

    os.makedirs(codegen_dir_split, exist_ok=True)
    os.makedirs(codegen_dir_sparse, exist_ok=True)

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

    print(f"  [mkl_spreg] Generating C code (split)...")
    gen_single_threaded_spmm_mkl_spreg(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre,
        ublocks, indptr, indices, csr_val,
        codegen_dir_split, matrix_name, vbr_dir, bench=bench_iterations
    )

    print(f"  [mkl_spreg] Evaluating split version...")
    avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_split_ns = \
        eval_single_file_split_timings_mkl_spreg(matrix_name, codegen_dir_split, bench_iterations)

    print(f"  [mkl_spreg] Generating C code (fully sparse)...")
    gen_single_threaded_spmm_mkl_spreg(
        val_sparse, indx_sparse, bindx_sparse,
        rpntr_sparse, cpntr_sparse,
        bpntrb_sparse, bpntre_sparse,
        ublocks_sparse, indptr_sparse,
        indices_sparse, csr_val_sparse,
        codegen_dir_sparse, matrix_name,
        sparse_vbr_dir, bench=bench_iterations
    )

    print(f"  [mkl_spreg] Evaluating fully sparse version...")
    sparse_avg_sparse_time, _, _, compile_time_sparse_ns = eval_single_file_split_timings_mkl_spreg(
        matrix_name, codegen_dir_sparse, bench_iterations
    )

    total_time = avg_sparse_time + avg_dense_time
    sparse_percentage = (avg_sparse_time / total_time * 100) if total_time > 0 else 0
    dense_percentage = (avg_dense_time / total_time * 100) if total_time > 0 else 0
    speedup = (sparse_avg_sparse_time / total_time) if total_time > 0 else 0

    dense_all = sum(block.get("rows", 0) * block.get("cols", 0) for block in dense_blocks)
    dense_nnz = sum(block.get("nnz", 0) for block in dense_blocks)
    sparse_nnz = matrix_nnz - dense_nnz
    extra_zeros = dense_all - dense_nnz
    dense_nnz_perc = (dense_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
    sparse_nnz_perc = (sparse_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0

    density_calculation = matrix_nnz / (matrix_rows * matrix_cols) if matrix_rows * matrix_cols > 0 else 0

    matrix_result = {
        "matrix_name": matrix_name,
        "matrix_dimensions": {
            "rows": matrix_rows,
            "cols": matrix_cols,
            "nnz": matrix_nnz,
            "density": round(density_calculation, 3),
        },
        "timing": {
            "sparse_time_ns": round(avg_sparse_time, 2),
            "dense_time_ns": round(avg_dense_time, 2),
            "total_time_ns": round(total_time, 2),
            "sparse_percentage": round(sparse_percentage, 3),
            "dense_percentage": round(dense_percentage, 3),
            "fully_sparse_time": sparse_avg_sparse_time,
            "speedup": round(speedup, 3),
            "expected_sparse_time_ns": ((100-dense_nnz_perc)/100)*sparse_avg_sparse_time,
            "dense_if_sparse_time_ns": sparse_avg_sparse_time - ((100-dense_nnz_perc)/100)*sparse_avg_sparse_time,
            "compile_time_split_s": compile_time_split_ns / 1e9 if compile_time_split_ns else 0.0,
            "compile_time_sparse_s": compile_time_sparse_ns / 1e9 if compile_time_sparse_ns else 0.0,
        },
        "nnz": {
            "sparse_nnz": sparse_nnz,
            "dense_all": dense_all,
            "dense_nnz": dense_nnz,
            "extra_zeros": extra_zeros,
            "dense_nnz_perc": round(dense_nnz_perc, 2),
            "sparse_nnz_perc": round(sparse_nnz_perc, 2),
        },
        "individual_dense_block_timings": {}
    }

    for block_id, block_time in avg_individual_block_times.items():
        block_info = dense_blocks[block_id - 1] if block_id - 1 < len(dense_blocks) else {}
        block_nnz = block_info.get("nnz", 0)
        dense_nnz_sum = sum(block.get("nnz", 0) for block in dense_blocks)
        matrix_result["individual_dense_block_timings"][f"block_{block_id}"] = {
            "time_ns": round(block_time, 2),
            "percentage_of_total_time": round((block_time / total_time * 100), 2) if total_time > 0 else 0,
            "percentage_of_dense_time": round((block_time / avg_dense_time * 100), 2) if avg_dense_time > 0 else 0,
            "percentage_of_total_nnz": round((block_nnz / matrix_nnz * 100), 3) if matrix_nnz > 0 else 0,
            "percentage_of_dense_nnz": round((block_nnz / dense_nnz_sum * 100), 3) if dense_nnz_sum > 0 else 0,
            "rows": block_info.get("rows", 0),
            "cols": block_info.get("cols", 0),
            "density_percent": round(block_info.get("density_percent", 0), 3),
            "nnz": block_nnz
        }

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
    parser.add_argument("--dense", type=str, choices=["naive", "mkl", "all"], default="all",
                        help="Dense kernel to use: naive (handwritten), mkl (cblas_dgemm), or all (default: all)")
    parser.add_argument("--sparse", type=str, choices=["naive", "spreg", "all"], default="all",
                        help="Sparse kernel to use: naive (handwritten), spreg (sparse-register-tiling), or all (default: all)")

    args = parser.parse_args()

    # Determine which combinations to run based on --dense and --sparse args
    dense_kernels = ["naive", "mkl"] if args.dense == "all" else [args.dense]
    sparse_kernels = ["naive", "spreg"] if args.sparse == "all" else [args.sparse]

    # Build list of (dense, sparse) combinations to run
    run_naive_naive = "naive" in dense_kernels and "naive" in sparse_kernels
    run_naive_spreg = "naive" in dense_kernels and "spreg" in sparse_kernels
    run_mkl_naive = "mkl" in dense_kernels and "naive" in sparse_kernels
    run_mkl_spreg = "mkl" in dense_kernels and "spreg" in sparse_kernels
    
    # Get matrices to process - merge positional arguments and --matrices flag, or use all
    matrices = args.matrices or args.matrices_flag
    if not matrices:
        matrices = get_available_matrices()
    
    # Check if --matrices flag was used (for output file naming)
    use_matrix_suffix = args.matrices_flag is not None
    
    benchmark_types = []
    if run_naive_naive:
        benchmark_types.append("naive_naive")
    if run_naive_spreg:
        benchmark_types.append("naive_spreg")
    if run_mkl_naive:
        benchmark_types.append("mkl_naive")
    if run_mkl_spreg:
        benchmark_types.append("mkl_spreg")
    print(f"Will process {len(matrices)} matrices with benchmark types: {', '.join(benchmark_types)}")
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Ensure required directories exist
    SUITESPARSE_DIR.mkdir(exist_ok=True)
    GENERATED_VBR_SPLIT_DIR.mkdir(exist_ok=True)
    GENERATED_VBR_SPARSE_DIR.mkdir(exist_ok=True)
    
    naive_naive_results = []
    naive_spreg_results = []
    mkl_naive_results = []
    mkl_spreg_results = []
    
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
            
            # Run naive_naive version (if enabled)
            if run_naive_naive:
                print(f"\n  === Running naive_naive SpMM benchmark ===")
                result = process_and_benchmark_matrix_naive_naive(
                    matrix_name, split_vbrc_data, sparse_vbrc_data,
                    matrix_rows, matrix_cols, matrix_nnz,
                    args.bench
                )
                if result:
                    existing_idx = next((i for i, r in enumerate(naive_naive_results) if r["matrix_name"] == matrix_name), None)
                    if existing_idx is not None:
                        naive_naive_results[existing_idx] = result
                        print(f"  [naive_naive] Updated existing result for {matrix_name}")
                    else:
                        naive_naive_results.append(result)
                        print(f"  [naive_naive] Added new result for {matrix_name}")
                    # Append matrix name to filename if --matrices flag was used
                    filename = f"sable_spmm_naive_naive_{matrix_name}.json" if use_matrix_suffix else "sable_spmm_naive_naive.json"
                    output_file = output_dir / filename
                    with open(output_file, 'w') as f:
                        json.dump(naive_naive_results, f, indent=2)
                    print(f"  [naive_naive] Results written to {output_file}")

            # Run naive_spreg version (if enabled)
            if run_naive_spreg:
                print(f"\n  === Running naive_spreg SpMM benchmark ===")
                result = process_and_benchmark_matrix_naive_spreg(
                    matrix_name, split_vbrc_data, sparse_vbrc_data,
                    matrix_rows, matrix_cols, matrix_nnz,
                    args.bench
                )
                if result:
                    existing_idx = next((i for i, r in enumerate(naive_spreg_results) if r["matrix_name"] == matrix_name), None)
                    if existing_idx is not None:
                        naive_spreg_results[existing_idx] = result
                        print(f"  [naive_spreg] Updated existing result for {matrix_name}")
                    else:
                        naive_spreg_results.append(result)
                        print(f"  [naive_spreg] Added new result for {matrix_name}")
                    # Append matrix name to filename if --matrices flag was used
                    filename = f"sable_spmm_naive_spreg_{matrix_name}.json" if use_matrix_suffix else "sable_spmm_naive_spreg.json"
                    output_file = output_dir / filename
                    with open(output_file, 'w') as f:
                        json.dump(naive_spreg_results, f, indent=2)
                    print(f"  [naive_spreg] Results written to {output_file}")

            # Run mkl_naive version (if enabled)
            if run_mkl_naive:
                print(f"\n  === Running mkl_naive SpMM benchmark ===")
                result = process_and_benchmark_matrix_mkl_naive(
                    matrix_name, split_vbrc_data, sparse_vbrc_data,
                    matrix_rows, matrix_cols, matrix_nnz,
                    args.bench
                )
                if result:
                    existing_idx = next((i for i, r in enumerate(mkl_naive_results) if r["matrix_name"] == matrix_name), None)
                    if existing_idx is not None:
                        mkl_naive_results[existing_idx] = result
                        print(f"  [mkl_naive] Updated existing result for {matrix_name}")
                    else:
                        mkl_naive_results.append(result)
                        print(f"  [mkl_naive] Added new result for {matrix_name}")
                    # Append matrix name to filename if --matrices flag was used
                    filename = f"sable_spmm_mkl_naive_{matrix_name}.json" if use_matrix_suffix else "sable_spmm_mkl_naive.json"
                    output_file = output_dir / filename
                    with open(output_file, 'w') as f:
                        json.dump(mkl_naive_results, f, indent=2)
                    print(f"  [mkl_naive] Results written to {output_file}")

            # Run mkl_spreg version (if enabled)
            if run_mkl_spreg:
                print(f"\n  === Running mkl_spreg SpMM benchmark ===")
                result = process_and_benchmark_matrix_mkl_spreg(
                    matrix_name, split_vbrc_data, sparse_vbrc_data,
                    matrix_rows, matrix_cols, matrix_nnz,
                    args.bench
                )
                if result:
                    existing_idx = next((i for i, r in enumerate(mkl_spreg_results) if r["matrix_name"] == matrix_name), None)
                    if existing_idx is not None:
                        mkl_spreg_results[existing_idx] = result
                        print(f"  [mkl_spreg] Updated existing result for {matrix_name}")
                    else:
                        mkl_spreg_results.append(result)
                        print(f"  [mkl_spreg] Added new result for {matrix_name}")
                    # Append matrix name to filename if --matrices flag was used
                    filename = f"sable_spmm_mkl_spreg_{matrix_name}.json" if use_matrix_suffix else "sable_spmm_mkl_spreg.json"
                    output_file = output_dir / filename
                    with open(output_file, 'w') as f:
                        json.dump(mkl_spreg_results, f, indent=2)
                    print(f"  [mkl_spreg] Results written to {output_file}")
            
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
    
    def print_results_summary(name, results):
        if results:
            print(f"\n{name} SpMM Results ({len(results)} matrices):")
            for result in results:
                num_dense_blocks = len(result['individual_dense_block_timings'])
                speedup_dispatch = result['timing'].get('speedup', 0)
                print(f"  {result['matrix_name']}: {num_dense_blocks} dense blocks, "
                      f"sparse: {result['timing']['sparse_time_ns']:.0f}ns, "
                      f"dense: {result['timing']['dense_time_ns']:.0f}ns, "
                      f"speedup: {speedup_dispatch:.3f}x")

    if run_naive_naive:
        print_results_summary("naive_naive (handwritten dense + handwritten sparse)", naive_naive_results)
    if run_naive_spreg:
        print_results_summary("naive_spreg (handwritten dense + sparse-register-tiling)", naive_spreg_results)
    if run_mkl_naive:
        print_results_summary("mkl_naive (MKL dense + handwritten sparse)", mkl_naive_results)
    if run_mkl_spreg:
        print_results_summary("mkl_spreg (MKL dense + sparse-register-tiling)", mkl_spreg_results)
    
    print(f"\nResults written to {output_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())
