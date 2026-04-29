#!/usr/bin/env python3
"""
Unified benchmark script for SABLE sparse matrix operations (SpMV and SpMM).

Use --operation to select one or both:
  spmv       - sparse matrix-vector multiplication
  spmm       - sparse matrix-matrix multiplication
  spmv,spmm  - both (default)

All dispatches use the frontend compiler.
"""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import scipy
from scipy.io import mmread
from scipy.sparse import csc_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "find-submatrices"))
from find_matrices import cleanup_matrix_files, get_matrix_info, get_matrix_paths
from ssgetpy import fetch

from sable import Matrix, Operation, Plan
from sable.build_config import DenseKernel, SparseKernel
from sable.compiler import build_compile_command_for_plan
from sable.extractors import BlockDetectorSkip, CSRConvertor
from sable.kernels import (
    MKLCSRSpmm,
    MKLCSRSpmv,
    MKLVBRSpmm,
    MKLVBRSpmv,
    MixedVBRSpmm,
    MixedVBRSpmv,
    NaiveCSRSpmm,
    NaiveCSRSpmv,
    NaiveVBRSpmm,
    NaiveVBRSpmv,
    SPRegCSRSpmm,
    SPV8CSRSpmv,
)
from sable.tensor import DenseInput, DenseLayout
from utils.fileio import parse_yaml_blocks, write_dense_matrix, write_dense_vector
from utils.utils import remove_outliers_deciles, set_ulimit


FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

COMPILE_TIMEOUT = 60 * 60 * 4
DEFAULT_SPMV_BENCH_ITERATIONS = 30
DEFAULT_SPMM_BENCH_ITERATIONS = 10
PHYSICAL_CORES = list(range(os.cpu_count() or 20))
SPMM_NRHS = 512

SPMV_SPARSE_KERNELS = (SparseKernel.NAIVE, SparseKernel.MKL, SparseKernel.SPV8)
SPMM_SPARSE_KERNELS = ("naive", "mkl", "spreg")

SUITESPARSE_DIR = FILEPATH / "Suitesparse"
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _generated_vector_path(size: int) -> str:
    return os.path.abspath(os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{size}.vector"))


def _generated_matrix_path(rows: int, cols: int) -> str:
    return os.path.abspath(os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{rows}x{cols}.matrix"))


def _executor_env(command: list[str], runtime_env: dict[str, str]) -> dict[str, str] | None:
    lib_dirs = [flag[2:] for flag in command if flag.startswith("-L")]
    if not lib_dirs and not runtime_env:
        return None

    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH")
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    env.update(runtime_env)
    return env


def _parse_timing_output(
    output: list[str],
    sparse_times: list[float],
    dense_times: list[float],
    individual_dense_block_times: dict[int, list[float]],
    extract_indiv_blocks: bool,
) -> None:
    for line in output:
        if line.startswith("Sparse: "):
            values = [float(x.strip()) for x in line[8:].strip().rstrip(",").split(",") if x.strip()]
            sparse_times.extend(values)
        elif line.startswith("Dense: ") and not line.startswith("Dense Block"):
            values = [float(x.strip()) for x in line[7:].strip().rstrip(",").split(",") if x.strip()]
            dense_times.extend(values)
        elif extract_indiv_blocks:
            dense_block_match = re.search(r"Dense Block (\d+): (.+)", line)
            if dense_block_match:
                block_id = int(dense_block_match.group(1))
                values = [
                    float(x.strip())
                    for x in dense_block_match.group(2).strip().rstrip(",").split(",")
                    if x.strip()
                ]
                individual_dense_block_times.setdefault(block_id, []).extend(values)


# ---------------------------------------------------------------------------
# Compilation and evaluation
# ---------------------------------------------------------------------------


def compile_frontend_executor(executor) -> Optional[Tuple[str, float]]:
    output_path = os.path.abspath(os.path.join(executor.artifact_dir, executor.filename))
    command = build_compile_command_for_plan(executor.plan, executor.c_path, output_path)

    print(f"  Compiling generated C code: {os.path.basename(executor.c_path)} (output: {output_path})")
    try:
        start_time = time.time_ns()
        result = subprocess.run(command, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
        compile_time_ns = time.time_ns() - start_time
        if result.returncode != 0:
            print(f"Compilation failed for {executor.c_path}: {result.stderr}")
            return None
        executor.binary_path = output_path
        executor.compile_command = command
        print(f"  Finished compiling {os.path.basename(executor.c_path)}. Starting benchmark runs...")
        return output_path, compile_time_ns
    except subprocess.TimeoutExpired:
        print(f"Compilation timeout for {executor.c_path}")
        return None
    except Exception as exc:
        print(f"Compilation error for {executor.c_path}: {exc}")
        return None


def eval_frontend_executor_timings(
    executor,
    bench_freq: int,
    threads: int = 1,
    extract_indiv_blocks: bool = True,
) -> Tuple[float, float, Dict[int, float], float]:
    cores_to_use = PHYSICAL_CORES[:threads]
    compile_result = compile_frontend_executor(executor)
    if compile_result is None:
        print(f"Failed to compile {executor.filename}, skipping evaluation")
        return 0, 0, {}, 0.0

    executable_path, compile_time_ns = compile_result
    sparse_times: list[float] = []
    dense_times: list[float] = []
    individual_dense_block_times: dict[int, list[float]] = {}

    if os.environ.get("SLURM_JOB_ID"):
        run_cmd = [executable_path]
    else:
        run_cmd = ["taskset", "-a", "-c", ",".join(str(x) for x in cores_to_use), executable_path]

    print(f"  Executing benchmark binary: {executable_path}")
    for _ in range(bench_freq):
        try:
            output = subprocess.check_output(
                run_cmd,
                cwd=executor.artifact_dir,
                env=_executor_env(executor.compile_command or [], executor.runtime_env),
                preexec_fn=set_ulimit,
            ).decode("utf-8").split("\n")
        except subprocess.CalledProcessError as exc:
            print(f"Error running {executor.filename}: {exc}")
            continue
        _parse_timing_output(output, sparse_times, dense_times, individual_dense_block_times, extract_indiv_blocks)

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


# ---------------------------------------------------------------------------
# Matrix download and discovery
# ---------------------------------------------------------------------------


def download_matrix_from_suitesparse(matrix_name: str) -> Optional[Tuple[str, Any, Optional[str], Optional[str]]]:
    SUITESPARSE_DIR.mkdir(exist_ok=True)

    original_ssgetpy_dir = os.environ.get("SSGETPY_DIR")
    try:
        os.environ["SSGETPY_DIR"] = str(SUITESPARSE_DIR)
        matrix_info = get_matrix_info(matrix_name)
        if matrix_info is None:
            return None

        fetch(matrix_name)
        tar_path, tar_dir, matrix_subdir, matrix_path = get_matrix_paths(matrix_info)

        if not os.path.exists(matrix_path):
            expected_path = SUITESPARSE_DIR / matrix_info.name / f"{matrix_info.name}.mtx"
            if expected_path.exists():
                matrix_path = str(expected_path)
                matrix_subdir = str(expected_path.parent)
                tar_candidates = list(SUITESPARSE_DIR.glob(f"{matrix_info.name}*.tar.gz"))
                tar_path = str(tar_candidates[0]) if tar_candidates else None
            else:
                print(f"Error: Matrix file not found at {matrix_path} or {expected_path}")
                return None

        return matrix_path, matrix_info, tar_path, matrix_subdir
    finally:
        if original_ssgetpy_dir is not None:
            os.environ["SSGETPY_DIR"] = original_ssgetpy_dir
        elif "SSGETPY_DIR" in os.environ:
            del os.environ["SSGETPY_DIR"]


def get_available_matrices() -> List[str]:
    return [f.stem for f in RESULTS_DIR.glob("*.yaml")]


# ---------------------------------------------------------------------------
# SpMV kernel helpers
# ---------------------------------------------------------------------------


def _dense_spmv_kernel(dense_kernel: DenseKernel):
    if dense_kernel == DenseKernel.MKL:
        return MKLVBRSpmv()
    if dense_kernel == DenseKernel.MIXED:
        return MixedVBRSpmv()
    return NaiveVBRSpmv()


def _sparse_spmv_kernel(sparse_kernel: SparseKernel):
    if sparse_kernel == SparseKernel.MKL:
        return MKLCSRSpmv()
    if sparse_kernel == SparseKernel.SPV8:
        return SPV8CSRSpmv()
    if sparse_kernel == SparseKernel.NAIVE:
        return NaiveCSRSpmv()
    raise ValueError(f"{sparse_kernel.value} is not a frontend SpMV sparse kernel")


# ---------------------------------------------------------------------------
# SpMM kernel helpers
# ---------------------------------------------------------------------------


def _dense_spmm_kernel(dense_kernel: DenseKernel):
    if dense_kernel == DenseKernel.MKL:
        return MKLVBRSpmm()
    if dense_kernel == DenseKernel.MIXED:
        return MixedVBRSpmm()
    return NaiveVBRSpmm()


def _sparse_spmm_kernel(sparse_kernel: str):
    if sparse_kernel == "mkl":
        return MKLCSRSpmm()
    if sparse_kernel == "spreg":
        return SPRegCSRSpmm()
    if sparse_kernel == "naive":
        return NaiveCSRSpmm()
    raise ValueError(f"Unknown SpMM sparse kernel: {sparse_kernel}")


# ---------------------------------------------------------------------------
# VBR conversion (shared structure, operation-specific RHS)
# ---------------------------------------------------------------------------


def _analyze_blocks_from_coords(
    block_coords: List[Tuple[int, int, int, int]],
    mat: scipy.sparse.spmatrix,
) -> List[Dict[str, Any]]:
    csr = mat.tocsr()
    dense_blocks = []
    for r_start, r_end, c_start, c_end in block_coords:
        rows = r_end - r_start
        cols = c_end - c_start
        block_nnz = csr[r_start:r_end, c_start:c_end].nnz
        block_size = rows * cols
        density = (block_nnz / block_size * 100) if block_size > 0 else 0
        dense_blocks.append({"rows": rows, "cols": cols, "density_percent": density, "nnz": block_nnz})
    return dense_blocks


def _convert_and_prepare(
    operation: Operation,
    matrix_name: str,
    dense_block_coords: List[Tuple[int, int, int, int]],
    mat: scipy.sparse.spmatrix,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    csr_mat = mat.tocsr()
    _, matrix_cols = csr_mat.shape

    if operation == Operation.SPMV:
        write_dense_vector(1.0, matrix_cols)
    else:
        write_dense_matrix(1.0, matrix_cols, SPMM_NRHS)

    dense_blocks = _analyze_blocks_from_coords(dense_block_coords, csr_mat)

    split_data = {
        "dense_blocks": dense_blocks,
        "block_coords": list(dense_block_coords),
        "matrix": csr_mat,
    }
    sparse_data = {
        "matrix": csr_mat,
    }

    return split_data, sparse_data


# ---------------------------------------------------------------------------
# Frontend compilation per operation
# ---------------------------------------------------------------------------


def _compile_spmv_frontend(
    matrix_name: str,
    matrix_source,
    block_coords: list[tuple[int, int, int, int]],
    artifact_dir: str,
    dense_kernel: DenseKernel,
    sparse_kernel: SparseKernel,
    bench_iterations: int,
):
    matrix = Matrix(matrix_source, name=matrix_name)
    write_dense_vector(1.0, matrix.ncols)

    plan = Plan(matrix, artifact_dir=artifact_dir)
    plan.rhs(DenseInput.vector(_generated_vector_path(matrix.ncols), matrix.ncols))
    if block_coords:
        vbr = plan.extract(BlockDetectorSkip(block_coords))
        plan.dispatch(vbr, _dense_spmv_kernel(dense_kernel))
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, _sparse_spmv_kernel(sparse_kernel))
    return plan.compile(filename=matrix_name, bench=bench_iterations)


def _compile_spmm_frontend(
    matrix_name: str,
    matrix_source,
    block_coords: list[tuple[int, int, int, int]],
    artifact_dir: str,
    dense_kernel: DenseKernel,
    sparse_kernel: str,
    bench_iterations: int,
):
    matrix = Matrix(matrix_source, name=matrix_name)
    write_dense_matrix(1.0, matrix.ncols, SPMM_NRHS)

    plan = Plan(matrix, artifact_dir=artifact_dir)
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(matrix.ncols, SPMM_NRHS),
            shape=(matrix.ncols, SPMM_NRHS),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    if block_coords:
        vbr = plan.extract(BlockDetectorSkip(block_coords))
        plan.dispatch(vbr, _dense_spmm_kernel(dense_kernel))
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, _sparse_spmm_kernel(sparse_kernel))
    return plan.compile(filename=matrix_name, bench=bench_iterations)


# ---------------------------------------------------------------------------
# Result building and benchmarking
# ---------------------------------------------------------------------------


def _build_matrix_result(
    matrix_name: str,
    dense_blocks: list[dict[str, Any]],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    avg_sparse_time: float,
    avg_dense_time: float,
    avg_individual_block_times: dict[int, float],
    sparse_avg_sparse_time: float,
    compile_time_split_ns: float,
    compile_time_sparse_ns: float,
    codegen_time_split_ms: int,
    codegen_time_sparse_ms: int,
) -> Dict[str, Any]:
    total_time = avg_sparse_time + avg_dense_time
    dense_all = sum(block.get("rows", 0) * block.get("cols", 0) for block in dense_blocks)
    dense_nnz = sum(block.get("nnz", 0) for block in dense_blocks)
    sparse_nnz = matrix_nnz - dense_nnz
    extra_zeros = dense_all - dense_nnz
    dense_nnz_perc = (dense_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
    sparse_nnz_perc = (sparse_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
    density_calculation = matrix_nnz / (matrix_rows * matrix_cols) if matrix_rows * matrix_cols > 0 else 0

    result = {
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
            "sparse_percentage": round((avg_sparse_time / total_time * 100), 3) if total_time > 0 else 0,
            "dense_percentage": round((avg_dense_time / total_time * 100), 3) if total_time > 0 else 0,
            "fully_sparse_time": sparse_avg_sparse_time,
            "speedup": round((sparse_avg_sparse_time / total_time), 3) if total_time > 0 else 0,
            "max_theoretical_speedup": round((sparse_avg_sparse_time / avg_sparse_time), 3) if avg_sparse_time > 0 else 0,
            "expected_sparse_time_ns": ((100 - dense_nnz_perc) / 100) * sparse_avg_sparse_time,
            "dense_if_sparse_time_ns": sparse_avg_sparse_time - ((100 - dense_nnz_perc) / 100) * sparse_avg_sparse_time,
            "compile_time_split_s": compile_time_split_ns / 1e9 if compile_time_split_ns else 0.0,
            "compile_time_sparse_s": compile_time_sparse_ns / 1e9 if compile_time_sparse_ns else 0.0,
            "codegen_time_split_ms": codegen_time_split_ms,
            "codegen_time_sparse_ms": codegen_time_sparse_ms,
        },
        "nnz": {
            "sparse_nnz": sparse_nnz,
            "dense_all": dense_all,
            "dense_nnz": dense_nnz,
            "extra_zeros": extra_zeros,
            "dense_nnz_perc": round(dense_nnz_perc, 2),
            "sparse_nnz_perc": round(sparse_nnz_perc, 2),
        },
        "individual_dense_block_timings": {},
    }

    for block_id, block_time in avg_individual_block_times.items():
        block_info = dense_blocks[block_id - 1] if block_id - 1 < len(dense_blocks) else {}
        block_nnz = block_info.get("nnz", 0)
        dense_nnz_sum = sum(block.get("nnz", 0) for block in dense_blocks)
        result["individual_dense_block_timings"][f"block_{block_id}"] = {
            "time_ns": round(block_time, 2),
            "percentage_of_total_time": round((block_time / total_time * 100), 2) if total_time > 0 else 0,
            "percentage_of_dense_time": round((block_time / avg_dense_time * 100), 2) if avg_dense_time > 0 else 0,
            "percentage_of_total_nnz": round((block_nnz / matrix_nnz * 100), 3) if matrix_nnz > 0 else 0,
            "percentage_of_dense_nnz": round((block_nnz / dense_nnz_sum * 100), 3) if dense_nnz_sum > 0 else 0,
            "rows": block_info.get("rows", 0),
            "cols": block_info.get("cols", 0),
            "density_percent": round(block_info.get("density_percent", 0), 3),
            "nnz": block_nnz,
        }

    return result


def _process_and_benchmark_frontend(
    operation: Operation,
    matrix_name: str,
    split_vbrc_data: Dict[str, Any],
    sparse_vbrc_data: Dict[str, Any],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    bench_iterations: int,
    dense_kernel: DenseKernel,
    sparse_kernel,
    threads: int = 1,
) -> Optional[Dict[str, Any]]:
    if threads != 1:
        raise ValueError("The frontend benchmark path is single-threaded for now")

    if operation == Operation.SPMV:
        sparse_label = sparse_kernel.value
        compile_fn = _compile_spmv_frontend
        dir_prefix = "Generated_SpMV_C"
    else:
        sparse_label = sparse_kernel
        compile_fn = _compile_spmm_frontend
        dir_prefix = "Generated_SpMM_C"

    variant_name = f"{dense_kernel.value}_{sparse_label}"
    base_codegen_dir = FILEPATH / f"{dir_prefix}_{dense_kernel.value}_{sparse_label}"
    codegen_dir_split = str(base_codegen_dir / "split")
    codegen_dir_sparse = str(base_codegen_dir / "sparse")
    os.makedirs(codegen_dir_split, exist_ok=True)
    os.makedirs(codegen_dir_sparse, exist_ok=True)

    print(f"  [{variant_name}] Generating frontend C code (split)...")
    split_executor = compile_fn(
        matrix_name,
        split_vbrc_data["matrix"],
        split_vbrc_data["block_coords"],
        codegen_dir_split,
        dense_kernel,
        sparse_kernel,
        bench_iterations,
    )

    print(f"  [{variant_name}] Evaluating split version...")
    avg_sparse_time, avg_dense_time, avg_individual_block_times, compile_time_split_ns = eval_frontend_executor_timings(
        split_executor, bench_iterations, threads=threads
    )

    print(f"  [{variant_name}] Generating frontend C code (fully sparse)...")
    sparse_executor = compile_fn(
        matrix_name,
        sparse_vbrc_data["matrix"],
        [],
        codegen_dir_sparse,
        dense_kernel,
        sparse_kernel,
        bench_iterations,
    )

    print(f"  [{variant_name}] Evaluating fully sparse version...")
    sparse_avg_sparse_time, _, _, compile_time_sparse_ns = eval_frontend_executor_timings(
        sparse_executor, bench_iterations, threads=threads
    )

    return _build_matrix_result(
        matrix_name,
        split_vbrc_data["dense_blocks"],
        matrix_rows,
        matrix_cols,
        matrix_nnz,
        avg_sparse_time,
        avg_dense_time,
        avg_individual_block_times,
        sparse_avg_sparse_time,
        compile_time_split_ns,
        compile_time_sparse_ns,
        split_executor.codegen_time_ms,
        sparse_executor.codegen_time_ms,
    )


def _append_result(
    all_results: dict[str, list[dict[str, Any]]],
    results_key: str,
    matrix_name: str,
    result: dict[str, Any],
    num_threads: int,
    output_file: pathlib.Path,
) -> None:
    results_list = all_results.setdefault(results_key, [])
    existing_idx = next((i for i, r in enumerate(results_list) if r["matrix_name"] == matrix_name), None)
    if existing_idx is not None:
        matrix_entry = results_list[existing_idx]
        print(f"  [{results_key}] Updating result for {matrix_name}")
    else:
        matrix_entry = {
            "matrix_name": result["matrix_name"],
            "matrix_dimensions": result["matrix_dimensions"],
            "timing": {},
            "nnz": result["nnz"],
        }
        results_list.append(matrix_entry)
        print(f"  [{results_key}] Added new result for {matrix_name}")

    thread_key = f"{num_threads} thread"
    thread_timing = dict(result["timing"])
    thread_timing["individual_dense_block_timings"] = result["individual_dense_block_timings"]
    matrix_entry["timing"][thread_key] = thread_timing
    with open(output_file, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"  [{results_key}] Results written to {output_file}")


# ---------------------------------------------------------------------------
# Sparse kernel resolution per operation
# ---------------------------------------------------------------------------


def _resolve_sparse_kernels(operation: Operation, sparse_arg: str, parser: argparse.ArgumentParser):
    if operation == Operation.SPMV:
        valid = {k.value for k in SPMV_SPARSE_KERNELS}
        if sparse_arg == "all":
            return list(SPMV_SPARSE_KERNELS)
        requested = [name.strip() for name in sparse_arg.split(",")]
        selected = [SparseKernel(n) for n in requested if n in valid]
        skipped = [n for n in requested if n not in valid]
        if skipped:
            print(f"  [spmv] Skipping sparse kernels not available for SpMV: {skipped}")
        if not selected:
            parser.error(f"No valid SpMV sparse kernels. Valid options: {valid}")
        return selected
    else:
        valid = set(SPMM_SPARSE_KERNELS)
        if sparse_arg == "all":
            return list(SPMM_SPARSE_KERNELS)
        requested = [name.strip() for name in sparse_arg.split(",")]
        selected = [n for n in requested if n in valid]
        skipped = [n for n in requested if n not in valid]
        if skipped:
            print(f"  [spmm] Skipping sparse kernels not available for SpMM: {skipped}")
        if not selected:
            parser.error(f"No valid SpMM sparse kernels. Valid options: {valid}")
        return selected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SABLE sparse matrix operations (SpMV / SpMM)",
        epilog=(
            "Examples:\n"
            "  %(prog)s --operation spmv,spmm eris1176 bloweybl\n"
            "  %(prog)s --operation spmv --sparse naive,mkl,spv8 --dense naive,mkl,mixed\n"
            "  %(prog)s  # both operations, all matrices, all kernels"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--operation", type=str, default="spmv,spmm",
                        help="Comma-separated operations: spmv, spmm, or spmv,spmm (default: spmv,spmm)")
    parser.add_argument("matrices", nargs="*", help="Matrix names to benchmark.")
    parser.add_argument("--matrices", dest="matrices_flag", nargs="*", metavar="MATRIX")
    parser.add_argument("--bench", type=int, default=None,
                        help=f"Benchmark iterations (default: {DEFAULT_SPMV_BENCH_ITERATIONS} for spmv, {DEFAULT_SPMM_BENCH_ITERATIONS} for spmm)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--sparse", type=str, default="all",
                        help="SpMV: naive,spv8,mkl. SpMM: naive,mkl,spreg. Invalid names silently skipped per operation.")
    parser.add_argument("--dense", type=str, default="all", help="naive, mixed, mkl, all, or comma-separated")
    parser.add_argument("--threads", type=str, default="1")
    args = parser.parse_args()

    operations = [Operation(op.strip()) for op in args.operation.split(",")]

    thread_counts = [int(t.strip()) for t in args.threads.split(",")]
    if any(thread_count != 1 for thread_count in thread_counts):
        parser.error("The frontend benchmark path is single-threaded for now; use --threads 1")

    valid_dense = {kernel.value for kernel in DenseKernel}
    if args.dense == "all":
        dense_kernels = list(DenseKernel)
    else:
        dense_names = [name.strip() for name in args.dense.split(",")]
        invalid = set(dense_names) - valid_dense
        if invalid:
            parser.error(f"Invalid dense kernel(s): {invalid}. Valid options: {valid_dense}")
        dense_kernels = [DenseKernel(name) for name in dense_names]

    matrices = args.matrices or args.matrices_flag
    specific_matrices_requested = matrices is not None and len(matrices) > 0
    if not matrices:
        matrices = get_available_matrices()

    if specific_matrices_requested and len(matrices) == 1:
        output_suffix = f"_{matrices[0]}"
    elif specific_matrices_requested:
        output_suffix = "_" + "_".join(matrices)
    else:
        output_suffix = ""

    ops_label = "+".join(op.value.upper() for op in operations)
    print(f"[{ops_label}] Will process {len(matrices)} matrices")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    SUITESPARSE_DIR.mkdir(exist_ok=True)

    all_results: dict[str, list[dict[str, Any]]] = {}

    for matrix_name in matrices:
        yaml_path = RESULTS_DIR / f"{matrix_name}.yaml"
        if not yaml_path.exists():
            print(f"Warning: YAML file not found for {matrix_name}, skipping")
            continue

        print(f"\nProcessing {matrix_name}...")
        print("  Downloading matrix from SuiteSparse...")
        download_result = download_matrix_from_suitesparse(matrix_name)
        if download_result is None:
            print(f"  Failed to download {matrix_name}, skipping")
            continue

        mtx_path, matrix_info, tar_path, matrix_subdir = download_result

        try:
            print(f"  Loading matrix from {mtx_path}...")
            A = csc_matrix(mmread(mtx_path), copy=False)
            matrix_rows, matrix_cols = A.shape
            matrix_nnz = A.nnz
            print(f"  Matrix shape: {matrix_rows} x {matrix_cols}, NNZ: {matrix_nnz}")

            print(f"  Parsing dense blocks from {yaml_path}...")
            dense_block_coords = parse_yaml_blocks(str(yaml_path))
            print(f"  Found {len(dense_block_coords)} dense blocks")

            for operation in operations:
                op_label = operation.value.upper()
                sparse_kernels = _resolve_sparse_kernels(operation, args.sparse, parser)
                bench_iterations = args.bench
                if bench_iterations is None:
                    bench_iterations = DEFAULT_SPMV_BENCH_ITERATIONS if operation == Operation.SPMV else DEFAULT_SPMM_BENCH_ITERATIONS

                print(f"\n  === Converting to frontend formats ({op_label}) ===")
                split_vbrc_data, sparse_vbrc_data = _convert_and_prepare(
                    operation, matrix_name, dense_block_coords, A
                )

                for num_threads in thread_counts:
                    for dense_kernel in dense_kernels:
                        for sparse_kernel in sparse_kernels:
                            sparse_label = sparse_kernel.value if isinstance(sparse_kernel, SparseKernel) else sparse_kernel
                            results_key = f"{operation.value}_{dense_kernel.value}_{sparse_label}"
                            print(f"\n  === Running {results_key} benchmark (threads={num_threads}) ===")

                            result = _process_and_benchmark_frontend(
                                operation,
                                matrix_name,
                                split_vbrc_data,
                                sparse_vbrc_data,
                                matrix_rows,
                                matrix_cols,
                                matrix_nnz,
                                bench_iterations,
                                dense_kernel=dense_kernel,
                                sparse_kernel=sparse_kernel,
                                threads=num_threads,
                            )

                            if result:
                                output_file = output_dir / f"sable_{results_key}{output_suffix}.json"
                                _append_result(all_results, results_key, matrix_name, result, num_threads, output_file)

            print(f"\nCompleted processing {matrix_name}")
        except Exception as exc:
            print(f"  Error processing {matrix_name}: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            if "matrix_info" in locals() and matrix_info is not None:
                print(f"  Cleaning up downloaded files for {matrix_name}...")
                if tar_path is not None or matrix_subdir is not None:
                    cleanup_matrix_files(tar_path, matrix_subdir)

    print("\n" + "=" * 60)
    print(f"Benchmark Summary ({ops_label})")
    print("=" * 60)
    for results_key, results_list in all_results.items():
        if results_list:
            print(f"\n{results_key} Results ({len(results_list)} matrices):")
            for result in results_list:
                for thread_key, thread_timing in result["timing"].items():
                    print(
                        f"  {result['matrix_name']} ({thread_key}): "
                        f"{len(thread_timing.get('individual_dense_block_timings', {}))} dense blocks, "
                        f"sparse: {thread_timing['sparse_time_ns']:.0f}ns, "
                        f"dense: {thread_timing['dense_time_ns']:.0f}ns, "
                        f"speedup: {thread_timing.get('speedup', 0):.3f}x"
                    )

    print(f"\nResults written to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
