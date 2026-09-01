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
from find_matrices import cleanup_matrix_files, get_matrix_info

from sable import Matrix, Operation, Plan
from sable.build_config import CSRKernel, VBRKernel, VDIAKernel
from sable.compiler import build_compile_command_for_plan
from sable.extractors import BandExtractorSkip, BlockDetectorSkip, CSRConvertor
from sable.kernels import (
    MKLCSRSpmm,
    MKLCSRSpmv,
    MKLVBRSpmm,
    MKLVBRSpmv,
    MixedVBRSpmm,
    MixedVBRSpmv,
    NaiveCSRSpmm,
    NaiveCSRSpmv,
    NaiveVDIASpmm,
    NaiveVDIASpmv,
    NaiveVBRSpmm,
    NaiveVBRSpmv,
    SPRegCSRSpmm,
    SPV8CSRSpmv,
    UZPCSRSpmv,
)
from sable.tensor import DenseInput, DenseLayout
from utils.fileio import parse_yaml_bands, parse_yaml_blocks, write_dense_matrix, write_dense_vector
from utils.utils import remove_outliers_deciles, set_ulimit


FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

COMPILE_TIMEOUT = 60 * 60 * 4
DEFAULT_SPMV_BENCH_ITERATIONS = 30
DEFAULT_SPMM_BENCH_ITERATIONS = 10
PHYSICAL_CORES = list(range(os.cpu_count() or 20))
SPMM_NRHS = 512

SPMV_CSR_KERNELS = (CSRKernel.NAIVE, CSRKernel.MKL, CSRKernel.SPV8, CSRKernel.UZP)
SPMM_CSR_KERNELS = (CSRKernel.NAIVE, CSRKernel.MKL, CSRKernel.SPREG)

SUITESPARSE_DIR = pathlib.Path(os.environ.get("SABLE_SUITESPARSE_DIR") or str(FILEPATH / "Suitesparse"))
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"
BANDS_RESULTS_DIR = FILEPATH / "find-submatrices" / "results_bands_075"
DEFAULT_BASELINE_RESULTS_DIR = FILEPATH / "results"
_RESULTS_JSON_CACHE: dict[pathlib.Path, list[dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _generated_vector_path(size: int) -> str:
    dense_dir = os.environ.get("SABLE_DENSE_TENSOR_DIR") or os.path.join(BASE_PATH, "Generated_dense_tensors")
    return os.path.abspath(os.path.join(dense_dir, f"generated_vector_{size}.vector"))


def _generated_matrix_path(rows: int, cols: int) -> str:
    dense_dir = os.environ.get("SABLE_DENSE_TENSOR_DIR") or os.path.join(BASE_PATH, "Generated_dense_tensors")
    return os.path.abspath(os.path.join(dense_dir, f"generated_matrix_{rows}x{cols}.matrix"))


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


def _parse_timing_values(raw_values: str) -> list[float]:
    return [float(x.strip()) for x in raw_values.strip().rstrip(",").split(",") if x.strip()]


def _parse_timing_output(
    output: list[str],
    dispatch_times: dict[int, list[float]],
    dispatch_part_times: dict[tuple[int, int], list[float]],
    extract_parts: bool,
) -> None:
    for line in output:
        dispatch_match = re.search(r"Dispatch (\d+): (.+)", line)
        if dispatch_match:
            dispatch_id = int(dispatch_match.group(1))
            dispatch_times.setdefault(dispatch_id, []).extend(_parse_timing_values(dispatch_match.group(2)))
            continue

        if extract_parts:
            part_match = re.search(r"Dispatch (\d+) Part (\d+): (.+)", line)
            if part_match:
                dispatch_id = int(part_match.group(1))
                part_id = int(part_match.group(2))
                dispatch_part_times.setdefault((dispatch_id, part_id), []).extend(
                    _parse_timing_values(part_match.group(3))
                )


def _thread_key(num_threads: int) -> str:
    return f"{num_threads} thread"


def _load_results_json(path: pathlib.Path) -> list[dict[str, Any]]:
    path = path.resolve()
    cached = _RESULTS_JSON_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} should contain a JSON list")
    _RESULTS_JSON_CACHE[path] = data
    return data


def _baseline_result_file(
    operation: Operation,
    csr_kernel: CSRKernel,
    baseline_results_dir: pathlib.Path,
) -> pathlib.Path:
    preferred = baseline_results_dir / f"sable_{operation.value}_mkl_{csr_kernel.value}.json"
    if preferred.exists():
        return preferred

    candidates = sorted(baseline_results_dir.glob(f"sable_{operation.value}_*_{csr_kernel.value}.json"))
    for candidate in candidates:
        try:
            data = _load_results_json(candidate)
        except Exception:
            continue
        if any("fully_sparse_time" in timing for entry in data for timing in entry.get("timing", {}).values()):
            return candidate

    raise FileNotFoundError(
        f"No fully sparse baseline file found for {operation.value}/{csr_kernel.value} in {baseline_results_dir}"
    )


def _lookup_fully_sparse_baseline(
    operation: Operation,
    csr_kernel: CSRKernel,
    matrix_name: str,
    num_threads: int,
    baseline_results_dir: pathlib.Path,
) -> tuple[float, str]:
    baseline_file = _baseline_result_file(operation, csr_kernel, baseline_results_dir)
    thread_key = _thread_key(num_threads)
    for entry in _load_results_json(baseline_file):
        if entry.get("matrix_name") != matrix_name:
            continue
        thread_timing = entry.get("timing", {}).get(thread_key)
        if thread_timing is None:
            raise KeyError(f"{baseline_file} has no '{thread_key}' baseline for {matrix_name}")
        if "fully_sparse_time" not in thread_timing:
            raise KeyError(f"{baseline_file} has no fully_sparse_time for {matrix_name} ({thread_key})")
        baseline_time = float(thread_timing["fully_sparse_time"])
        source = f"{baseline_file.name}:{matrix_name}:{thread_key}:fully_sparse_time"
        return baseline_time, source
    raise KeyError(f"{baseline_file} has no baseline entry for matrix {matrix_name}")


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
    extract_parts: bool = True,
) -> Tuple[Dict[int, float], Dict[str, float], float]:
    cores_to_use = PHYSICAL_CORES[:threads]
    compile_result = compile_frontend_executor(executor)
    if compile_result is None:
        print(f"Failed to compile {executor.filename}, skipping evaluation")
        return {}, {}, 0.0

    executable_path, compile_time_ns = compile_result
    dispatch_times: dict[int, list[float]] = {}
    dispatch_part_times: dict[tuple[int, int], list[float]] = {}

    if os.environ.get("SLURM_JOB_ID"):
        run_cmd = [executable_path]
    else:
        run_cmd = ["taskset", "-a", "-c", ",".join(str(x) for x in cores_to_use), executable_path]

    print(f"  Executing benchmark binary: {executable_path}")
    for _ in range(bench_freq):
        try:
            output = subprocess.check_output(
                run_cmd,
                cwd=executor.runtime_cwd or executor.artifact_dir,
                env=_executor_env(executor.compile_command or [], executor.runtime_env),
                preexec_fn=set_ulimit,
            ).decode("utf-8").split("\n")
        except subprocess.CalledProcessError as exc:
            print(f"Error running {executor.filename}: {exc}")
            continue
        _parse_timing_output(output, dispatch_times, dispatch_part_times, extract_parts)

    avg_dispatch_times = {}
    for dispatch_id, times in dispatch_times.items():
        times_clean = remove_outliers_deciles(times)
        avg_dispatch_times[dispatch_id] = statistics.mean(times_clean) if times_clean else 0

    avg_dispatch_part_times = {}
    if extract_parts:
        for (dispatch_id, part_id), times in dispatch_part_times.items():
            times_clean = remove_outliers_deciles(times)
            key = f"dispatch_{dispatch_id}_part_{part_id}"
            avg_dispatch_part_times[key] = statistics.mean(times_clean) if times_clean else 0

    return avg_dispatch_times, avg_dispatch_part_times, compile_time_ns


# ---------------------------------------------------------------------------
# Matrix download and discovery
# ---------------------------------------------------------------------------


def download_matrix_from_suitesparse(matrix_name: str) -> Optional[Tuple[str, Any, Optional[str], Optional[str]]]:
    SUITESPARSE_DIR.mkdir(exist_ok=True)

    matrix_info = get_matrix_info(matrix_name)
    if matrix_info is None:
        return None

    # 1. Prefer the shared ssgetpy cache (~/.ssgetpy), which find_vdia.py and
    #    other tools populate. If the matrix is already extracted there, reuse it
    #    and return tar_path/matrix_subdir as None so the caller's cleanup step
    #    leaves the shared copy untouched.
    cache_subdir, _ = matrix_info.localpath(format="MM", extract=True)
    cache_mtx = os.path.join(cache_subdir, f"{matrix_info.name}.mtx")
    if os.path.exists(cache_mtx):
        print(f"  Using cached matrix from {cache_mtx}")
        return cache_mtx, matrix_info, None, None

    # 2. Otherwise download into the project-local Suitesparse/ directory. These
    #    files are owned by this run and get cleaned up afterward (unless
    #    SABLE_NO_CLEANUP is set).
    destpath = str(SUITESPARSE_DIR / "MM" / matrix_info.group)
    print(f"  Not in cache; downloading {matrix_name} into {destpath} ...")
    matrix_subdir, tar_path = matrix_info.download(format="MM", destpath=destpath, extract=True)
    matrix_path = os.path.join(matrix_subdir, f"{matrix_info.name}.mtx")

    if not os.path.exists(matrix_path):
        print(f"Error: Matrix file not found at {matrix_path}")
        return None

    return matrix_path, matrix_info, tar_path, matrix_subdir


def get_available_matrices() -> List[str]:
    names = {f.stem for f in RESULTS_DIR.glob("*.yaml")}
    names.update(f.stem for f in BANDS_RESULTS_DIR.glob("*.yaml"))
    return sorted(names)


# The named matrix sets of the paper, as recorded in matrices.json. Extraction
# leaves YAML for every matrix it found structure in (117 for VBR+CSR), which
# is a superset of what the paper evaluates, so a bare run benchmarks far more
# than the paper reports. --matrix-set restricts to a paper set instead.
MATRIX_SETS_FILE = FILEPATH / "matrices.json"
PAPER_SET_GROUPS = ("vbr_csr", "vdia_only", "fukaya")


def get_matrix_set(name: str) -> List[str]:
    """Matrix names of a paper set: 'paper' (all 78) or one group of it."""
    with open(MATRIX_SETS_FILE) as f:
        spec = json.load(f)
    groups = PAPER_SET_GROUPS if name == "paper" else (name,)
    names = {m["name"] for g in groups for m in spec[g]["matrices"]}
    return sorted(names)


# ---------------------------------------------------------------------------
# SpMV kernel helpers
# ---------------------------------------------------------------------------


def _vbr_spmv_kernel(vbr_kernel: VBRKernel):
    if vbr_kernel == VBRKernel.MKL:
        return MKLVBRSpmv()
    if vbr_kernel == VBRKernel.MIXED:
        return MixedVBRSpmv()
    return NaiveVBRSpmv()


def _vdia_spmv_kernel(vdia_kernel: VDIAKernel):
    if vdia_kernel == VDIAKernel.NAIVE:
        return NaiveVDIASpmv()
    if vdia_kernel == VDIAKernel.MKL_DIA:
        from sable.kernels import MKLDIASpmv
        return MKLDIASpmv()
    raise ValueError(f"Unknown SpMV VDIA kernel: {vdia_kernel}")


def _csr_spmv_kernel(csr_kernel: CSRKernel):
    if csr_kernel == CSRKernel.MKL:
        return MKLCSRSpmv()
    if csr_kernel == CSRKernel.SPV8:
        return SPV8CSRSpmv()
    if csr_kernel == CSRKernel.UZP:
        return UZPCSRSpmv()
    if csr_kernel == CSRKernel.NAIVE:
        return NaiveCSRSpmv()
    raise ValueError(f"{csr_kernel.value} is not a frontend SpMV CSR kernel")


# ---------------------------------------------------------------------------
# SpMM kernel helpers
# ---------------------------------------------------------------------------


def _vbr_spmm_kernel(vbr_kernel: VBRKernel):
    if vbr_kernel == VBRKernel.MKL:
        return MKLVBRSpmm()
    if vbr_kernel == VBRKernel.MIXED:
        return MixedVBRSpmm()
    return NaiveVBRSpmm()


def _vdia_spmm_kernel(vdia_kernel: VDIAKernel):
    if vdia_kernel == VDIAKernel.NAIVE:
        return NaiveVDIASpmm()
    if vdia_kernel == VDIAKernel.MKL_DIA:
        from sable.kernels import MKLDIASpmm
        return MKLDIASpmm()
    raise ValueError(f"Unknown SpMM VDIA kernel: {vdia_kernel}")


def _csr_spmm_kernel(csr_kernel: CSRKernel):
    if csr_kernel == CSRKernel.MKL:
        return MKLCSRSpmm()
    if csr_kernel == CSRKernel.SPREG:
        return SPRegCSRSpmm()
    if csr_kernel == CSRKernel.NAIVE:
        return NaiveCSRSpmm()
    raise ValueError(f"{csr_kernel.value} is not a frontend SpMM CSR kernel")


# ---------------------------------------------------------------------------
# VBR conversion (shared structure, operation-specific RHS)
# ---------------------------------------------------------------------------


def _analyze_blocks_from_coords(
    block_coords: List[Tuple[int, int, int, int]],
    mat: scipy.sparse.spmatrix,
) -> List[Dict[str, Any]]:
    csr = mat.tocsr()
    regions = []
    for r_start, r_end, c_start, c_end in block_coords:
        rows = r_end - r_start
        cols = c_end - c_start
        block_nnz = csr[r_start:r_end, c_start:c_end].nnz
        block_size = rows * cols
        density = (block_nnz / block_size * 100) if block_size > 0 else 0
        regions.append({"rows": rows, "cols": cols, "density_percent": density, "nnz": block_nnz})
    return regions


def _analyze_bands_from_data(bands: list[dict[str, Any]]) -> List[Dict[str, Any]]:
    regions = []
    for band in bands:
        area = int(band.get("vdia_area", 0))
        nnz = int(band.get("total_nnz", 0))
        density = (nnz / area * 100) if area > 0 else 0
        regions.append({"rows": area, "cols": 1, "density_percent": density, "nnz": nnz})
    return regions


def _convert_and_prepare(
    operation: Operation,
    matrix_name: str,
    format_regions: list[Any],
    mat: scipy.sparse.spmatrix,
    format_kind: str,
    write_rhs: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    csr_mat = mat.tocsr()
    _, matrix_cols = csr_mat.shape

    # The dense right-hand side is runtime input, not generated code: the
    # emitted C only embeds its path and opens it when the benchmark runs.
    # Writing it costs ncols x SPMM_NRHS values of text -- gigabytes for a
    # wide matrix -- so codegen skips it and the benchmark path writes it.
    if write_rhs:
        if operation == Operation.SPMV:
            write_dense_vector(1.0, matrix_cols)
        else:
            write_dense_matrix(1.0, matrix_cols, SPMM_NRHS)

    if format_kind == "vdia":
        region_stats = _analyze_bands_from_data(format_regions)
    else:
        region_stats = _analyze_blocks_from_coords(format_regions, csr_mat)

    composed_data = {
        "format_kind": format_kind,
        "format_regions": list(format_regions),
        "region_stats": region_stats,
        "matrix": csr_mat,
    }
    baseline_data = {
        "matrix": csr_mat,
    }

    return composed_data, baseline_data


# ---------------------------------------------------------------------------
# Frontend compilation per operation
# ---------------------------------------------------------------------------


def _compile_spmv_frontend(
    matrix_name: str,
    matrix_source,
    format_kind: str,
    format_regions: list[Any],
    artifact_dir: str,
    format_kernel,
    csr_kernel: CSRKernel,
    bench_iterations: int,
    data_dir: str | None = None,
    data_key: str | None = None,
    write_rhs: bool = True,
):
    matrix = Matrix(matrix_source, name=matrix_name)
    if write_rhs:
        write_dense_vector(1.0, matrix.ncols)

    plan = Plan(matrix, artifact_dir=artifact_dir)
    plan.rhs(DenseInput.vector(_generated_vector_path(matrix.ncols), matrix.ncols))
    if format_regions:
        if format_kind == "vdia":
            vdia = plan.extract(BandExtractorSkip(format_regions))
            plan.dispatch(vdia, _vdia_spmv_kernel(format_kernel))
        elif format_kind == "vbr":
            vbr = plan.extract(BlockDetectorSkip(format_regions))
            plan.dispatch(vbr, _vbr_spmv_kernel(format_kernel))
        else:
            raise ValueError(f"Unknown format kind: {format_kind}")
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, _csr_spmv_kernel(csr_kernel))
    return plan.compile(
        filename=matrix_name,
        bench=bench_iterations,
        data_dir=data_dir,
        data_key=data_key,
    )


def _compile_spmm_frontend(
    matrix_name: str,
    matrix_source,
    format_kind: str,
    format_regions: list[Any],
    artifact_dir: str,
    format_kernel,
    csr_kernel: CSRKernel,
    bench_iterations: int,
    data_dir: str | None = None,
    data_key: str | None = None,
    write_rhs: bool = True,
):
    matrix = Matrix(matrix_source, name=matrix_name)
    if write_rhs:
        write_dense_matrix(1.0, matrix.ncols, SPMM_NRHS)

    plan = Plan(matrix, artifact_dir=artifact_dir)
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(matrix.ncols, SPMM_NRHS),
            shape=(matrix.ncols, SPMM_NRHS),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    if format_regions:
        if format_kind == "vdia":
            vdia = plan.extract(BandExtractorSkip(format_regions))
            plan.dispatch(vdia, _vdia_spmm_kernel(format_kernel))
        elif format_kind == "vbr":
            vbr = plan.extract(BlockDetectorSkip(format_regions))
            plan.dispatch(vbr, _vbr_spmm_kernel(format_kernel))
        else:
            raise ValueError(f"Unknown format kind: {format_kind}")
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, _csr_spmm_kernel(csr_kernel))
    return plan.compile(
        filename=matrix_name,
        bench=bench_iterations,
        data_dir=data_dir,
        data_key=data_key,
    )


# ---------------------------------------------------------------------------
# Result building and benchmarking
# ---------------------------------------------------------------------------


def _build_matrix_result(
    matrix_name: str,
    region_stats: list[dict[str, Any]],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    dispatch_times: dict[int, float],
    dispatch_part_times: dict[str, float],
    baseline_dispatch_times: dict[int, float],
    compile_time_composed_ns: float,
    compile_time_baseline_ns: float,
    codegen_time_composed_ms: int,
    codegen_time_baseline_ms: int,
    baseline_source: str,
) -> Dict[str, Any]:
    total_time = sum(dispatch_times.values())
    baseline_time = sum(baseline_dispatch_times.values())
    format_area = sum(region.get("rows", 0) * region.get("cols", 0) for region in region_stats)
    claimed_nnz = sum(region.get("nnz", 0) for region in region_stats)
    residual_nnz = matrix_nnz - claimed_nnz
    extra_values = format_area - claimed_nnz
    claimed_nnz_perc = (claimed_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
    residual_nnz_perc = (residual_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
    density_calculation = matrix_nnz / (matrix_rows * matrix_cols) if matrix_rows * matrix_cols > 0 else 0
    dispatch_timing = {
        f"dispatch_{dispatch_id}": {
            "time_ns": round(time_ns, 2),
            "percentage_of_total_time": round((time_ns / total_time * 100), 3) if total_time > 0 else 0,
        }
        for dispatch_id, time_ns in sorted(dispatch_times.items())
    }
    baseline_timing = {
        f"dispatch_{dispatch_id}": round(time_ns, 2)
        for dispatch_id, time_ns in sorted(baseline_dispatch_times.items())
    }

    result = {
        "matrix_name": matrix_name,
        "matrix_dimensions": {
            "rows": matrix_rows,
            "cols": matrix_cols,
            "nnz": matrix_nnz,
            "density": round(density_calculation, 3),
        },
        "timing": {
            "total_time_ns": round(total_time, 2),
            "dispatch_times": dispatch_timing,
            "dispatch_part_times": {key: round(value, 2) for key, value in sorted(dispatch_part_times.items())},
            "csr_baseline_time_ns": round(baseline_time, 2),
            "fully_sparse_baseline_time_ns": round(baseline_time, 2),
            "fully_sparse_baseline_source": baseline_source,
            "csr_baseline_dispatch_times": baseline_timing,
            "speedup": round((baseline_time / total_time), 3) if total_time > 0 else 0,
            "compile_time_composed_s": compile_time_composed_ns / 1e9 if compile_time_composed_ns else 0.0,
            "compile_time_csr_baseline_s": compile_time_baseline_ns / 1e9 if compile_time_baseline_ns else 0.0,
            "codegen_time_composed_ms": codegen_time_composed_ms,
            "codegen_time_csr_baseline_ms": codegen_time_baseline_ms,
        },
        "nnz": {
            "format_claimed_nnz": claimed_nnz,
            "residual_nnz": residual_nnz,
            "format_area": format_area,
            "extra_values": extra_values,
            "format_claimed_nnz_perc": round(claimed_nnz_perc, 2),
            "residual_nnz_perc": round(residual_nnz_perc, 2),
        },
    }
    return result


def _process_and_benchmark_frontend(
    operation: Operation,
    matrix_name: str,
    composed_data: Dict[str, Any],
    baseline_data: Dict[str, Any],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    bench_iterations: int,
    format_kernel,
    csr_kernel,
    threads: int = 1,
    baseline_source_mode: str = "run",
    baseline_results_dir: pathlib.Path = DEFAULT_BASELINE_RESULTS_DIR,
    allow_baseline_run_on_missing: bool = False,
    codegen_only: bool = False,
) -> Optional[Dict[str, Any]]:
    if threads != 1:
        raise ValueError("The frontend benchmark path is single-threaded for now")

    if operation == Operation.SPMV:
        csr_label = csr_kernel.value
        compile_fn = _compile_spmv_frontend
        dir_prefix = "Generated_SpMV_C"
    else:
        csr_label = csr_kernel.value
        compile_fn = _compile_spmm_frontend
        dir_prefix = "Generated_SpMM_C"

    format_kind = composed_data["format_kind"]
    variant_name = f"{format_kernel.value}_{csr_label}"
    codegen_root = pathlib.Path(os.environ.get("SABLE_CODEGEN_DIR") or str(FILEPATH))
    base_codegen_dir = codegen_root / f"{dir_prefix}_{format_kernel.value}_{csr_label}"
    # The staged data depends only on the extraction, so every kernel variant
    # of this matrix shares one file here instead of writing its own copy.
    staged_data_dir = str(codegen_root / "Generated_Staged_Data")
    codegen_dir_composed = str(base_codegen_dir / "composed")
    codegen_dir_baseline = str(base_codegen_dir / "csr_baseline")
    os.makedirs(codegen_dir_composed, exist_ok=True)
    if baseline_source_mode == "run" or allow_baseline_run_on_missing:
        os.makedirs(codegen_dir_baseline, exist_ok=True)

    print(f"  [{variant_name}] Generating frontend C code (composed)...")
    composed_executor = compile_fn(
        matrix_name,
        composed_data["matrix"],
        format_kind,
        composed_data["format_regions"],
        codegen_dir_composed,
        format_kernel,
        csr_kernel,
        bench_iterations,
        data_dir=staged_data_dir,
        data_key=f"{matrix_name}_{format_kind}",
        write_rhs=not codegen_only,
    )

    if codegen_only:
        os.makedirs(codegen_dir_baseline, exist_ok=True)
        print(f"  [{variant_name}] Generating frontend C code (CSR baseline)...")
        compile_fn(
            matrix_name,
            baseline_data["matrix"],
            format_kind,
            [],
            codegen_dir_baseline,
            format_kernel,
            csr_kernel,
            bench_iterations,
            data_dir=staged_data_dir,
            data_key=f"{matrix_name}_csr_baseline",
            write_rhs=not codegen_only,
        )
        print(f"  [{variant_name}] Codegen only; skipping compilation and timing")
        return None

    print(f"  [{variant_name}] Evaluating composed version...")
    dispatch_times, dispatch_part_times, compile_time_composed_ns = eval_frontend_executor_timings(
        composed_executor, bench_iterations, threads=threads
    )

    baseline_dispatch_times: dict[int, float]
    compile_time_baseline_ns = 0.0
    codegen_time_baseline_ms = 0
    baseline_source = ""

    if baseline_source_mode == "existing":
        try:
            baseline_time_ns, baseline_source = _lookup_fully_sparse_baseline(
                operation,
                csr_kernel,
                matrix_name,
                threads,
                baseline_results_dir,
            )
            baseline_dispatch_times = {1: baseline_time_ns}
            print(f"  [{variant_name}] Using existing fully sparse baseline: {baseline_source}")
        except Exception as exc:
            if not allow_baseline_run_on_missing:
                print(f"  [{variant_name}] Missing existing fully sparse baseline: {exc}")
                return None
            print(f"  [{variant_name}] Existing baseline missing ({exc}); running CSR baseline instead")
            baseline_source_mode = "run"

    if baseline_source_mode == "run":
        print(f"  [{variant_name}] Generating frontend C code (CSR baseline)...")
        baseline_executor = compile_fn(
            matrix_name,
            baseline_data["matrix"],
            format_kind,
            [],
            codegen_dir_baseline,
            format_kernel,
            csr_kernel,
            bench_iterations,
            data_dir=staged_data_dir,
            data_key=f"{matrix_name}_csr_baseline",
            write_rhs=not codegen_only,
        )

        print(f"  [{variant_name}] Evaluating CSR baseline...")
        baseline_dispatch_times, _, compile_time_baseline_ns = eval_frontend_executor_timings(
            baseline_executor, bench_iterations, threads=threads
        )
        codegen_time_baseline_ms = baseline_executor.codegen_time_ms
        baseline_source = "measured_in_this_run"

    return _build_matrix_result(
        matrix_name,
        composed_data["region_stats"],
        matrix_rows,
        matrix_cols,
        matrix_nnz,
        dispatch_times,
        dispatch_part_times,
        baseline_dispatch_times,
        compile_time_composed_ns,
        compile_time_baseline_ns,
        composed_executor.codegen_time_ms,
        codegen_time_baseline_ms,
        baseline_source,
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
    matrix_entry["timing"][thread_key] = thread_timing
    with open(output_file, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"  [{results_key}] Results written to {output_file}")


# ---------------------------------------------------------------------------
# Kernel resolution per operation
# ---------------------------------------------------------------------------


def _resolve_csr_kernels(operation: Operation, csr_arg: str, parser: argparse.ArgumentParser):
    if operation == Operation.SPMV:
        valid = {k.value for k in SPMV_CSR_KERNELS}
        if csr_arg == "all":
            return list(SPMV_CSR_KERNELS)
        requested = [name.strip() for name in csr_arg.split(",")]
        selected = [CSRKernel(n) for n in requested if n in valid]
        skipped = [n for n in requested if n not in valid]
        if skipped:
            print(f"  [spmv] Skipping CSR kernels not available for SpMV: {skipped}")
        if not selected:
            parser.error(f"No valid SpMV CSR kernels. Valid options: {valid}")
        return selected
    else:
        valid = {k.value for k in SPMM_CSR_KERNELS}
        if csr_arg == "all":
            return list(SPMM_CSR_KERNELS)
        requested = [name.strip() for name in csr_arg.split(",")]
        selected = [CSRKernel(n) for n in requested if n in valid]
        skipped = [n for n in requested if n not in valid]
        if skipped:
            print(f"  [spmm] Skipping CSR kernels not available for SpMM: {skipped}")
        if not selected:
            parser.error(f"No valid SpMM CSR kernels. Valid options: {valid}")
        return selected


def _resolve_vbr_kernels(arg: str, parser: argparse.ArgumentParser) -> list[VBRKernel]:
    if arg == "none":
        return []
    if arg == "all":
        return list(VBRKernel)
    names = [name.strip() for name in arg.split(",")]
    valid = {kernel.value for kernel in VBRKernel} | {"blocksmixed"}
    invalid = set(names) - valid
    if invalid:
        parser.error(f"Invalid VBR kernel(s): {invalid}. Valid options: {valid}")
    return [VBRKernel(name) for name in names]


def _resolve_vdia_kernels(arg: str, parser: argparse.ArgumentParser) -> list[VDIAKernel]:
    if arg == "none":
        return []
    if arg == "all":
        return list(VDIAKernel)
    names = [name.strip() for name in arg.split(",")]
    valid = {kernel.value for kernel in VDIAKernel}
    invalid = set(names) - valid
    if invalid:
        parser.error(f"Invalid VDIA kernel(s): {invalid}. Valid options: {valid}")
    return [VDIAKernel(name) for name in names]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SABLE sparse matrix operations (SpMV / SpMM)",
        epilog=(
            "Examples:\n"
            "  %(prog)s --operation spmv,spmm eris1176 bloweybl\n"
            "  %(prog)s --operation spmv --csr-kernels naive,mkl,spv8 --vbr-kernels blocknaive,blockmkl --vdia-kernels bandnaive\n"
            "  %(prog)s --operation spmv,spmm --vbr-kernels none --vdia-kernels bandnaive --baseline-source existing bcsstk13\n"
            "  %(prog)s  # both operations, all matrices, all kernels"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--operation", type=str, default="spmv,spmm",
                        help="Comma-separated operations: spmv, spmm, or spmv,spmm (default: spmv,spmm)")
    parser.add_argument("matrices", nargs="*", help="Matrix names to benchmark.")
    parser.add_argument("--matrices", dest="matrices_flag", nargs="*", metavar="MATRIX")
    parser.add_argument("--matrix-set", choices=("paper",) + PAPER_SET_GROUPS,
                        default=None,
                        help="Benchmark a named matrix set from matrices.json "
                             "instead of every extracted matrix: 'paper' is the "
                             "78 the evaluation reports. Output keeps the "
                             "canonical sable_<op>_<kernels>.json names.")
    parser.add_argument("--bench", type=int, default=None,
                        help=f"Benchmark iterations (default: {DEFAULT_SPMV_BENCH_ITERATIONS} for spmv, {DEFAULT_SPMM_BENCH_ITERATIONS} for spmm)")
    parser.add_argument("--codegen-only", action="store_true",
                        help="Generate the C and .sabledata for each matrix and configuration, "
                             "then stop: nothing is compiled, run, or timed, and no results JSON "
                             "is written. The dense right-hand sides are not written either -- "
                             "they are runtime input the generated C opens when it runs, and a "
                             "benchmark run writes them. Needs only the matrices and the "
                             "extraction YAML, so it runs without build_native.sh and without "
                             "AVX-512.")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--csr-kernels", type=str, default="all",
                        help="SpMV: naive,spv8,mkl,uzp. SpMM: naive,mkl,spreg. Invalid names silently skipped per operation.")
    parser.add_argument("--vbr-kernels", type=str, default="all",
                        help="blocknaive, blockmixed, blockmkl, all, none, or comma-separated")
    parser.add_argument("--vdia-kernels", type=str, default="all",
                        help="bandnaive, all, none, or comma-separated")
    parser.add_argument("--threads", type=str, default="1")
    parser.add_argument("--baseline-source", choices=("existing", "run"), default="run",
                        help="Use fully sparse timings from --baseline-results-dir, or rerun CSR baselines (default: run)")
    parser.add_argument("--baseline-results-dir", type=str, default=str(DEFAULT_BASELINE_RESULTS_DIR),
                        help="Directory containing sable_<op>_mkl_<csr>.json fully sparse timing files")
    parser.add_argument("--allow-baseline-run-on-missing", action="store_true",
                        help="If --baseline-source existing is missing an entry, run the CSR baseline instead")
    parser.add_argument("--bands-results-dir", type=str, default=None,
                        help="Override directory for band YAML files (default: find-submatrices/results_bands/)")
    args = parser.parse_args()

    global BANDS_RESULTS_DIR
    if args.bands_results_dir:
        _bands_path = pathlib.Path(args.bands_results_dir)
        if not _bands_path.is_absolute():
            _bands_path = FILEPATH / _bands_path
        BANDS_RESULTS_DIR = _bands_path

    operations = [Operation(op.strip()) for op in args.operation.split(",")]

    thread_counts = [int(t.strip()) for t in args.threads.split(",")]
    if any(thread_count != 1 for thread_count in thread_counts):
        parser.error("The frontend benchmark path is single-threaded for now; use --threads 1")

    vbr_kernels = _resolve_vbr_kernels(args.vbr_kernels, parser)
    vdia_kernels = _resolve_vdia_kernels(args.vdia_kernels, parser)
    if not vbr_kernels and not vdia_kernels:
        parser.error("At least one VBR or VDIA kernel must be selected")

    matrices = args.matrices or args.matrices_flag
    specific_matrices_requested = matrices is not None and len(matrices) > 0
    if specific_matrices_requested and args.matrix_set:
        parser.error("--matrix-set and an explicit matrix list are mutually exclusive")
    if args.matrix_set:
        # A named set is the whole run, not a subselection of one, so it keeps
        # the canonical filenames the plotting scripts read.
        matrices = get_matrix_set(args.matrix_set)
    elif not matrices:
        matrices = get_available_matrices()

    if specific_matrices_requested and len(matrices) == 1:
        output_suffix = f"_{matrices[0]}"
    elif specific_matrices_requested:
        joined = "_".join(matrices)
        if len(joined) > 100:
            import hashlib
            output_suffix = f"_{len(matrices)}matrices_{hashlib.md5(joined.encode()).hexdigest()[:8]}"
        else:
            output_suffix = "_" + joined
    else:
        output_suffix = ""

    ops_label = "+".join(op.value.upper() for op in operations)
    print(f"[{ops_label}] Will process {len(matrices)} matrices")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    baseline_results_dir = pathlib.Path(args.baseline_results_dir)
    if not baseline_results_dir.is_absolute():
        baseline_results_dir = FILEPATH / baseline_results_dir
    SUITESPARSE_DIR.mkdir(exist_ok=True)

    all_results: dict[str, list[dict[str, Any]]] = {}

    for matrix_name in matrices:
        yaml_path = RESULTS_DIR / f"{matrix_name}.yaml"
        bands_yaml_path = BANDS_RESULTS_DIR / f"{matrix_name}.yaml"
        if not yaml_path.exists() and not bands_yaml_path.exists():
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

            if yaml_path.exists():
                print(f"  Parsing VBR blocks from {yaml_path}...")
                block_coords = parse_yaml_blocks(str(yaml_path))
            else:
                block_coords = []
            print(f"  Found {len(block_coords)} VBR blocks")

            if bands_yaml_path.exists():
                print(f"  Parsing VDIA bands from {bands_yaml_path}...")
                vdia_bands = parse_yaml_bands(str(bands_yaml_path))
            else:
                vdia_bands = []
            print(f"  Found {len(vdia_bands)} VDIA bands")

            for operation in operations:
                op_label = operation.value.upper()
                csr_kernels = _resolve_csr_kernels(operation, args.csr_kernels, parser)
                bench_iterations = args.bench
                if bench_iterations is None:
                    bench_iterations = DEFAULT_SPMV_BENCH_ITERATIONS if operation == Operation.SPMV else DEFAULT_SPMM_BENCH_ITERATIONS

                print(f"\n  === Converting to frontend formats ({op_label}) ===")
                vbr_data, baseline_data = _convert_and_prepare(
                    operation, matrix_name, block_coords, A, "vbr",
                    write_rhs=not args.codegen_only,
                )
                vdia_data, _ = _convert_and_prepare(
                    operation, matrix_name, vdia_bands, A, "vdia",
                    write_rhs=not args.codegen_only,
                )
                format_runs = [
                    ("vbr", kernel, vbr_data)
                    for kernel in vbr_kernels
                ] + [
                    ("vdia", kernel, vdia_data)
                    for kernel in vdia_kernels
                ]

                for num_threads in thread_counts:
                    for format_kind, format_kernel, composed_data in format_runs:
                        for csr_kernel in csr_kernels:
                            csr_label = csr_kernel.value
                            results_key = f"{operation.value}_{format_kernel.value}_{csr_label}"
                            if args.codegen_only:
                                print(f"\n  === Generating {results_key} code ===")
                            else:
                                print(f"\n  === Running {results_key} benchmark (threads={num_threads}) ===")

                            result = _process_and_benchmark_frontend(
                                operation,
                                matrix_name,
                                composed_data,
                                baseline_data,
                                matrix_rows,
                                matrix_cols,
                                matrix_nnz,
                                bench_iterations,
                                format_kernel=format_kernel,
                                csr_kernel=csr_kernel,
                                threads=num_threads,
                                baseline_source_mode=args.baseline_source,
                                baseline_results_dir=baseline_results_dir,
                                allow_baseline_run_on_missing=args.allow_baseline_run_on_missing,
                                codegen_only=args.codegen_only,
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
                if not os.environ.get("SABLE_NO_CLEANUP"):
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
                    dispatch_times = thread_timing.get("dispatch_times", {})
                    dispatch_summary = ", ".join(
                        f"{name}: {info.get('time_ns', 0):.0f}ns"
                        for name, info in dispatch_times.items()
                    )
                    print(
                        f"  {result['matrix_name']} ({thread_key}): "
                        f"total: {thread_timing['total_time_ns']:.0f}ns, "
                        f"CSR baseline: {thread_timing.get('csr_baseline_time_ns', 0):.0f}ns, "
                        f"speedup: {thread_timing.get('speedup', 0):.3f}x"
                        + (f", {dispatch_summary}" if dispatch_summary else "")
                    )

    print(f"\nResults written to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
