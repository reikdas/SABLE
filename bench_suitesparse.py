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
import resource
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy
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
    MKLDIASpmm,
    MKLDIASpmv,
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
from utils.fileio import (
    dense_matrix_path,
    dense_vector_path,
    parse_yaml_bands,
    parse_yaml_blocks,
    write_dense_matrix,
    write_dense_vector,
)


FILEPATH = pathlib.Path(__file__).resolve().parent

COMPILE_TIMEOUT = 60 * 60 * 4
DEFAULT_SPMV_BENCH_ITERATIONS = 30
DEFAULT_SPMM_BENCH_ITERATIONS = 10
PHYSICAL_CORES = list(range(os.cpu_count() or 20))
SPMM_NRHS = 512

SPMV_CSR_KERNELS = (CSRKernel.NAIVE, CSRKernel.MKL, CSRKernel.SPV8, CSRKernel.UZP)
SPMM_CSR_KERNELS = (CSRKernel.NAIVE, CSRKernel.MKL, CSRKernel.SPREG)

# The dense-side dispatch for VDIA, per operation. SpMV chooses between the
# naive band kernel and MKL's DIA; SpMM runs the naive one only, which is what
# plots/fukaya_results.py already reads back (sable_spmm_bandnaive_*, against
# sable_spmv_bandmkl_* on the SpMV side).
SPMV_VDIA_KERNELS = (VDIAKernel.NAIVE, VDIAKernel.MKL_DIA)
SPMM_VDIA_KERNELS = (VDIAKernel.NAIVE,)

SUITESPARSE_DIR = pathlib.Path(os.environ.get("SABLE_SUITESPARSE_DIR") or str(FILEPATH / "Suitesparse"))
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"
BANDS_RESULTS_DIR = FILEPATH / "find-submatrices" / "results_bands_075"
DEFAULT_BASELINE_RESULTS_DIR = FILEPATH / "results"
BASELINE_TIME_KEY = "csr_baseline_time_ns"
_RESULTS_JSON_CACHE: dict[pathlib.Path, list[dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def set_ulimit():
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def remove_outliers_deciles(data):
    """Drop values outside the 10th..90th percentiles"""
    if len(data) < 10:
        return data
    d1 = numpy.percentile(data, 10)
    d9 = numpy.percentile(data, 90)
    return [x for x in data if d1 <= x <= d9]


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
    preferred = baseline_results_dir / f"sable_{operation.value}_blockmixed_{csr_kernel.value}.json"
    if preferred.exists():
        return preferred

    candidates = sorted(baseline_results_dir.glob(f"sable_{operation.value}_*_{csr_kernel.value}.json"))
    for candidate in candidates:
        try:
            data = _load_results_json(candidate)
        except Exception:
            continue
        if any(BASELINE_TIME_KEY in timing for entry in data for timing in entry.get("timing", {}).values()):
            return candidate

    raise FileNotFoundError(
        f"No results file with CSR baselines found for {operation.value}/{csr_kernel.value} in {baseline_results_dir}"
    )


def _lookup_existing_baseline(
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
        if BASELINE_TIME_KEY not in thread_timing:
            raise KeyError(f"{baseline_file} has no {BASELINE_TIME_KEY} for {matrix_name} ({thread_key})")
        baseline_time = float(thread_timing[BASELINE_TIME_KEY])
        source = f"{baseline_file.name}:{matrix_name}:{thread_key}:{BASELINE_TIME_KEY}"
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

# What the evaluation ran, and so what a --matrix-set run does unless kernels
# are named: the mixed block kernel on the VBR+CSR set, and, on every matrix
# with bands, MKL's DIA for SpMV and the naive band kernel for SpMM.
PAPER_VBR_KERNELS = (VBRKernel.MIXED,)
PAPER_VBR_SET = "vbr_csr"
PAPER_VDIA_KERNELS = {Operation.SPMV: (VDIAKernel.MKL_DIA,), Operation.SPMM: (VDIAKernel.NAIVE,)}


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
# Frontend compilation
# ---------------------------------------------------------------------------


def _format_kernel_for(operation: Operation, format_kind: str, format_kernel):
    if format_kind == "vdia":
        return _vdia_spmv_kernel(format_kernel) if operation == Operation.SPMV else _vdia_spmm_kernel(format_kernel)
    if format_kind == "vbr":
        return _vbr_spmv_kernel(format_kernel) if operation == Operation.SPMV else _vbr_spmm_kernel(format_kernel)
    raise ValueError(f"Unknown format kind: {format_kind}")


def _compile_frontend(
    operation: Operation,
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
    num_threads: int = 1,
):
    """Extract the format's regions, dispatch them and the CSR residual, and
    generate the program. With no regions this is the CSR-only baseline."""
    matrix = Matrix(matrix_source, name=matrix_name)
    plan = Plan(matrix, artifact_dir=artifact_dir)
    # Codegen only embeds the right-hand side's path; a benchmark run writes the file it names.
    if operation == Operation.SPMV:
        rhs_path = write_dense_vector(1.0, matrix.ncols) if write_rhs else dense_vector_path(matrix.ncols)
        plan.rhs(DenseInput.vector(rhs_path, matrix.ncols))
    else:
        if write_rhs:
            rhs_path = write_dense_matrix(1.0, matrix.ncols, SPMM_NRHS)
        else:
            rhs_path = dense_matrix_path(matrix.ncols, SPMM_NRHS)
        plan.rhs(DenseInput.matrix(rhs_path, shape=(matrix.ncols, SPMM_NRHS), layout=DenseLayout.ROW_MAJOR))
    if format_regions:
        extractor = BandExtractorSkip(format_regions) if format_kind == "vdia" else BlockDetectorSkip(format_regions)
        fmt = plan.extract(extractor)
        plan.dispatch(fmt, _format_kernel_for(operation, format_kind, format_kernel), num_threads=num_threads)
    csr = plan.extract(CSRConvertor())
    csr_kernel_obj = _csr_spmv_kernel(csr_kernel) if operation == Operation.SPMV else _csr_spmm_kernel(csr_kernel)
    plan.dispatch(csr, csr_kernel_obj, num_threads=num_threads)
    return plan.compile(filename=matrix_name, bench=bench_iterations, data_dir=data_dir, data_key=data_key)


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
    staged_data_time_composed_ms: int = 0,
    staged_data_time_baseline_ms: int = 0,
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
            "csr_baseline_source": baseline_source,
            "csr_baseline_dispatch_times": baseline_timing,
            "speedup": round((baseline_time / total_time), 3) if total_time > 0 else 0,
            "compile_time_composed_s": compile_time_composed_ns / 1e9 if compile_time_composed_ns else 0.0,
            "compile_time_csr_baseline_s": compile_time_baseline_ns / 1e9 if compile_time_baseline_ns else 0.0,
            # Emitting the C. Writing the staged data is timed on its own, and is
            # zero for a variant that reused the file an earlier variant wrote.
            "codegen_time_composed_ms": codegen_time_composed_ms,
            "codegen_time_csr_baseline_ms": codegen_time_baseline_ms,
            "staged_data_time_composed_ms": staged_data_time_composed_ms,
            "staged_data_time_csr_baseline_ms": staged_data_time_baseline_ms,
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
    baseline_cache: dict | None = None,
) -> Optional[Dict[str, Any]]:
    csr_label = csr_kernel.value
    dir_prefix = "Generated_SpMV_C" if operation == Operation.SPMV else "Generated_SpMM_C"
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

    def compile_baseline():
        os.makedirs(codegen_dir_baseline, exist_ok=True)
        print(f"  [{variant_name}] Generating frontend C code (CSR baseline)...")
        return _compile_frontend(
            operation,
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
            num_threads=threads,
        )

    print(f"  [{variant_name}] Generating frontend C code (composed)...")
    composed_executor = _compile_frontend(
        operation,
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
        num_threads=threads,
    )

    if codegen_only:
        compile_baseline()
        print(f"  [{variant_name}] Codegen only; skipping compilation and timing")
        return None

    print(f"  [{variant_name}] Evaluating composed version...")
    dispatch_times, dispatch_part_times, compile_time_composed_ns = eval_frontend_executor_timings(
        composed_executor, bench_iterations, threads=threads
    )
    if not dispatch_times:
        # The program failed to compile, timed out, or never printed a timing.
        # A record built from that would read as a measured time of zero.
        print(f"  [{variant_name}] Composed program produced no timings; nothing recorded for {matrix_name}")
        return None

    # The CSR-only program depends on the CSR kernel and the thread count, not
    # on the format kernel, so one measurement serves every variant of a matrix.
    baseline_key = (operation.value, csr_label, threads)
    baseline = baseline_cache.get(baseline_key) if baseline_cache is not None else None
    if baseline is not None:
        print(f"  [{variant_name}] Reusing this matrix's {csr_label} CSR baseline")

    if baseline is None and baseline_source_mode == "existing":
        try:
            baseline_time_ns, baseline_source = _lookup_existing_baseline(
                operation,
                csr_kernel,
                matrix_name,
                threads,
                baseline_results_dir,
            )
            baseline = {"dispatch_times": {1: baseline_time_ns}, "compile_time_ns": 0.0,
                        "codegen_time_ms": 0, "staged_data_time_ms": 0, "source": baseline_source}
            print(f"  [{variant_name}] Using existing CSR baseline: {baseline_source}")
        except Exception as exc:
            if not allow_baseline_run_on_missing:
                print(f"  [{variant_name}] Missing existing CSR baseline: {exc}")
                return None
            print(f"  [{variant_name}] Existing baseline missing ({exc}); running CSR baseline instead")

    if baseline is None:
        baseline_executor = compile_baseline()
        print(f"  [{variant_name}] Evaluating CSR baseline...")
        baseline_dispatch_times, _, compile_time_baseline_ns = eval_frontend_executor_timings(
            baseline_executor, bench_iterations, threads=threads
        )
        if not baseline_dispatch_times:
            print(f"  [{variant_name}] CSR baseline produced no timings; nothing recorded for {matrix_name}")
            return None
        baseline = {"dispatch_times": baseline_dispatch_times, "compile_time_ns": compile_time_baseline_ns,
                    "codegen_time_ms": baseline_executor.codegen_time_ms,
                    "staged_data_time_ms": baseline_executor.staged_data_time_ms,
                    "source": "measured_in_this_run"}

    if baseline_cache is not None:
        baseline_cache[baseline_key] = baseline

    return _build_matrix_result(
        matrix_name,
        composed_data["region_stats"],
        matrix_rows,
        matrix_cols,
        matrix_nnz,
        dispatch_times,
        dispatch_part_times,
        baseline["dispatch_times"],
        compile_time_composed_ns,
        baseline["compile_time_ns"],
        composed_executor.codegen_time_ms,
        baseline["codegen_time_ms"],
        baseline["source"],
        composed_executor.staged_data_time_ms,
        baseline["staged_data_time_ms"],
    )


def _output_entries(
    all_results: dict[str, list[dict[str, Any]]],
    results_key: str,
    output_file: pathlib.Path,
) -> list[dict[str, Any]]:
    """The entries of one output file, read once per run and then kept in step with it."""
    if results_key not in all_results:
        entries: list[dict[str, Any]] = []
        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else []
        all_results[results_key] = entries
    return all_results[results_key]


def _already_measured(entries: list[dict[str, Any]], matrix_name: str, num_threads: int) -> bool:
    """True if this driver has already timed the matrix at this thread count.

    dispatch_times is what a run of this driver records, so its presence tells
    a finished measurement from an entry that came from somewhere else.
    """
    for entry in entries:
        if entry.get("matrix_name") == matrix_name:
            timing = entry.get("timing", {}).get(_thread_key(num_threads), {})
            return bool(timing.get("dispatch_times")) and timing.get("total_time_ns", 0) > 0
    return False


def _append_result(
    all_results: dict[str, list[dict[str, Any]]],
    results_key: str,
    matrix_name: str,
    result: dict[str, Any],
    num_threads: int,
    output_file: pathlib.Path,
) -> None:
    """Merge one matrix's result into the output file.

    The file's existing entries are the starting point, so a run over some of
    the matrices updates those and leaves the others, and the other thread
    counts of this matrix, in place.
    """
    results_list = _output_entries(all_results, results_key, output_file)
    existing_idx = next((i for i, r in enumerate(results_list) if r["matrix_name"] == matrix_name), None)
    timing = dict(results_list[existing_idx].get("timing", {})) if existing_idx is not None else {}
    timing[_thread_key(num_threads)] = dict(result["timing"])
    matrix_entry = {
        "matrix_name": result["matrix_name"],
        "matrix_dimensions": result["matrix_dimensions"],
        "timing": timing,
        "nnz": result["nnz"],
    }
    if existing_idx is not None:
        results_list[existing_idx] = matrix_entry
        print(f"  [{results_key}] Updating result for {matrix_name}")
    else:
        results_list.append(matrix_entry)
        print(f"  [{results_key}] Added new result for {matrix_name}")

    # Write beside the file and rename, so an interrupted run never leaves a
    # truncated results file behind.
    tmp_file = output_file.with_name(output_file.name + f".tmp.{os.getpid()}")
    with open(tmp_file, "w") as f:
        json.dump(results_list, f, indent=2)
    os.replace(tmp_file, output_file)
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


_UNBUILT_REPORTED: set[tuple[str, str]] = set()


def _buildable_csr_kernels(operation: Operation, csr_kernels: list[CSRKernel]) -> list[CSRKernel]:
    """Drop the CSR kernels whose native component is not built on this machine.

    A kernel names what it links in source_files() (SpV8 its object file, for
    one). When such a file is missing every program using the kernel fails at
    the link step, once per matrix and per format kernel, so it is reported
    once here and the kernel is left out of the run instead.
    """
    buildable = []
    for csr_kernel in csr_kernels:
        kernel = _csr_spmv_kernel(csr_kernel) if operation == Operation.SPMV else _csr_spmm_kernel(csr_kernel)
        source_files = getattr(kernel, "source_files", None)
        missing = [path for path in (source_files() if source_files else []) if not os.path.exists(path)]
        if not missing:
            buildable.append(csr_kernel)
            continue
        if (operation.value, csr_kernel.value) not in _UNBUILT_REPORTED:
            _UNBUILT_REPORTED.add((operation.value, csr_kernel.value))
            others = f" and {len(missing) - 1} more" if len(missing) > 1 else ""
            print(f"  [{operation.value}] Skipping the {csr_kernel.value} CSR kernel: it links {missing[0]}{others}, "
                  "which does not exist. build_native.sh builds the native components (it leaves SpV8 out "
                  "on a CPU without AVX-512, which SpV8 requires); a missing source file means the "
                  "submodule is not checked out.")
    return buildable


def _resolve_vbr_kernels(arg: str, parser: argparse.ArgumentParser) -> list[VBRKernel]:
    if arg == "none":
        return []
    if arg == "all":
        return list(VBRKernel)
    names = [name.strip() for name in arg.split(",")]
    valid = {kernel.value for kernel in VBRKernel}
    invalid = set(names) - valid
    if invalid:
        parser.error(f"Invalid VBR kernel(s): {invalid}. Valid options: {valid}")
    return [VBRKernel(name) for name in names]


def _resolve_vdia_kernels(operation: Operation, arg: str,
                          parser: argparse.ArgumentParser) -> list[VDIAKernel]:
    """The VDIA kernels to run for one operation.

    Unlike VBR, the available dense-side dispatch depends on the operation, so
    this takes the operation the way _resolve_csr_kernels does. A name that is
    real but not offered for this operation is skipped with a note rather than
    being an error, so "--vdia-kernels bandmkl --operation spmv,spmm" still
    runs the SpMV half. An empty result is allowed: VDIA is optional, and the
    run falls back to the VBR entries of format_runs.
    """
    available = SPMV_VDIA_KERNELS if operation == Operation.SPMV else SPMM_VDIA_KERNELS
    if arg == "none":
        return []
    if arg == "all":
        return list(available)
    names = [name.strip() for name in arg.split(",")]
    invalid = set(names) - {kernel.value for kernel in VDIAKernel}
    if invalid:
        parser.error(f"Invalid VDIA kernel(s): {invalid}. Valid options: "
                     f"{ {kernel.value for kernel in VDIAKernel} }")
    valid = {kernel.value for kernel in available}
    skipped = [name for name in names if name not in valid]
    if skipped:
        print(f"  [{operation.value}] Skipping VDIA kernels not available for "
              f"{operation.value.upper()}: {skipped}")
    return [VDIAKernel(name) for name in names if name in valid]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    global BANDS_RESULTS_DIR, COMPILE_TIMEOUT
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
    parser.add_argument("--vbr-kernels", type=str, default=None,
                        help="blocknaive, blockmixed, blockmkl, all, none, or comma-separated "
                             "(default: all; with --matrix-set, blockmixed on the VBR+CSR set, "
                             "which is what the evaluation reports)")
    parser.add_argument("--vdia-kernels", type=str, default=None,
                        help="bandnaive, bandmkl, all, none, or comma-separated. "
                             "SpMV offers both; SpMM offers bandnaive only, and "
                             "names not available for an operation are skipped "
                             "for that operation. (default: all; with --matrix-set, "
                             "bandmkl for SpMV and bandnaive for SpMM, as in the evaluation)")
    parser.add_argument("--threads", type=str, default="1",
                        help="Comma-separated thread counts; each is a separate run pinned to that "
                             "many cores and recorded under its own '<n> thread' key (default: 1)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip every (matrix, kernels, thread count) this driver has already "
                             "timed into the output file, for resuming a run that stopped part-way. "
                             "A matrix with nothing left to run is not even loaded.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the configurations that would run for each matrix and stop, "
                             "without loading a matrix: how large a run is before starting it")
    parser.add_argument("--compile-timeout", type=int, default=COMPILE_TIMEOUT, metavar="SECONDS",
                        help="Give up on compiling one generated program after this long "
                             "(default: %(default)s, i.e. four hours)")
    parser.add_argument("--baseline-source", choices=("existing", "run"), default="run",
                        help="Take CSR baselines from --baseline-results-dir, or measure them in this run (default: run)")
    parser.add_argument("--baseline-results-dir", type=str, default=str(DEFAULT_BASELINE_RESULTS_DIR),
                        help="Directory holding sable_<op>_blockmixed_<csr>.json files whose entries carry csr_baseline_time_ns")
    parser.add_argument("--allow-baseline-run-on-missing", action="store_true",
                        help="If --baseline-source existing is missing an entry, run the CSR baseline instead")
    parser.add_argument("--bands-results-dir", type=str, default=None,
                        help="Override directory for band YAML files (default: find-submatrices/results_bands/)")
    args = parser.parse_args()

    if args.bands_results_dir:
        _bands_path = pathlib.Path(args.bands_results_dir)
        if not _bands_path.is_absolute():
            _bands_path = FILEPATH / _bands_path
        BANDS_RESULTS_DIR = _bands_path

    COMPILE_TIMEOUT = args.compile_timeout

    operations = [Operation(op.strip()) for op in args.operation.split(",")]

    thread_counts = [int(t.strip()) for t in args.threads.split(",")]
    if any(thread_count < 1 for thread_count in thread_counts):
        parser.error("--threads counts must be at least 1")

    # A --matrix-set run reproduces the evaluation, so unless kernels are named
    # it runs the evaluation's kernels, not every kernel on every matrix.
    paper_vbr = bool(args.matrix_set) and args.vbr_kernels is None
    paper_vdia = bool(args.matrix_set) and args.vdia_kernels is None
    vbr_kernels = list(PAPER_VBR_KERNELS) if paper_vbr else _resolve_vbr_kernels(args.vbr_kernels or "all", parser)
    vdia_arg = args.vdia_kernels or "all"
    # VDIA is resolved per operation, since which kernels exist depends on it.
    # "none" is the only argument that selects nothing for every operation.
    if not vbr_kernels and not paper_vdia and vdia_arg == "none":
        parser.error("At least one VBR or VDIA kernel must be selected")
    paper_vbr_set = set(get_matrix_set(PAPER_VBR_SET)) if paper_vbr else None
    if paper_vbr or paper_vdia:
        print("Matrix set given without kernels: running the evaluation's configuration "
              "(blockmixed on the VBR+CSR set; bandmkl for SpMV and bandnaive for SpMM on "
              "matrices with bands). Name --vbr-kernels/--vdia-kernels to run something else.")

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
    if not output_dir.is_absolute():
        output_dir = FILEPATH / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() != DEFAULT_BASELINE_RESULTS_DIR.resolve():
        # The suffix keeps a run over a few named matrices out of the shipped
        # measurements in results/. A directory of one's own has nothing to
        # protect, and gets the canonical names the plotting scripts read.
        output_suffix = ""
    baseline_results_dir = pathlib.Path(args.baseline_results_dir)
    if not baseline_results_dir.is_absolute():
        baseline_results_dir = FILEPATH / baseline_results_dir
    SUITESPARSE_DIR.mkdir(exist_ok=True)

    all_results: dict[str, list[dict[str, Any]]] = {}
    measured: dict[str, list[tuple[str, int]]] = {}
    failed: list[str] = []
    already_done = 0
    dry_run_total = 0
    spreg_threads_noted = False

    for matrix_name in matrices:
        yaml_path = RESULTS_DIR / f"{matrix_name}.yaml"
        bands_yaml_path = BANDS_RESULTS_DIR / f"{matrix_name}.yaml"
        regions = {
            "vbr": parse_yaml_blocks(str(yaml_path)) if yaml_path.exists() else [],
            "vdia": parse_yaml_bands(str(bands_yaml_path)) if bands_yaml_path.exists() else [],
        }
        print(f"\nProcessing {matrix_name}: {len(regions['vbr'])} VBR blocks, {len(regions['vdia'])} VDIA bands")

        # Plan this matrix's runs before touching the matrix itself. A format
        # with no regions here would compile to the CSR-only program recorded
        # under the format's name, so it is left out, as is anything an earlier
        # run already measured.
        run_vbr = bool(regions["vbr"]) and (paper_vbr_set is None or matrix_name in paper_vbr_set)
        planned = []
        for operation in operations:
            csr_kernels = _resolve_csr_kernels(operation, args.csr_kernels, parser)
            if not args.codegen_only:
                # Codegen links nothing, so it needs no native component.
                csr_kernels = _buildable_csr_kernels(operation, csr_kernels)
            if paper_vdia:
                vdia_kernels = list(PAPER_VDIA_KERNELS[operation])
            else:
                vdia_kernels = _resolve_vdia_kernels(operation, vdia_arg, parser)
            kernels_by_kind = {
                "vbr": vbr_kernels if run_vbr else [],
                "vdia": vdia_kernels if regions["vdia"] else [],
            }
            for num_threads in thread_counts:
                for format_kind, format_kernels in kernels_by_kind.items():
                    for format_kernel in format_kernels:
                        for csr_kernel in csr_kernels:
                            if num_threads > 1 and csr_kernel == CSRKernel.SPREG:
                                if not spreg_threads_noted:
                                    print("  [spmm] The spreg CSR kernel is wired for one thread; "
                                          "skipping it for the other thread counts")
                                    spreg_threads_noted = True
                                continue
                            results_key = f"{operation.value}_{format_kernel.value}_{csr_kernel.value}"
                            output_file = output_dir / f"sable_{results_key}{output_suffix}.json"
                            if args.skip_existing and not args.codegen_only and _already_measured(
                                _output_entries(all_results, results_key, output_file), matrix_name, num_threads
                            ):
                                already_done += 1
                                continue
                            planned.append((operation, num_threads, format_kind, format_kernel, csr_kernel,
                                            results_key, output_file))
        if not planned:
            print(f"  Nothing to run for {matrix_name}: no regions for the requested formats, "
                  "or every configuration is already measured")
            continue
        if args.dry_run:
            for _op, num_threads, _kind, _fk, _ck, results_key, _out in planned:
                print(f"  would run {results_key} ({_thread_key(num_threads)})")
            dry_run_total += len(planned)
            continue

        print("  Downloading matrix from SuiteSparse...")
        download_result = download_matrix_from_suitesparse(matrix_name)
        if download_result is None:
            print(f"  Failed to download {matrix_name}, skipping")
            failed.append(f"{matrix_name}: matrix not available")
            continue

        mtx_path, matrix_info, tar_path, matrix_subdir = download_result

        try:
            print(f"  Loading matrix from {mtx_path}...")
            A = csc_matrix(mmread(mtx_path), copy=False)
            matrix_rows, matrix_cols = A.shape
            matrix_nnz = A.nnz
            print(f"  Matrix shape: {matrix_rows} x {matrix_cols}, NNZ: {matrix_nnz}")

            prepared: dict[tuple[Operation, str], tuple[dict[str, Any], dict[str, Any]]] = {}
            # One CSR baseline per (operation, CSR kernel, thread count) for
            # this matrix, shared by every format kernel.
            baseline_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

            for operation, num_threads, format_kind, format_kernel, csr_kernel, results_key, output_file in planned:
                if (operation, format_kind) not in prepared:
                    print(f"\n  === Converting to frontend formats ({operation.value.upper()}, {format_kind}) ===")
                    prepared[(operation, format_kind)] = _convert_and_prepare(
                        operation, matrix_name, regions[format_kind], A, format_kind,
                        write_rhs=not args.codegen_only,
                    )
                composed_data, baseline_data = prepared[(operation, format_kind)]
                bench_iterations = args.bench
                if bench_iterations is None:
                    bench_iterations = DEFAULT_SPMV_BENCH_ITERATIONS if operation == Operation.SPMV else DEFAULT_SPMM_BENCH_ITERATIONS

                if args.codegen_only:
                    print(f"\n  === Generating {results_key} code ===")
                else:
                    print(f"\n  === Running {results_key} benchmark (threads={num_threads}) ===")

                label = f"{matrix_name} {results_key} ({_thread_key(num_threads)})"
                try:
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
                        baseline_cache=baseline_cache,
                    )
                except Exception as exc:
                    # One configuration failing must not cost the rest of the matrix.
                    print(f"  Error in {label}: {exc}")
                    traceback.print_exc()
                    failed.append(label)
                    continue

                if result:
                    _append_result(all_results, results_key, matrix_name, result, num_threads, output_file)
                    measured.setdefault(results_key, []).append((matrix_name, num_threads))
                elif not args.codegen_only:
                    failed.append(label)

            print(f"\nCompleted processing {matrix_name}")
        except Exception as exc:
            print(f"  Error processing {matrix_name}: {exc}")
            traceback.print_exc()
            failed.append(f"{matrix_name}: {exc}")
        finally:
            if not os.environ.get("SABLE_NO_CLEANUP"):
                if tar_path is not None or matrix_subdir is not None:
                    print(f"  Cleaning up downloaded files for {matrix_name}...")
                    cleanup_matrix_files(tar_path, matrix_subdir)

    if args.dry_run:
        print(f"\n{dry_run_total} configuration(s) would run; {already_done} already measured")
        return 0

    print("\n" + "=" * 60)
    print(f"Benchmark Summary ({ops_label})")
    print("=" * 60)
    # Only what this run measured: the output files also hold earlier entries.
    for results_key, runs in measured.items():
        print(f"\n{results_key} Results ({len(runs)} measured in this run):")
        entries = {entry["matrix_name"]: entry for entry in all_results[results_key]}
        for matrix_name, num_threads in runs:
            thread_key = _thread_key(num_threads)
            thread_timing = entries[matrix_name]["timing"][thread_key]
            dispatch_summary = ", ".join(
                f"{name}: {info.get('time_ns', 0):.0f}ns"
                for name, info in thread_timing.get("dispatch_times", {}).items()
            )
            print(
                f"  {matrix_name} ({thread_key}): "
                f"total: {thread_timing['total_time_ns']:.0f}ns, "
                f"CSR baseline: {thread_timing.get('csr_baseline_time_ns', 0):.0f}ns, "
                f"speedup: {thread_timing.get('speedup', 0):.3f}x"
                + (f", {dispatch_summary}" if dispatch_summary else "")
            )
    if already_done:
        print(f"\nSkipped {already_done} configuration(s) already measured (--skip-existing)")
    if failed:
        print(f"\n{len(failed)} configuration(s) produced no result and were not recorded:")
        for label in failed:
            print(f"  {label}")
        print("Re-run with --skip-existing to retry just these.")

    print(f"\nResults written to {output_dir}/")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
