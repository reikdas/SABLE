#!/usr/bin/env python3

import argparse
import itertools
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
import yaml
from scipy.io import mmread
from scipy.sparse import csr_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "find-submatrices"))
from find_matrices import cleanup_matrix_files, get_matrix_info

from sable import Matrix, Operation, Plan
from sable.build_config import CSRKernel, VBRKernel, VDIAKernel
from sable.compiler import build_compile_command_for_plan
from sable.extractors import (
    BandExtractorSkip,
    BlockDetector,
    BlockDetectorSkip,
    CSRConvertor,
    SpMMBandExtractor,
    SpMVBandExtractor,
)
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
from sable.formats import CSR, VBR, VDIA
from sable.kernels.vbr import vbr_blocks
from sable.tensor import DenseInput, DenseLayout
from utils.fileio import check_values, parse_yaml_bands, parse_yaml_blocks, write_check_matrix, write_check_vector


FILEPATH = pathlib.Path(__file__).resolve().parent

COMPILE_TIMEOUT = 60 * 60 * 4
DEFAULT_BENCH_ITERATIONS = {Operation.SPMV: 30, Operation.SPMM: 10}
PHYSICAL_CORES = list(range(os.cpu_count() or 20))
# Timed binaries run on cores FIRST_CORE .. FIRST_CORE + threads - 1 (outside SLURM, which places them).
FIRST_CORE = int(os.environ.get("SABLE_FIRST_CORE", "0"))
SPMM_NRHS = 512

SPMV_CSR_KERNELS = (CSRKernel.NAIVE, CSRKernel.MKL, CSRKernel.SPV8, CSRKernel.UZP)
SPMM_CSR_KERNELS = (CSRKernel.NAIVE, CSRKernel.MKL, CSRKernel.SPREG)

KINDS = ("vbr", "vdia")
FORMAT_LABELS = {VBR: "vbr", VDIA: "vdia", CSR: "csr"}

SUITESPARSE_DIR = pathlib.Path(os.environ.get("SABLE_SUITESPARSE_DIR") or str(FILEPATH / "Suitesparse"))
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"
BASELINE_TIME_KEY = "csr_baseline_time_ns"
# Each operation's band extractor. Keys are the band entries of the densities dict.
BAND_EXTRACTORS = {Operation.SPMV.value: SpMVBandExtractor, Operation.SPMM.value: SpMMBandExtractor}
# An output value passes the check when it is within this fraction of
# sum_j |A_ij| |X_jk| of the SciPy product; reordered sums stay far below it.
CHECK_RTOL = 1e-9
_RESULTS_JSON_CACHE: dict[pathlib.Path, list[dict[str, Any]]] = {}

# Result file names are <op>_<format token>_csr-<csr kernel>.json, where a
# format token is <format_token(kind)>-<kernel>: spmv_vbr050-blockmixed_csr-naive.json,
# spmm_vdia-bandnaive_csr-spreg.json.


def density_token(kind: str, density: float | str) -> str:
    """'vbr050' for blocks at 0.5; 'vdiaspmv' / 'vdiaspmm' for bands, whose
    entry in the densities dict names the operation whose extractor found them.
    Names the stored extractions, which are per operation once bands are in."""
    if isinstance(density, str):
        return f"{kind}{density}"
    return f"{kind}{int(round(density * 100)):03d}"


def format_token(kind: str, density: float | str) -> str:
    """'vbr050' for blocks at 0.5, 'vdia' for bands: names result files, whose
    operation prefix already says which extractor found the bands."""
    return "vdia" if kind == "vdia" else density_token(kind, density)


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


def _has_baseline_times(path: pathlib.Path) -> bool:
    try:
        data = _load_results_json(path)
    except Exception:
        return False
    return any(BASELINE_TIME_KEY in timing for entry in data for timing in entry.get("timing", {}).values())


def _baseline_result_file(
    operation: Operation,
    csr_kernel: CSRKernel,
    baseline_results_dir: pathlib.Path,
    baseline_from: str | None,
) -> pathlib.Path:
    """The results file to take existing CSR baselines from.
    """
    if baseline_from:
        path = baseline_results_dir / f"{operation.value}_{baseline_from}_csr-{csr_kernel.value}.json"
        if not path.exists():
            raise FileNotFoundError(f"--baseline-from {baseline_from}: {path} does not exist")
        return path

    candidates = [
        path
        for path in sorted(baseline_results_dir.glob(f"{operation.value}_*_csr-{csr_kernel.value}.json"))
        if _has_baseline_times(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No results file with {BASELINE_TIME_KEY} for {operation.value}/{csr_kernel.value} "
            f"in {baseline_results_dir}"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Several results files hold {operation.value}/{csr_kernel.value} baselines ({names}); "
            "pick one with --baseline-from <format token, e.g. vbr050-blockmixed>"
        )
    return candidates[0]


def _lookup_existing_baseline(
    operation: Operation,
    csr_kernel: CSRKernel,
    matrix_name: str,
    num_threads: int,
    baseline_results_dir: pathlib.Path,
    baseline_from: str | None,
) -> tuple[float, str]:
    baseline_file = _baseline_result_file(operation, csr_kernel, baseline_results_dir, baseline_from)
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
    max_run_seconds: float | None = None,
    run_timeout: float | None = None,
) -> Tuple[Dict[int, float], Dict[str, float], float, int, Optional[bytes]]:
    """Run the program up to `bench_freq` times and average its timings.

    No new invocation starts once `max_run_seconds` have passed (the first
    always runs), and an invocation longer than `run_timeout` is killed and
    ends the runs. Returns the averages, the compile time, the number of
    completed invocations, and the last invocation's raw output, which holds
    the computed values the output check reads.
    """
    cores_to_use = PHYSICAL_CORES[FIRST_CORE:FIRST_CORE + threads]
    compile_result = compile_frontend_executor(executor)
    if compile_result is None:
        print(f"Failed to compile {executor.filename}, skipping evaluation")
        return {}, {}, 0.0, 0, None

    executable_path, compile_time_ns = compile_result
    dispatch_times: dict[int, list[float]] = {}
    dispatch_part_times: dict[tuple[int, int], list[float]] = {}

    if os.environ.get("SLURM_JOB_ID"):
        run_cmd = [executable_path]
    else:
        run_cmd = ["taskset", "-a", "-c", ",".join(str(x) for x in cores_to_use), executable_path]

    print(f"  Executing benchmark binary: {executable_path}")
    invocations = 0
    last_output = None
    start = time.monotonic()
    for _ in range(bench_freq):
        if invocations and max_run_seconds is not None and time.monotonic() - start >= max_run_seconds:
            print(f"  Run budget of {max_run_seconds:.0f}s reached after {invocations} invocation(s)")
            break
        try:
            raw = subprocess.check_output(
                run_cmd,
                cwd=executor.runtime_cwd or executor.artifact_dir,
                env=_executor_env(executor.compile_command or [], executor.runtime_env),
                preexec_fn=set_ulimit,
                timeout=run_timeout,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Error running {executor.filename}: {exc}")
            continue
        except subprocess.TimeoutExpired:
            print(f"  An invocation of {executor.filename} exceeded {run_timeout:.0f}s; no further invocations")
            break
        invocations += 1
        last_output = raw
        # The timings come before the first blank line, the computed values after it.
        timing_text = raw.partition(b"\n\n")[0].decode("utf-8")
        _parse_timing_output(timing_text.split("\n"), dispatch_times, dispatch_part_times, extract_parts)

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

    return avg_dispatch_times, avg_dispatch_part_times, compile_time_ns, invocations, last_output


def check_output_values(operation: Operation, mat: scipy.sparse.spmatrix, output: Optional[bytes]) -> Optional[dict[str, Any]]:
    """Compare a program's computed values with SciPy's product of the same check input.

    The program prints its output after the first blank line, row-major for
    SpMM. None when there is no output to check.
    """
    if output is None:
        return None
    computed = numpy.fromstring(output[output.find(b"\n\n") + 2:], sep="\n")
    nrows, ncols = mat.shape
    if operation == Operation.SPMV:
        rhs = check_values(ncols)
    else:
        rhs = check_values(ncols * SPMM_NRHS).reshape(ncols, SPMM_NRHS)
    expected = numpy.asarray(mat @ rhs)
    if computed.size != expected.size:
        return {"values_match": False, "error": f"{computed.size} values printed, {expected.size} expected"}
    error = numpy.abs(computed.reshape(expected.shape) - expected)
    del computed, expected
    # The check input is positive, so |A| |X| is |A| X.
    scale = numpy.asarray(abs(mat) @ rhs)
    del rhs
    values_match = bool(numpy.all(error <= CHECK_RTOL * scale))
    max_abs_diff = float(error.max()) if error.size else 0.0
    numpy.divide(error, numpy.maximum(scale, numpy.finfo(float).tiny, out=scale), out=error)
    return {
        "values_match": values_match,
        "max_abs_diff": max_abs_diff,
        "max_rel_err": float(error.max()) if error.size else 0.0,
        "rtol": CHECK_RTOL,
    }


def _report_check(label: str, check: Optional[dict[str, Any]]) -> None:
    if check is None:
        return
    if check["values_match"]:
        print(f"  [{label}] Output check passed (max relative error {check['max_rel_err']:.2e})")
    else:
        print(f"  [{label}] VALUES MISMATCH: {check}")


# ---------------------------------------------------------------------------
# Matrix download and discovery
# ---------------------------------------------------------------------------


def download_matrix_from_suitesparse(matrix_name: str) -> Optional[Tuple[str, Any, Optional[str], Optional[str]]]:
    # A local copy laid out as <dir>/<name>/<name>.mtx (SABLE_LOCAL_SUITESPARSE_DIR) is
    # used first, read only: no matrix_info is returned, so nothing is cleaned up.
    local_dir = os.environ.get("SABLE_LOCAL_SUITESPARSE_DIR")
    if local_dir:
        local_mtx = pathlib.Path(local_dir) / matrix_name / f"{matrix_name}.mtx"
        if local_mtx.exists():
            print(f"  Using local matrix {local_mtx}")
            return str(local_mtx), None, None, None

    SUITESPARSE_DIR.mkdir(exist_ok=True)

    matrix_info = get_matrix_info(matrix_name)
    if matrix_info is None:
        return None

    # Prefer the shared ssgetpy cache (~/.ssgetpy), which find-submatrices/scan_bands.py and
    #    other tools populate. If the matrix is already extracted there, reuse it
    #    and return tar_path/matrix_subdir as None so the caller's cleanup step
    #    leaves the shared copy untouched.
    cache_subdir, _ = matrix_info.localpath(format="MM", extract=True)
    cache_mtx = os.path.join(cache_subdir, f"{matrix_info.name}.mtx")
    if os.path.exists(cache_mtx):
        print(f"  Using cached matrix from {cache_mtx}")
        return cache_mtx, matrix_info, None, None

    # Otherwise download into the project-local Suitesparse/ directory. These
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


def extraction_dir(composition: list[str], densities: dict[str, float]) -> pathlib.Path:
    """Where a composition's extractions live: find-submatrices/results_<tokens in order>.

    A file there holds every format of one matrix, searched in the recorded
    order with each search confined to what the previous one left, so a
    composition extracted bands-first and one extracted blocks-first are
    different directories (results_vdiaspmv_vbr050 and results_vbr050_vdiaspmv).
    """
    tokens = "_".join(density_token(kind, densities[kind]) for kind in composition)
    return FILEPATH / "find-submatrices" / f"results_{tokens}"


def load_extraction(path: pathlib.Path) -> tuple[list[str], dict[str, list[Any]]]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    order = [str(kind) for kind in (data.get("order") or [])]
    return order, {"vdia": parse_yaml_bands(str(path)), "vbr": parse_yaml_blocks(str(path))}


def stored_regions(matrix_name: str, composition: list[str], extraction_path: pathlib.Path) -> dict[str, list[Any]] | None:
    """The stored regions of a composition, or None when it is not stored.

    A composition file wins when present. Blocks alone also fall back to
    find-submatrices/results, where the older block extractions were stored
    (searched on the full matrix).
    """
    if extraction_path.exists():
        order, regions = load_extraction(extraction_path)
        if order != list(composition):
            raise ValueError(f"{extraction_path} was extracted in the order {order}, not {list(composition)}")
        return {kind: regions[kind] for kind in composition}
    if composition == ["vbr"]:
        legacy = RESULTS_DIR / f"{matrix_name}.yaml"
        if legacy.exists():
            return {"vbr": parse_yaml_blocks(str(legacy))}
    return None


def get_available_matrices(composition: list[str], densities: dict[str, float]) -> List[str]:
    """Matrices with stored regions for every format of the composition."""
    directory = extraction_dir(composition, densities)
    names = {path.stem for path in directory.glob("*.yaml")}
    if composition == ["vbr"]:
        names |= {path.stem for path in RESULTS_DIR.glob("*.yaml")}
    available = []
    for name in sorted(names):
        try:
            regions = stored_regions(name, composition, directory / f"{name}.yaml")
        except ValueError:
            continue
        if regions is not None and all(regions[kind] for kind in composition):
            available.append(name)
    return available


def check_batch(matrices: List[str], composition: list[str], densities: dict[str, float],
                parser: argparse.ArgumentParser) -> None:
    """Refuse a batch unless every matrix has stored regions for every format.

    A format with no regions for a matrix would compile to a program without
    that format's dispatch, recorded under the format's name, so a mixed batch
    is an error rather than something to skip: run the matrices that lack a
    format as their own batch without that format's kernels, or extract the
    composition for them with --extract.
    """
    directory = extraction_dir(composition, densities)
    problems: dict[str, list[str]] = {}
    for name in matrices:
        try:
            regions = stored_regions(name, composition, directory / f"{name}.yaml")
        except ValueError as exc:
            problems.setdefault(str(exc), []).append(name)
            continue
        for kind in composition:
            if regions is None or not regions[kind]:
                problems.setdefault(f"no {kind.upper()} regions stored for", []).append(name)
    if problems:
        text = "; ".join(f"{what}: {', '.join(names)}" for what, names in problems.items())
        parser.error(f"{text}. Every matrix in a batch needs regions for each requested format "
                     f"(stored under {directory} for this composition); run the others as a "
                     "separate batch, or search from scratch with --extract.")


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
    if vdia_kernel != VDIAKernel.NAIVE:
        raise ValueError(f"{vdia_kernel.value} is not a frontend SpMV VDIA kernel")
    return NaiveVDIASpmv()


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
    if vdia_kernel != VDIAKernel.NAIVE:
        raise ValueError(f"{vdia_kernel.value} is not a frontend SpMM VDIA kernel")
    return NaiveVDIASpmm()


def _csr_spmm_kernel(csr_kernel: CSRKernel):
    if csr_kernel == CSRKernel.MKL:
        return MKLCSRSpmm()
    if csr_kernel == CSRKernel.SPREG:
        return SPRegCSRSpmm()
    if csr_kernel == CSRKernel.NAIVE:
        return NaiveCSRSpmm()
    raise ValueError(f"{csr_kernel.value} is not a frontend SpMM CSR kernel")


# ---------------------------------------------------------------------------
# Kernels per operation
# ---------------------------------------------------------------------------


def _kernel_for(operation: Operation, kind: str, kernel_enum):
    if kind == "vbr":
        return _vbr_spmv_kernel(kernel_enum) if operation == Operation.SPMV else _vbr_spmm_kernel(kernel_enum)
    if kind == "vdia":
        return _vdia_spmv_kernel(kernel_enum) if operation == Operation.SPMV else _vdia_spmm_kernel(kernel_enum)
    raise ValueError(f"Unknown format kind: {kind}")


def _csr_kernel_for(operation: Operation, csr_kernel: CSRKernel):
    return _csr_spmv_kernel(csr_kernel) if operation == Operation.SPMV else _csr_spmm_kernel(csr_kernel)


# ---------------------------------------------------------------------------
# Extraction from scratch, in a stated order
# ---------------------------------------------------------------------------


def _plain(value):
    """YAML-safe copy: numpy scalars to Python scalars, tuples to lists."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, numpy.floating):
        return float(value)
    return value


def extract_regions(
    matrix_name: str,
    mat: scipy.sparse.spmatrix,
    composition: list[str],
    densities: dict[str, float],
    block_params: dict[str, Any],
    out_path: pathlib.Path,
) -> dict[str, list[Any]]:
    """Search the formats of `composition` in order, each on the residual the
    previous one left, and record what was found in out_path."""
    plan = Plan(Matrix(mat, name=matrix_name), artifact_dir=str(out_path.parent))
    regions: dict[str, list[Any]] = {}
    seconds: dict[str, float] = {}
    selection: dict[str, Any] = {}
    for kind in composition:
        if kind == "vdia":
            extractor = BAND_EXTRACTORS[densities["vdia"]]()
            start = time.perf_counter()
            plan.extract(extractor)
            seconds["vdia"] = time.perf_counter() - start
            regions["vdia"] = extractor.bands
            selection = extractor.selection
        else:
            detector = BlockDetector(min_density=densities["vbr"], **block_params)
            start = time.perf_counter()
            fmt = plan.extract(detector)
            seconds["vbr"] = time.perf_counter() - start
            regions["vbr"] = [tuple(int(v) for v in block) for block in fmt.blocks]
        print(f"  [{kind}] {len(regions[kind])} regions found in {seconds[kind]:.1f}s; "
              f"residual nnz {plan.residual.nnz}")

    record: dict[str, Any] = {"order": list(composition)}
    if "vbr" in composition:
        record.update({"block_density": densities["vbr"], "block_area": block_params["min_area"],
                       "block_gamma": block_params["gamma"], "block_threads": block_params["threads"],
                       "block_timeout_seconds": block_params["timeout_seconds"]})
    if "vdia" in composition:
        record["band_extractor"] = densities["vdia"]
        record["band_selection"] = selection
    record["extraction_seconds"] = seconds
    record["bands"] = regions.get("vdia", [])
    record["blocks"] = [{"rows": [r0, r1], "cols": [c0, c1]} for r0, r1, c0, c1 in regions.get("vbr", [])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(_plain(record), f, sort_keys=False)
    print(f"  Extraction recorded in {out_path}")
    return regions


def regions_for(
    matrix_name: str,
    mat: scipy.sparse.spmatrix,
    composition: list[str],
    densities: dict[str, float],
    block_params: dict[str, Any],
    extract: bool,
    re_extract: bool,
    directory: pathlib.Path,
) -> dict[str, list[Any]]:
    """The regions each format of the composition packs, from the stored
    extraction or, with --extract, from a fresh search that is then stored."""
    path = directory / f"{matrix_name}.yaml"
    if extract and (re_extract or not path.exists()):
        print(f"  Extracting {' then '.join(composition)} from scratch...")
        return extract_regions(matrix_name, mat, composition, densities, block_params, path)
    regions = stored_regions(matrix_name, composition, path)
    if regions is None:
        raise FileNotFoundError(
            f"No stored extraction of {' then '.join(composition)} for {matrix_name} at {path}; "
            "run with --extract to search it")
    return regions


# ---------------------------------------------------------------------------
# Building one program: the formats in order, then the CSR residual
# ---------------------------------------------------------------------------


def _compile_program(
    operation: Operation,
    matrix_name: str,
    mat: scipy.sparse.spmatrix,
    steps: list[tuple[str, Any, list[Any]]],
    artifact_dir: str,
    csr_kernel: CSRKernel,
    bench_iterations: int,
    data_dir: str | None = None,
    data_key: str | None = None,
    num_threads: int = 1,
):
    """Extract the formats of `steps` in order (replaying stored regions
    through the skip extractors, so the second format packs only what the
    first left), dispatch each to its kernel, and give the residual to the
    CSR kernel. With no steps this is the CSR-only baseline program."""
    matrix = Matrix(mat, name=matrix_name)
    plan = Plan(matrix, artifact_dir=artifact_dir)
    if operation == Operation.SPMV:
        plan.rhs(DenseInput.vector(write_check_vector(matrix.ncols), matrix.ncols))
    else:
        plan.rhs(
            DenseInput.matrix(
                write_check_matrix(matrix.ncols, SPMM_NRHS),
                shape=(matrix.ncols, SPMM_NRHS),
                layout=DenseLayout.ROW_MAJOR,
            )
        )
    for kind, kernel_enum, regions in steps:
        extractor = BandExtractorSkip(regions) if kind == "vdia" else BlockDetectorSkip(regions)
        fmt = plan.extract(extractor)
        plan.dispatch(fmt, _kernel_for(operation, kind, kernel_enum), num_threads=num_threads)
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, _csr_kernel_for(operation, csr_kernel), num_threads=num_threads)
    return plan.compile(filename=matrix_name, bench=bench_iterations, data_dir=data_dir, data_key=data_key)


# ---------------------------------------------------------------------------
# Result building and benchmarking
# ---------------------------------------------------------------------------

# The per-variant result record. Every per-variant file under results/ holds a
# list of these, and the older files were migrated through these same
# functions, so the shape is defined here and nowhere else.
#
#   matrix_name, matrix_dimensions
#   dispatch_labels        dispatch number -> format (vbr, vdia, csr); a property of
#                          the program, so it sits beside the timings, not inside them
#   dispatch_part_regions  timed part -> the region it computes: rows, cols, nnz for a
#                          packed VBR block; row_start, nrows, ndiags, nnz for a VDIA
#                          segment; plus `region`, the index of the stored block or
#                          band it came from. None when unknown, {} when nothing has parts.
#   nnz                    what the formats claimed of the matrix and what was left to CSR
#   timing["<n> thread"]   one measurement record per thread count


def _part_sort_key(key: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in re.findall(r"\d+", key))


def _dispatch_labels(plan: Plan) -> dict[str, str]:
    return {
        f"dispatch_{index}": FORMAT_LABELS.get(type(dispatch.fmt), type(dispatch.fmt).__name__.lower())
        for index, dispatch in enumerate(plan.dispatches, start=1)
    }


def _containing_region(regions: list[Any], r0: int, r1: int, c0: int, c1: int) -> int | None:
    for index, (br0, br1, bc0, bc1) in enumerate(regions):
        if br0 <= r0 and r1 <= br1 and bc0 <= c0 and c1 <= bc1:
            return index
    return None


def _nonzeros(values: list[float], start: int, count: int) -> int:
    return sum(1 for value in values[start:start + count] if value != 0.0)


def _dispatch_part_regions(plan: Plan, regions_by_kind: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """One record per timed part, keyed like dispatch_part_times, in emission order.

    A VBR kernel times one part per packed block, and the packer cuts the
    matrix along every block boundary and skips empty cells, so a part is a
    cell inside a stored block rather than the block itself. A VDIA kernel
    times one part per segment, in the order the packer appended them band by
    band. The nonzero counts are taken from the packed values, so for a format
    extracted after another they count only what that format actually holds.
    """
    regions: dict[str, dict[str, Any]] = {}
    for index, dispatch in enumerate(plan.dispatches, start=1):
        fmt = dispatch.fmt
        if isinstance(fmt, VBR):
            for part, (r0, r1, c0, c1, offset) in enumerate(vbr_blocks(fmt), start=1):
                rows, cols = int(r1 - r0), int(c1 - c0)
                regions[f"dispatch_{index}_part_{part}"] = {
                    "rows": rows,
                    "cols": cols,
                    "nnz": _nonzeros(fmt.val.values, int(offset), rows * cols),
                    "region": _containing_region(regions_by_kind.get("vbr", []), r0, r1, c0, c1),
                }
        elif isinstance(fmt, VDIA) and hasattr(dispatch.kernel, "emit_timed_calls"):
            # Only kernels that time each segment as its own part have per-segment parts.
            band_of_segment = [
                band_index
                for band_index, band in enumerate(regions_by_kind.get("vdia", []))
                for _segment in band.get("segments", [])
            ]
            for seg in range(fmt.nsegments):
                nrows = int(fmt.seg_nrows[seg])
                ndiags = int(fmt.seg_ndiags[seg])
                regions[f"dispatch_{index}_part_{seg + 1}"] = {
                    "row_start": int(fmt.seg_row_start[seg]),
                    "nrows": nrows,
                    "ndiags": ndiags,
                    "nnz": _nonzeros(fmt.val.values, int(fmt.seg_val_ptr[seg]), nrows * ndiags),
                    "region": band_of_segment[seg] if seg < len(band_of_segment) else None,
                }
    return regions


def _nnz_record_from_plan(plan: Plan, matrix_nnz: int) -> dict[str, Any]:
    """What the formats hold, counted from the packed values, and what was left to CSR."""
    claimed = 0
    area = 0
    residual = 0
    for dispatch in plan.dispatches:
        fmt = dispatch.fmt
        if isinstance(fmt, VBR):
            for r0, r1, c0, c1, offset in vbr_blocks(fmt):
                cells = (r1 - r0) * (c1 - c0)
                area += cells
                claimed += _nonzeros(fmt.val.values, int(offset), cells)
        elif isinstance(fmt, VDIA):
            for seg in range(fmt.nsegments):
                cells = int(fmt.seg_nrows[seg]) * int(fmt.seg_ndiags[seg])
                area += cells
                claimed += _nonzeros(fmt.val.values, int(fmt.seg_val_ptr[seg]), cells)
        elif isinstance(fmt, CSR):
            residual += int(fmt.nnz)
    return _nnz_record(claimed, residual, area, matrix_nnz)


def _nnz_record(claimed_nnz: int, residual_nnz: int, format_area: int, matrix_nnz: int) -> dict[str, Any]:
    return {
        "format_claimed_nnz": claimed_nnz,
        "residual_nnz": residual_nnz,
        "format_area": format_area,
        "extra_values": format_area - claimed_nnz,
        "format_claimed_nnz_perc": round(claimed_nnz / matrix_nnz * 100, 2) if matrix_nnz > 0 else 0,
        "residual_nnz_perc": round(residual_nnz / matrix_nnz * 100, 2) if matrix_nnz > 0 else 0,
    }


def _entry_record(
    matrix_name: str,
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    dispatch_labels: dict[str, str],
    dispatch_part_regions: dict[str, dict[str, Any]] | None,
    nnz_record: dict[str, Any],
) -> Dict[str, Any]:
    """The per-matrix fields of a result entry; `timing` is added per thread count."""
    cells = matrix_rows * matrix_cols
    return {
        "matrix_name": matrix_name,
        "matrix_dimensions": {
            "rows": matrix_rows,
            "cols": matrix_cols,
            "nnz": matrix_nnz,
            "density": round(matrix_nnz / cells, 3) if cells > 0 else 0,
        },
        "dispatch_labels": dispatch_labels,
        "dispatch_part_regions": (
            dict(sorted(dispatch_part_regions.items(), key=lambda item: _part_sort_key(item[0])))
            if dispatch_part_regions is not None
            else None
        ),
        "nnz": nnz_record,
    }


def _timing_record(
    dispatch_times: dict[int, float],
    dispatch_part_times: dict[str, float],
    baseline_time_ns: float,
    compile_time_composed_s: float,
    compile_time_csr_baseline_s: float,
    codegen_time_composed_ms: int | None,
    codegen_time_csr_baseline_ms: int | None,
    invocations: int,
    csr_baseline_invocations: int | None,
    verification: Optional[dict[str, Any]],
    csr_baseline_verification: Optional[dict[str, Any]],
) -> Dict[str, Any]:
    """The measurement record for one thread count.

    `verification` is the output check of the composed program and
    `csr_baseline_verification` that of the CSR-only program; the latter is
    None when the baseline was taken from an earlier result file.
    """
    total_time = sum(dispatch_times.values())
    return {
        "total_time_ns": round(total_time, 2),
        "dispatch_times": {
            f"dispatch_{dispatch_id}": {
                "time_ns": round(time_ns, 2),
                "percentage_of_total_time": round(time_ns / total_time * 100, 3) if total_time > 0 else 0,
            }
            for dispatch_id, time_ns in sorted(dispatch_times.items())
        },
        "dispatch_part_times": {
            key: round(value, 2)
            for key, value in sorted(dispatch_part_times.items(), key=lambda item: _part_sort_key(item[0]))
        },
        "csr_baseline_time_ns": round(baseline_time_ns, 2),
        "speedup": round(baseline_time_ns / total_time, 3) if total_time > 0 else 0,
        "compile_time_composed_s": compile_time_composed_s,
        "compile_time_csr_baseline_s": compile_time_csr_baseline_s,
        "codegen_time_composed_ms": codegen_time_composed_ms,
        "codegen_time_csr_baseline_ms": codegen_time_csr_baseline_ms,
        "invocations": invocations,
        "csr_baseline_invocations": csr_baseline_invocations,
        "verification": verification,
        "csr_baseline_verification": csr_baseline_verification,
    }


def _build_matrix_result(
    matrix_name: str,
    composed_executor,
    regions_by_kind: dict[str, list[Any]],
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    dispatch_times: dict[int, float],
    dispatch_part_times: dict[str, float],
    compile_time_composed_ns: float,
    baseline: dict[str, Any],
    invocations: int,
    verification: Optional[dict[str, Any]],
) -> Dict[str, Any]:
    plan = composed_executor.plan
    entry = _entry_record(
        matrix_name,
        matrix_rows,
        matrix_cols,
        matrix_nnz,
        _dispatch_labels(plan),
        _dispatch_part_regions(plan, regions_by_kind),
        _nnz_record_from_plan(plan, matrix_nnz),
    )
    entry["timing"] = _timing_record(
        dispatch_times,
        dispatch_part_times,
        baseline["time_ns"],
        compile_time_composed_ns / 1e9 if compile_time_composed_ns else 0.0,
        baseline["compile_time_ns"] / 1e9 if baseline["compile_time_ns"] else 0.0,
        composed_executor.codegen_time_ms,
        baseline["codegen_time_ms"],
        invocations,
        baseline.get("invocations"),
        verification,
        baseline.get("verification"),
    )
    return entry


def _csr_baseline(
    operation: Operation,
    matrix_name: str,
    mat: scipy.sparse.spmatrix,
    codegen_dir_baseline: str,
    csr_kernel: CSRKernel,
    bench_iterations: int,
    threads: int,
    baseline_source_mode: str,
    baseline_results_dir: pathlib.Path,
    allow_baseline_run_on_missing: bool,
    baseline_from: str | None,
    staged_data_dir: str,
    label: str,
    max_run_seconds: float | None = None,
    run_timeout: float | None = None,
) -> Optional[Dict[str, Any]]:
    """The CSR-only program's time for this matrix, looked up or measured.

    None when it was neither found nor measured, so no result is recorded
    against a missing baseline.
    """
    if baseline_source_mode == "existing":
        try:
            time_ns, source = _lookup_existing_baseline(
                operation, csr_kernel, matrix_name, threads, baseline_results_dir, baseline_from
            )
            print(f"  [{label}] Using existing CSR baseline: {source}")
            return {"time_ns": time_ns, "compile_time_ns": 0.0,
                    "codegen_time_ms": None, "invocations": None, "verification": None}
        except Exception as exc:
            if not allow_baseline_run_on_missing:
                print(f"  [{label}] Missing existing CSR baseline: {exc}")
                return None
            print(f"  [{label}] Existing baseline missing ({exc}); running CSR baseline instead")

    print(f"  [{label}] Generating frontend C code (CSR baseline)...")
    os.makedirs(codegen_dir_baseline, exist_ok=True)
    baseline_executor = _compile_program(
        operation,
        matrix_name,
        mat,
        [],
        codegen_dir_baseline,
        csr_kernel,
        bench_iterations,
        data_dir=staged_data_dir,
        data_key=f"{matrix_name}_csr_baseline",
        num_threads=threads,
    )
    print(f"  [{label}] Evaluating CSR baseline...")
    baseline_dispatch_times, _, compile_time_ns, invocations, output = eval_frontend_executor_timings(
        baseline_executor, bench_iterations, threads=threads,
        max_run_seconds=max_run_seconds, run_timeout=run_timeout,
    )
    if not baseline_dispatch_times:
        print(f"  [{label}] CSR baseline produced no timings")
        return None
    verification = check_output_values(operation, mat, output)
    _report_check(f"{label} CSR baseline", verification)
    return {
        "time_ns": sum(baseline_dispatch_times.values()),
        "compile_time_ns": compile_time_ns,
        "codegen_time_ms": baseline_executor.codegen_time_ms,
        "invocations": invocations,
        "verification": verification,
    }


def _benchmark_variant(
    operation: Operation,
    matrix_name: str,
    mat: scipy.sparse.spmatrix,
    steps: list[tuple[str, Any, list[Any]]],
    results_key: str,
    matrix_rows: int,
    matrix_cols: int,
    matrix_nnz: int,
    bench_iterations: int,
    csr_kernel: CSRKernel,
    baseline_results_dir: pathlib.Path,
    threads: int = 1,
    baseline_source_mode: str = "run",
    allow_baseline_run_on_missing: bool = False,
    baseline_from: str | None = None,
    baseline_cache: dict | None = None,
    max_run_seconds: float | None = None,
    run_timeout: float | None = None,
    extraction_key: str = "",
) -> Optional[Dict[str, Any]]:
    """Time one composed program against the CSR baseline and build its record.

    `extraction_key` names the extraction the steps pack (e.g. vdiaspmv,
    vbr050_vdiaspmm); programs of one matrix share staged data only within it.

    None when either program produced no timings, so a failed compile or run
    is skipped rather than recorded as zero.
    """
    csr_label = csr_kernel.value
    codegen_root = pathlib.Path(os.environ.get("SABLE_CODEGEN_DIR") or str(FILEPATH))
    codegen_dir_composed = str(codegen_root / f"Generated_C_{results_key}")
    # The CSR baseline depends on the CSR kernel only, so it lives beside the
    # variants rather than under one of them and is measured once per matrix.
    codegen_dir_baseline = str(codegen_root / f"Generated_C_{operation.value}_csr-{csr_label}_baseline")
    # Every kernel variant of one extraction order reads the same staged data.
    staged_data_dir = str(codegen_root / "Generated_Staged_Data")
    os.makedirs(codegen_dir_composed, exist_ok=True)

    print(f"  [{results_key}] Generating frontend C code (composed)...")
    composed_executor = _compile_program(
        operation,
        matrix_name,
        mat,
        steps,
        codegen_dir_composed,
        csr_kernel,
        bench_iterations,
        data_dir=staged_data_dir,
        data_key=f"{matrix_name}_{extraction_key or '_'.join(kind for kind, _kernel, _regions in steps)}",
        num_threads=threads,
    )
    print(f"  [{results_key}] Evaluating composed version...")
    dispatch_times, dispatch_part_times, compile_time_composed_ns, invocations, output = eval_frontend_executor_timings(
        composed_executor, bench_iterations, threads=threads,
        max_run_seconds=max_run_seconds, run_timeout=run_timeout,
    )
    if not dispatch_times:
        print(f"  [{results_key}] Composed program produced no timings; not recorded")
        return None
    verification = check_output_values(operation, mat, output)
    _report_check(results_key, verification)

    baseline_key = (operation.value, csr_label, threads)
    baseline = baseline_cache.get(baseline_key) if baseline_cache is not None else None
    if baseline is not None:
        print(f"  [{results_key}] Reusing this matrix's {csr_label} CSR baseline")
    else:
        baseline = _csr_baseline(
            operation, matrix_name, mat, codegen_dir_baseline, csr_kernel, bench_iterations, threads,
            baseline_source_mode, baseline_results_dir, allow_baseline_run_on_missing, baseline_from,
            staged_data_dir, results_key, max_run_seconds=max_run_seconds, run_timeout=run_timeout,
        )
        if baseline is None:
            return None
        if baseline_cache is not None:
            baseline_cache[baseline_key] = baseline

    return _build_matrix_result(
        matrix_name,
        composed_executor,
        {kind: regions for kind, _kernel, regions in steps},
        matrix_rows,
        matrix_cols,
        matrix_nnz,
        dispatch_times,
        dispatch_part_times,
        compile_time_composed_ns,
        baseline,
        invocations,
        verification,
    )


def _load_output_entries(output_file: pathlib.Path) -> list[dict[str, Any]]:
    if not output_file.exists():
        return []
    with open(output_file) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _append_result(
    all_results: dict[str, list[dict[str, Any]]],
    results_key: str,
    matrix_name: str,
    result: dict[str, Any],
    num_threads: int,
    output_file: pathlib.Path,
) -> dict[str, Any]:
    """Merge one matrix's result into the output file and return its entry.

    The file's existing entries are the starting point, so a rerun over some
    of the matrices updates those and leaves the others in place.
    """
    if results_key not in all_results:
        all_results[results_key] = _load_output_entries(output_file)
    results_list = all_results[results_key]

    entry_fields = {key: value for key, value in result.items() if key != "timing"}
    thread_key = _thread_key(num_threads)
    existing_idx = next((i for i, r in enumerate(results_list) if r["matrix_name"] == matrix_name), None)
    if existing_idx is not None:
        timing = dict(results_list[existing_idx].get("timing", {}))
        print(f"  [{results_key}] Updating result for {matrix_name}")
    else:
        timing = {}
        print(f"  [{results_key}] Added new result for {matrix_name}")
    timing[thread_key] = dict(result["timing"])
    matrix_entry = {**entry_fields, "timing": timing}
    if existing_idx is not None:
        results_list[existing_idx] = matrix_entry
    else:
        results_list.append(matrix_entry)

    with open(output_file, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"  [{results_key}] Results written to {output_file}")
    return matrix_entry


def _record_verification(
    output_file: pathlib.Path,
    results_key: str,
    matrix_name: str,
    num_threads: int,
    verification: Optional[dict[str, Any]],
    csr_baseline_verification: Optional[dict[str, Any]],
) -> None:
    """Store an output check in an existing result entry without touching its timings."""
    entries = _load_output_entries(output_file)
    thread_key = _thread_key(num_threads)
    entry = next((e for e in entries if e["matrix_name"] == matrix_name), None)
    thread_timing = entry.get("timing", {}).get(thread_key) if entry else None
    if thread_timing is None:
        print(f"  [{results_key}] No '{thread_key}' result for {matrix_name} in {output_file}; check not stored")
        return
    thread_timing["verification"] = verification
    if csr_baseline_verification is not None:
        thread_timing["csr_baseline_verification"] = csr_baseline_verification
    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  [{results_key}] Output check stored for {matrix_name} in {output_file}")


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
    valid = {kernel.value for kernel in VBRKernel}
    invalid = set(names) - valid
    if invalid:
        parser.error(f"Invalid VBR kernel(s): {invalid}. Valid options: {valid}")
    return [VBRKernel(name) for name in names]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SABLE sparse matrix operations (SpMV / SpMM)",
        epilog=(
            "Examples:\n"
            "  %(prog)s --operation spmv --vbr-kernels blockmixed --no-vdia --csr-kernels naive heart1\n"
            "  %(prog)s --operation spmv --vbr-kernels none --csr-kernels mkl,spv8 ohne2   # SpMV band extractor + NaiveVDIASpmv\n"
            "  %(prog)s --operation spmv,spmm --order vdia,vbr --extract heart1 nemeth22   # bands, then blocks in the residual\n"
            "  %(prog)s --operation spmv --order vbr,vdia --extract heart1                  # blocks first\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--operation", type=str, default="spmv,spmm",
                        help="Comma-separated operations: spmv, spmm, or spmv,spmm (default: spmv,spmm)")
    parser.add_argument("matrices", nargs="*", help="Matrix names to benchmark.")
    parser.add_argument("--matrices", dest="matrices_flag", nargs="*", metavar="MATRIX")
    parser.add_argument("--spmv-bench", type=int, default=None,
                        help=f"SpMV benchmark iterations (default: {DEFAULT_BENCH_ITERATIONS[Operation.SPMV]})")
    parser.add_argument("--spmm-bench", type=int, default=None,
                        help=f"SpMM benchmark iterations (default: {DEFAULT_BENCH_ITERATIONS[Operation.SPMM]})")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--csr-kernels", type=str, default="all",
                        help="SpMV: naive,spv8,mkl,uzp. SpMM: naive,mkl,spreg. Names not available for an operation are skipped for it with a note.")
    parser.add_argument("--vbr-kernels", type=str, default="all",
                        help="blocknaive, blockmixed, blockmkl, all, none, or comma-separated")
    parser.add_argument("--no-vdia", action="store_true",
                        help="Leave VDIA out. Otherwise each operation packs the bands select_bands finds "
                             "with its kernel's slot cost into its VDIA kernel -- SpMV 0.90 at every width "
                             "(>= 35%% of the nonzeros) into NaiveVDIASpmv, SpMM the measured spmm_slot_cost "
                             "(>= 60%% of the nonzeros) into NaiveVDIASpmm; stored under results_vdiaspmv / "
                             "results_vdiaspmm, results named <op>_vdia-bandnaive_csr-<csr>. A matrix "
                             "an extractor declines runs CSR only and is skipped for that operation.")
    parser.add_argument("--order", type=str, default=None, metavar="KIND,KIND",
                        help="Extraction order when both VBR and VDIA kernels are requested: "
                             "vdia,vbr (bands first, blocks in what is left) or vbr,vdia. The "
                             "formats are composed into one program in this order.")
    parser.add_argument("--extract", action="store_true",
                        help="Search the formats from scratch in --order (BlockDetector and the operation's "
                             "band extractor, each on the residual of the previous one) instead of replaying "
                             "stored regions, and record what was found under --extraction-dir. A "
                             "matrix already recorded there is reused unless --re-extract is given. "
                             "Requires an explicit matrix list.")
    parser.add_argument("--re-extract", action="store_true",
                        help="With --extract, search again even when a recording exists")
    parser.add_argument("--extraction-dir", type=str, default=None,
                        help="Parent of the directories a composition's extractions are stored in, "
                             "results_<format tokens in order>, one per operation when bands are extracted "
                             "(default: find-submatrices, e.g. find-submatrices/results_vdiaspmv_vbr050)")
    parser.add_argument("--threads", type=str, default="1",
                        help="Comma-separated thread counts; each count is a separate run pinned to "
                             "that many cores, recorded under its own '<n> thread' key (default: 1)")
    parser.add_argument("--baseline-source", choices=("existing", "run"), default="run",
                        help="Take CSR baselines from --baseline-results-dir, or measure them in this run (default: run)")
    parser.add_argument("--baseline-results-dir", type=str, default="results",
                        help="Directory holding <op>_<format token>_csr-<csr>.json files whose "
                             "entries carry csr_baseline_time_ns")
    parser.add_argument("--baseline-from", type=str, default=None, metavar="FORMAT_TOKEN",
                        help="With --baseline-source existing, take baselines from "
                             "<op>_<FORMAT_TOKEN>_csr-<csr>.json (e.g. vbr050-blockmixed) "
                             "when several files hold them")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Write to <op>_<format token>_csr-<csr>_<suffix>.json instead of the "
                             "canonical file (by default results merge into the canonical one)")
    parser.add_argument("--allow-baseline-run-on-missing", action="store_true",
                        help="If --baseline-source existing is missing an entry, run the CSR baseline instead")
    # The block YAMLs under find-submatrices/results were produced by partition_matrix
    # with --min-density 0.5 --min-area 2500. The YAMLs do not record the threshold,
    # so it is stated here; pass --block-density for a directory produced
    # differently.
    parser.add_argument("--block-density", type=float, default=0.5,
                        help="Minimum block density: what the stored block YAMLs were extracted with, "
                             "and what --extract searches with (default: %(default)s)")
    parser.add_argument("--block-area", type=int, default=2500, help="--extract: minimum block area (default: 2500)")
    parser.add_argument("--block-gamma", type=float, default=1.5, help="--extract: partitioner gamma (default: 1.5)")
    parser.add_argument("--block-timeout", type=float, default=4.0 * 60.0 * 60.0,
                        help="--extract: partitioner time budget in seconds per matrix (default: 4 hours)")
    parser.add_argument("--block-threads", type=int, default=20, help="--extract: partitioner threads (default: 20)")
    parser.add_argument("--max-run-seconds", type=float, default=None,
                        help="Start no new invocation of a program once this many seconds of its runs have "
                             "passed; at least one always runs. The record notes how many completed.")
    parser.add_argument("--run-timeout", type=float, default=None,
                        help="Kill an invocation that runs longer than this many seconds and stop that program")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check the output of each program against SciPy and store the check in the "
                             "existing result entry, leaving its timings untouched (one invocation, one "
                             "iteration per program)")
    args = parser.parse_args()

    densities = {"vbr": args.block_density}
    block_params = {"min_area": args.block_area, "gamma": args.block_gamma,
                    "timeout_seconds": args.block_timeout, "threads": args.block_threads}

    operations = [Operation(op.strip()) for op in args.operation.split(",")]
    bench_flags = {Operation.SPMV: args.spmv_bench, Operation.SPMM: args.spmm_bench}
    for operation, given in bench_flags.items():
        if given is not None and operation not in operations:
            parser.error(f"--{operation.value}-bench given but {operation.value} is not in --operation")
    bench_iterations = {
        operation: 1 if args.verify_only else given if given is not None else DEFAULT_BENCH_ITERATIONS[operation]
        for operation, given in bench_flags.items()
    }
    thread_counts = [int(t.strip()) for t in args.threads.split(",")]
    if any(thread_count < 1 for thread_count in thread_counts):
        parser.error("--threads counts must be at least 1")
    if FIRST_CORE < 0 or FIRST_CORE + max(thread_counts) > len(PHYSICAL_CORES):
        parser.error(f"SABLE_FIRST_CORE={FIRST_CORE} with {max(thread_counts)} threads needs cores beyond "
                     f"the {len(PHYSICAL_CORES)} available")

    vbr_kernels = _resolve_vbr_kernels(args.vbr_kernels, parser)
    vdia_kernels = [] if args.no_vdia else [VDIAKernel.NAIVE]
    want_blocks = bool(vbr_kernels)
    want_bands = bool(vdia_kernels)
    if not want_blocks and not want_bands:
        parser.error("Nothing to compose: --vbr-kernels none with --no-vdia")

    # The composition: which formats one program holds, in extraction order.
    if want_blocks and want_bands:
        if not args.order:
            parser.error("Both VBR and VDIA kernels are requested; state the extraction order with "
                         "--order vdia,vbr or --order vbr,vdia")
        composition = [kind.strip() for kind in args.order.split(",")]
        if sorted(composition) != sorted(KINDS):
            parser.error(f"--order must list each of {', '.join(KINDS)} exactly once, got {args.order}")
    else:
        composition = ["vbr"] if want_blocks else ["vdia"]
    densities_by_op = {op: {**densities, "vdia": op.value} for op in operations}
    extraction_paths = {}
    for op, dens in densities_by_op.items():
        path = extraction_dir(composition, dens)
        if args.extraction_dir:
            root = pathlib.Path(args.extraction_dir)
            path = (root if root.is_absolute() else FILEPATH / root) / path.name
        extraction_paths[op] = path

    matrices = args.matrices or args.matrices_flag
    if args.extract:
        if not matrices:
            parser.error("--extract searches each matrix from scratch; pass the matrices explicitly")
    elif not matrices:
        matrices = sorted({m for dens in densities_by_op.values() for m in get_available_matrices(composition, dens)})
    elif "vdia" not in composition:
        # With bands, an operation's extractor may decline a matrix, which then
        # runs CSR only and is skipped for that operation rather than refused.
        check_batch(list(matrices), composition, densities, parser)

    # Results go to the canonical <op>_<format token>_csr-<csr>.json whatever
    # the run covers; entries merge into what the file already holds.
    output_suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    ops_label = "+".join(op.value.upper() for op in operations)
    print(f"[{ops_label}] Will process {len(matrices)} matrices; formats {' then '.join(composition)}")
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = FILEPATH / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_results_dir = pathlib.Path(args.baseline_results_dir)
    if not baseline_results_dir.is_absolute():
        baseline_results_dir = FILEPATH / baseline_results_dir
    SUITESPARSE_DIR.mkdir(exist_ok=True)

    all_results: dict[str, list[dict[str, Any]]] = {}
    run_summary: dict[str, list[dict[str, Any]]] = {}
    empty_extractions: list[str] = []

    for matrix_name in matrices:
        print(f"\nProcessing {matrix_name}...")
        # One CSR baseline per (operation, CSR kernel, thread count) for this
        # matrix, shared by every format kernel.
        baseline_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        print("  Downloading matrix from SuiteSparse...")
        download_result = download_matrix_from_suitesparse(matrix_name)
        if download_result is None:
            print(f"  Failed to download {matrix_name}, skipping")
            continue

        mtx_path, matrix_info, tar_path, matrix_subdir = download_result

        try:
            print(f"  Loading matrix from {mtx_path}...")
            A = csr_matrix(mmread(mtx_path))
            if numpy.iscomplexobj(A):
                # The kernels are real-valued; casting would drop the imaginary parts.
                print(f"  {matrix_name} is complex; not benchmarked")
                continue
            # Explicit zeros would be computed by the CSR-only baseline but dropped from
            # every residual (ResidualMatrix.without), inflating speedups; drop them here.
            A.eliminate_zeros()
            matrix_rows, matrix_cols = A.shape
            matrix_nnz = A.nnz
            print(f"  Matrix shape: {matrix_rows} x {matrix_cols}, NNZ: {matrix_nnz}")

            regions_cache: dict[pathlib.Path, dict[str, list[Any]]] = {}
            for operation in operations:
                dens = densities_by_op[operation]
                path = extraction_paths[operation]
                try:
                    if path not in regions_cache:
                        regions_cache[path] = regions_for(matrix_name, A, composition, dens, block_params,
                                                          args.extract, args.re_extract, path)
                except FileNotFoundError as exc:
                    print(f"  [{operation.value}] {exc}; skipped")
                    continue
                regions = regions_cache[path]
                for kind in composition:
                    print(f"  [{operation.value}] {len(regions[kind])} {kind.upper()} regions")
                empty = [kind for kind in composition if not regions[kind]]
                if empty:
                    if empty == ["vdia"]:
                        # The extractor declined the matrix for this operation: SABLE runs it as CSR only.
                        print(f"  [{operation.value}] the band extractor selected no bands for "
                              f"{matrix_name}; not benchmarked for {operation.value}")
                        continue
                    # A format that found nothing would leave a program without its
                    # dispatch, recorded under the format's name; refuse instead.
                    print(f"  No {', '.join(kind.upper() for kind in empty)} regions for {matrix_name}; not benchmarked")
                    if matrix_name not in empty_extractions:
                        empty_extractions.append(matrix_name)
                    continue
                csr_kernels = _resolve_csr_kernels(operation, args.csr_kernels, parser)
                kernels_by_kind = {"vbr": vbr_kernels, "vdia": vdia_kernels}
                for num_threads in thread_counts:
                    for kernels in itertools.product(*(kernels_by_kind[kind] for kind in composition)):
                        steps = [(kind, kernel, regions[kind]) for kind, kernel in zip(composition, kernels)]
                        format_tokens = "_".join(
                            f"{format_token(kind, dens[kind])}-{kernel.value}" for kind, kernel in zip(composition, kernels)
                        )
                        for csr_kernel in csr_kernels:
                            results_key = f"{operation.value}_{format_tokens}_csr-{csr_kernel.value}"
                            print(f"\n  === Running {results_key} benchmark (threads={num_threads}) ===")

                            result = _benchmark_variant(
                                operation,
                                matrix_name,
                                A,
                                steps,
                                results_key,
                                matrix_rows,
                                matrix_cols,
                                matrix_nnz,
                                bench_iterations[operation],
                                csr_kernel=csr_kernel,
                                threads=num_threads,
                                baseline_source_mode=args.baseline_source,
                                baseline_results_dir=baseline_results_dir,
                                allow_baseline_run_on_missing=args.allow_baseline_run_on_missing,
                                baseline_from=args.baseline_from,
                                baseline_cache=baseline_cache,
                                max_run_seconds=args.max_run_seconds,
                                run_timeout=args.run_timeout,
                                extraction_key=path.name.removeprefix("results_"),
                            )

                            if result and args.verify_only:
                                thread_timing = result["timing"]
                                _record_verification(
                                    output_dir / f"{results_key}{output_suffix}.json", results_key, matrix_name,
                                    num_threads, thread_timing["verification"],
                                    thread_timing["csr_baseline_verification"],
                                )
                            elif result:
                                output_file = output_dir / f"{results_key}{output_suffix}.json"
                                entry = _append_result(all_results, results_key, matrix_name, result, num_threads, output_file)
                                # The merged entry carries every thread count, so list it once.
                                summary = run_summary.setdefault(results_key, [])
                                summary[:] = [e for e in summary if e["matrix_name"] != matrix_name] + [entry]

            print(f"\nCompleted processing {matrix_name}")
        except Exception as exc:
            print(f"  Error processing {matrix_name}: {exc}")
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
    for results_key, results_list in run_summary.items():
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
    if empty_extractions:
        print(f"Not benchmarked, a requested format found no regions: {', '.join(empty_extractions)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
