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
from find_vdia import find_vdia_regions

from sable import Matrix, Operation, Plan
from sable.build_config import CSRKernel, VBRKernel, VDIAKernel
from sable.compiler import build_compile_command_for_plan
from sable.extractors import (
    BandExtractorSkip,
    BlockDetector,
    BlockDetectorSkip,
    CSRConvertor,
)
from sable.extractors.band_extractor import pack_bands_as_vdia
from sable.kernels import (
    MKLCSRSpmm,
    MKLCSRSpmv,
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
from sable.formats import CSR, VBR, VDIA
from sable.kernels.vbr import vbr_blocks
from sable.tensor import DenseInput, DenseLayout
from utils.fileio import parse_yaml_bands, parse_yaml_blocks, write_dense_matrix, write_dense_vector


FILEPATH = pathlib.Path(__file__).resolve().parent

COMPILE_TIMEOUT = 60 * 60 * 4
DEFAULT_BENCH_ITERATIONS = {Operation.SPMV: 30, Operation.SPMM: 10}
PHYSICAL_CORES = list(range(os.cpu_count() or 20))
SPMM_NRHS = 512

SPMV_CSR_KERNELS = (CSRKernel.NAIVE, CSRKernel.MKL, CSRKernel.SPV8, CSRKernel.UZP)
SPMM_CSR_KERNELS = (CSRKernel.NAIVE, CSRKernel.MKL, CSRKernel.SPREG)

SPMV_VDIA_KERNELS = (VDIAKernel.NAIVE, VDIAKernel.MKL_DIA)
SPMM_VDIA_KERNELS = (VDIAKernel.NAIVE,)
KINDS = ("vbr", "vdia")
FORMAT_LABELS = {VBR: "vbr", VDIA: "vdia", CSR: "csr"}

SUITESPARSE_DIR = pathlib.Path(os.environ.get("SABLE_SUITESPARSE_DIR") or str(FILEPATH / "Suitesparse"))
RESULTS_DIR = FILEPATH / "find-submatrices" / "results"
BANDS_RESULTS_DIR = FILEPATH / "find-submatrices" / "results_bands_075"
BASELINE_TIME_KEY = "csr_baseline_time_ns"
_RESULTS_JSON_CACHE: dict[pathlib.Path, list[dict[str, Any]]] = {}

# Result file names are <op>_<format token>_csr-<csr kernel>.json, where a
# format token is <kind><density>-<kernel>: spmv_vbr050-blockmixed_csr-naive.json,
# spmm_vdia075-bandnaive_csr-spreg.json.


def band_density_from_dir(path: pathlib.Path) -> Optional[float]:
    """0.75 for results_bands_075; None when the name carries no density."""
    match = re.search(r"_(\d{3})$", path.name)
    return int(match.group(1)) / 100 if match else None


def density_token(kind: str, density: float) -> str:
    """'vbr050' for blocks at 0.5, 'vdia075' for bands at 0.75."""
    return f"{kind}{int(round(density * 100)):03d}"


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

    # Prefer the shared ssgetpy cache (~/.ssgetpy), which find_vdia.py and
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


def _legacy_yaml(kind: str, matrix_name: str) -> pathlib.Path:
    return (RESULTS_DIR if kind == "vbr" else BANDS_RESULTS_DIR) / f"{matrix_name}.yaml"


def _parse_regions(kind: str, path: pathlib.Path) -> list[Any]:
    return parse_yaml_blocks(str(path)) if kind == "vbr" else parse_yaml_bands(str(path))


def extraction_dir(composition: list[str], densities: dict[str, float]) -> pathlib.Path:
    """Where a composition's extractions live: find-submatrices/results_<tokens in order>.

    A file there holds every format of one matrix, searched in the recorded
    order with each search confined to what the previous one left, so a
    composition extracted bands-first and one extracted blocks-first are
    different directories (results_vdia075_vbr050 and results_vbr050_vdia075).
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

    A composition file wins when present. A single format also falls back to
    the per-kind directories the older extractions were stored in, which were
    searched on the full matrix.
    """
    if extraction_path.exists():
        order, regions = load_extraction(extraction_path)
        if order != list(composition):
            raise ValueError(f"{extraction_path} was extracted in the order {order}, not {list(composition)}")
        return {kind: regions[kind] for kind in composition}
    if len(composition) == 1:
        kind = composition[0]
        legacy = _legacy_yaml(kind, matrix_name)
        if legacy.exists():
            return {kind: _parse_regions(kind, legacy)}
    return None


def get_available_matrices(composition: list[str], densities: dict[str, float]) -> List[str]:
    """Matrices with stored regions for every format of the composition."""
    directory = extraction_dir(composition, densities)
    names = {path.stem for path in directory.glob("*.yaml")}
    if len(composition) == 1:
        names |= {path.stem for path in _legacy_yaml(composition[0], "*").parent.glob("*.yaml")}
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


class _BandSearch:
    """The band finder run on the residual, keeping every band it accepts.

    BandExtractor.extract records only a summary of each band, and replaying
    the search through BandExtractorSkip needs the segments too, so this runs
    the same finder with the same settings and keeps the complete records.
    """

    produces = VDIA

    def __init__(self, min_density: float):
        self.min_density = min_density
        self.bands: list[dict[str, Any]] = []
        self.seconds = 0.0

    def extract(self, A):
        start = time.perf_counter()
        regions = find_vdia_regions(A.to_csr(), min_density=self.min_density)
        self.bands = [region.to_dict() for region in regions]
        fmt = pack_bands_as_vdia(A, self.bands)
        self.seconds = time.perf_counter() - start
        return fmt, A.without(fmt)


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
    for kind in composition:
        if kind == "vdia":
            searcher = _BandSearch(densities["vdia"])
            plan.extract(searcher)
            regions["vdia"] = searcher.bands
            seconds["vdia"] = searcher.seconds
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
        record["band_density"] = densities["vdia"]
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
        plan.rhs(DenseInput.vector(write_dense_vector(1.0, matrix.ncols), matrix.ncols))
    else:
        plan.rhs(
            DenseInput.matrix(
                write_dense_matrix(1.0, matrix.ncols, SPMM_NRHS),
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
        elif isinstance(fmt, VDIA):
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
    baseline_source: str,
    compile_time_composed_s: float,
    compile_time_csr_baseline_s: float,
    codegen_time_composed_ms: int | None,
    codegen_time_csr_baseline_ms: int | None,
    staged_data_time_composed_ms: int | None,
    staged_data_time_csr_baseline_ms: int | None,
) -> Dict[str, Any]:
    """The measurement record for one thread count."""
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
        "csr_baseline_source": baseline_source,
        "speedup": round(baseline_time_ns / total_time, 3) if total_time > 0 else 0,
        "compile_time_composed_s": compile_time_composed_s,
        "compile_time_csr_baseline_s": compile_time_csr_baseline_s,
        "codegen_time_composed_ms": codegen_time_composed_ms,
        "codegen_time_csr_baseline_ms": codegen_time_csr_baseline_ms,
        "staged_data_time_composed_ms": staged_data_time_composed_ms,
        "staged_data_time_csr_baseline_ms": staged_data_time_csr_baseline_ms,
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
        baseline["source"],
        compile_time_composed_ns / 1e9 if compile_time_composed_ns else 0.0,
        baseline["compile_time_ns"] / 1e9 if baseline["compile_time_ns"] else 0.0,
        composed_executor.codegen_time_ms,
        baseline["codegen_time_ms"],
        composed_executor.staged_data_time_ms,
        baseline["staged_data_time_ms"],
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
) -> Optional[Dict[str, Any]]:
    """The CSR-only program's time for this matrix, looked up or measured."""
    if baseline_source_mode == "existing":
        try:
            time_ns, source = _lookup_existing_baseline(
                operation, csr_kernel, matrix_name, threads, baseline_results_dir, baseline_from
            )
            print(f"  [{label}] Using existing CSR baseline: {source}")
            return {"time_ns": time_ns, "source": source, "compile_time_ns": 0.0,
                    "codegen_time_ms": None, "staged_data_time_ms": None}
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
    baseline_dispatch_times, _, compile_time_ns = eval_frontend_executor_timings(
        baseline_executor, bench_iterations, threads=threads
    )
    return {
        "time_ns": sum(baseline_dispatch_times.values()),
        "source": "measured_in_this_run",
        "compile_time_ns": compile_time_ns,
        "codegen_time_ms": baseline_executor.codegen_time_ms,
        "staged_data_time_ms": baseline_executor.staged_data_time_ms,
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
) -> Optional[Dict[str, Any]]:
    """Time one composed program against the CSR baseline and build its record."""
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
        data_key="_".join([matrix_name] + [kind for kind, _kernel, _regions in steps]),
        num_threads=threads,
    )
    print(f"  [{results_key}] Evaluating composed version...")
    dispatch_times, dispatch_part_times, compile_time_composed_ns = eval_frontend_executor_timings(
        composed_executor, bench_iterations, threads=threads
    )

    baseline_key = (operation.value, csr_label, threads)
    baseline = baseline_cache.get(baseline_key) if baseline_cache is not None else None
    if baseline is not None:
        print(f"  [{results_key}] Reusing this matrix's {csr_label} CSR baseline")
    else:
        baseline = _csr_baseline(
            operation, matrix_name, mat, codegen_dir_baseline, csr_kernel, bench_iterations, threads,
            baseline_source_mode, baseline_results_dir, allow_baseline_run_on_missing, baseline_from,
            staged_data_dir, results_key,
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


def _resolve_vdia_kernels(operation: Operation, arg: str,
                          parser: argparse.ArgumentParser) -> list[VDIAKernel]:
    """The VDIA kernels to run for one operation.

    Unlike VBR, which kernels exist depends on the operation, so this takes
    the operation the way _resolve_csr_kernels does. "all" is every kernel the
    operation offers. A name the operation does not offer is an error, so
    "--vdia-kernels bandmkl --operation spmv,spmm" is refused rather than run
    for the SpMV half only. "none" selects nothing: VDIA is optional.
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
    unavailable = [name for name in names if name not in {kernel.value for kernel in available}]
    if unavailable:
        parser.error(f"VDIA kernel(s) not available for {operation.value.upper()}: {unavailable}. "
                     f"{operation.value.upper()} offers: {', '.join(kernel.value for kernel in available)}")
    return [VDIAKernel(name) for name in names]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SABLE sparse matrix operations (SpMV / SpMM)",
        epilog=(
            "Examples:\n"
            "  %(prog)s --operation spmv --vbr-kernels blockmixed --vdia-kernels none --csr-kernels naive heart1\n"
            "  %(prog)s --operation spmv --vdia-kernels bandmkl --vbr-kernels none --csr-kernels mkl,spv8 ohne2\n"
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
    parser.add_argument("--vdia-kernels", type=str, default="all",
                        help="bandnaive, bandmkl, all, none, or comma-separated. "
                             "SpMV offers both; SpMM offers bandnaive only. Naming a "
                             "kernel an operation in --operation does not offer is an error.")
    parser.add_argument("--order", type=str, default=None, metavar="KIND,KIND",
                        help="Extraction order when both VBR and VDIA kernels are requested: "
                             "vdia,vbr (bands first, blocks in what is left) or vbr,vdia. The "
                             "formats are composed into one program in this order.")
    parser.add_argument("--extract", action="store_true",
                        help="Search the formats from scratch in --order (BlockDetector and the band "
                             "finder, each on the residual of the previous one) instead of replaying "
                             "stored regions, and record what was found under --extraction-dir. A "
                             "matrix already recorded there is reused unless --re-extract is given. "
                             "Requires an explicit matrix list.")
    parser.add_argument("--re-extract", action="store_true",
                        help="With --extract, search again even when a recording exists")
    parser.add_argument("--extraction-dir", type=str, default=None,
                        help="Where a composition's extractions are stored (default: "
                             "find-submatrices/results_<format tokens in order>, e.g. results_vdia075_vbr050)")
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
    parser.add_argument("--bands-results-dir", type=str, default=None,
                        help="Override directory for the stored single-format band YAML files "
                             "(default: find-submatrices/results_bands_075/)")
    # The block YAMLs under find-submatrices/results were produced by partition_matrix
    # with --min-density 0.5 --min-area 2500. The YAMLs do not record the threshold,
    # so it is stated here; pass --block-density for a directory produced
    # differently. Band directories carry their density in the name (results_bands_075).
    parser.add_argument("--block-density", type=float, default=0.5,
                        help="Minimum block density: what the stored block YAMLs were extracted with, "
                             "and what --extract searches with (default: %(default)s)")
    parser.add_argument("--band-density", type=float, default=None,
                        help="Minimum band density: what the stored band YAMLs were extracted with, and "
                             "what --extract searches with (default: read from the band directory's _NNN suffix)")
    parser.add_argument("--block-area", type=int, default=2500, help="--extract: minimum block area (default: 2500)")
    parser.add_argument("--block-gamma", type=float, default=1.5, help="--extract: partitioner gamma (default: 1.5)")
    parser.add_argument("--block-timeout", type=float, default=4.0 * 60.0 * 60.0,
                        help="--extract: partitioner time budget in seconds per matrix (default: 4 hours)")
    parser.add_argument("--block-threads", type=int, default=20, help="--extract: partitioner threads (default: 20)")
    args = parser.parse_args()

    global BANDS_RESULTS_DIR
    if args.bands_results_dir:
        _bands_path = pathlib.Path(args.bands_results_dir)
        if not _bands_path.is_absolute():
            _bands_path = FILEPATH / _bands_path
        BANDS_RESULTS_DIR = _bands_path
    band_density = args.band_density if args.band_density is not None else band_density_from_dir(BANDS_RESULTS_DIR)
    if band_density is None:
        parser.error(f"Cannot read a band density from {BANDS_RESULTS_DIR.name}; pass --band-density")
    densities = {"vbr": args.block_density, "vdia": band_density}
    block_params = {"min_area": args.block_area, "gamma": args.block_gamma,
                    "timeout_seconds": args.block_timeout, "threads": args.block_threads}

    operations = [Operation(op.strip()) for op in args.operation.split(",")]
    bench_flags = {Operation.SPMV: args.spmv_bench, Operation.SPMM: args.spmm_bench}
    for operation, given in bench_flags.items():
        if given is not None and operation not in operations:
            parser.error(f"--{operation.value}-bench given but {operation.value} is not in --operation")
    bench_iterations = {
        operation: given if given is not None else DEFAULT_BENCH_ITERATIONS[operation]
        for operation, given in bench_flags.items()
    }
    thread_counts = [int(t.strip()) for t in args.threads.split(",")]
    if any(thread_count < 1 for thread_count in thread_counts):
        parser.error("--threads counts must be at least 1")

    vbr_kernels = _resolve_vbr_kernels(args.vbr_kernels, parser)
    # Which VDIA kernels exist depends on the operation, so they are resolved
    # per requested operation; a name an operation does not offer is an error.
    vdia_kernels_by_op = {op: _resolve_vdia_kernels(op, args.vdia_kernels, parser) for op in operations}
    want_blocks = bool(vbr_kernels)
    want_bands = any(vdia_kernels_by_op.values())
    if not want_blocks and not want_bands:
        parser.error("At least one VBR or VDIA kernel must be selected")

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
    if args.extraction_dir:
        extraction_path = pathlib.Path(args.extraction_dir)
        if not extraction_path.is_absolute():
            extraction_path = FILEPATH / extraction_path
    else:
        extraction_path = extraction_dir(composition, densities)

    matrices = args.matrices or args.matrices_flag
    if args.extract:
        if not matrices:
            parser.error("--extract searches each matrix from scratch; pass the matrices explicitly")
    elif not matrices:
        matrices = get_available_matrices(composition, densities)
    else:
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
            matrix_rows, matrix_cols = A.shape
            matrix_nnz = A.nnz
            print(f"  Matrix shape: {matrix_rows} x {matrix_cols}, NNZ: {matrix_nnz}")

            regions = regions_for(matrix_name, A, composition, densities, block_params,
                                  args.extract, args.re_extract, extraction_path)
            for kind in composition:
                print(f"  {len(regions[kind])} {kind.upper()} regions")
            empty = [kind for kind in composition if not regions[kind]]
            if empty:
                # A format that found nothing would leave a program without its
                # dispatch, recorded under the format's name; refuse instead.
                print(f"  No {', '.join(kind.upper() for kind in empty)} regions for {matrix_name}; not benchmarked")
                empty_extractions.append(matrix_name)
                continue

            for operation in operations:
                csr_kernels = _resolve_csr_kernels(operation, args.csr_kernels, parser)
                kernels_by_kind = {"vbr": vbr_kernels, "vdia": vdia_kernels_by_op[operation]}
                for num_threads in thread_counts:
                    for kernels in itertools.product(*(kernels_by_kind[kind] for kind in composition)):
                        steps = [(kind, kernel, regions[kind]) for kind, kernel in zip(composition, kernels)]
                        format_tokens = "_".join(
                            f"{density_token(kind, densities[kind])}-{kernel.value}" for kind, kernel in zip(composition, kernels)
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
                            )

                            if result:
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
