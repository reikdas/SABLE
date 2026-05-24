#!/usr/bin/env python3
"""Compare compiled (staged) vs interpreted VDIA+CSR SpMV on a single matrix."""

import os
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "find-submatrices"))

from ssgetpy import fetch
from find_matrices import get_matrix_info, get_matrix_paths
from utils.fileio import parse_yaml_bands

from sable import Matrix, Plan
from sable.extractors import BandExtractorSkip, CSRConvertor
from sable.kernels import NaiveVDIASpmv, NaiveCSRSpmv
from sable.tensor import DenseInput

REPO_DIR = pathlib.Path(__file__).resolve().parent
SCRATCH = pathlib.Path("/local/scratch/a/das160")
SUITESPARSE_DIR = REPO_DIR / "Suitesparse"
DENSE_DIR = SCRATCH / "sable_dense_tmp"
ARTIFACT_BASE = SCRATCH / "sable_interp_bench"
BANDS_DIR = REPO_DIR / "find-submatrices" / "results_bands"

BENCH_ITERS = 20
MATRIX_NAME = "pdb1HYS"


def download_matrix(name: str) -> str:
    SUITESPARSE_DIR.mkdir(exist_ok=True)
    old = os.environ.get("SSGETPY_DIR")
    try:
        os.environ["SSGETPY_DIR"] = str(SUITESPARSE_DIR)
        info = get_matrix_info(name)
        if info is None:
            raise RuntimeError(f"Matrix {name!r} not found in SuiteSparse")
        fetch(name)
        _, _, _, matrix_path = get_matrix_paths(info)
        if not os.path.exists(matrix_path):
            alt = SUITESPARSE_DIR / info.name / f"{info.name}.mtx"
            if alt.exists():
                matrix_path = str(alt)
            else:
                raise RuntimeError(f"Matrix file not found at {matrix_path} or {alt}")
        return matrix_path
    finally:
        if old is not None:
            os.environ["SSGETPY_DIR"] = old
        elif "SSGETPY_DIR" in os.environ:
            del os.environ["SSGETPY_DIR"]


def write_dense_vector(val: float, size: int) -> str:
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    path = DENSE_DIR / f"generated_vector_{size}.vector"
    if not path.exists():
        with open(path, "w") as f:
            f.write(",".join([str(val)] * size))
    return str(path)


def parse_timings(output: str) -> dict[int, list[float]]:
    timings: dict[int, list[float]] = {}
    for line in output.splitlines():
        if line.startswith("Dispatch ") and "Part" not in line:
            parts = line.split(":", 1)
            idx = int(parts[0].split()[1])
            vals = [float(v) for v in parts[1].strip().rstrip(",").split(",") if v.strip()]
            timings[idx] = vals
    return timings


def run_one(mode: str, matrix_path: str, bands: list, bench: int) -> dict:
    artifact_dir = str(ARTIFACT_BASE / mode)
    os.makedirs(artifact_dir, exist_ok=True)

    matrix = Matrix(matrix_path, name=MATRIX_NAME)
    vec_path = write_dense_vector(1.0, matrix.ncols)

    plan = Plan(matrix, artifact_dir=artifact_dir)
    plan.rhs(DenseInput.vector(vec_path, matrix.ncols))
    vdia = plan.extract(BandExtractorSkip(bands=bands))
    plan.dispatch(vdia, NaiveVDIASpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmv())

    t0 = time.time()
    if mode == "compiled":
        executor = plan.compile(filename=MATRIX_NAME, bench=bench)
    else:
        executor = plan.interpret(filename=MATRIX_NAME, bench=bench)
    codegen_ms = (time.time() - t0) * 1000

    t0 = time.time()
    executor.build()
    compile_ms = (time.time() - t0) * 1000

    output = executor.run()
    timings = parse_timings(output)

    dispatch_medians = {}
    for idx, vals in sorted(timings.items()):
        median_ns = statistics.median(vals)
        dispatch_medians[idx] = median_ns

    total_ns = sum(dispatch_medians.values())
    return {
        "mode": mode,
        "codegen_ms": codegen_ms,
        "compile_ms": compile_ms,
        "dispatch_medians_us": {k: v / 1000 for k, v in dispatch_medians.items()},
        "total_us": total_ns / 1000,
        "c_path": executor.c_path,
    }


def main():
    print(f"Matrix: {MATRIX_NAME}")
    print(f"Operation: SpMV")
    print(f"Bench iterations: {BENCH_ITERS}")
    print()

    print("Downloading matrix...")
    matrix_path = download_matrix(MATRIX_NAME)
    print(f"  {matrix_path}")

    bands_path = BANDS_DIR / f"{MATRIX_NAME}.yaml"
    if not bands_path.exists():
        raise RuntimeError(f"Band YAML not found: {bands_path}")
    bands = parse_yaml_bands(str(bands_path))
    print(f"  Bands: {len(bands)} regions")
    total_segments = sum(len(b.get("segments", [])) if isinstance(b, dict) else len(getattr(b, "segments", []))
                         for b in bands)
    print(f"  Total segments: {total_segments}")
    print()

    results = {}
    for mode in ["compiled", "interpreted"]:
        print(f"Running {mode}...")
        r = run_one(mode, matrix_path, bands, BENCH_ITERS)
        results[mode] = r
        print(f"  Codegen:  {r['codegen_ms']:.1f} ms")
        print(f"  Compile:  {r['compile_ms']:.1f} ms")
        for idx, us in sorted(r["dispatch_medians_us"].items()):
            label = "VDIA" if idx == 1 else "CSR"
            print(f"  Dispatch {idx} ({label}): {us:.1f} us")
        print(f"  Total:    {r['total_us']:.1f} us")
        print(f"  C source: {r['c_path']}")
        print()

    c = results["compiled"]
    i = results["interpreted"]
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'':20s} {'Compiled':>12s} {'Interpreted':>12s} {'Ratio':>8s}")
    print(f"{'Codegen':<20s} {c['codegen_ms']:>10.1f}ms {i['codegen_ms']:>10.1f}ms {i['codegen_ms']/c['codegen_ms']:>7.2f}x")
    print(f"{'Compile':<20s} {c['compile_ms']:>10.1f}ms {i['compile_ms']:>10.1f}ms {i['compile_ms']/c['compile_ms']:>7.2f}x")
    for idx in sorted(c["dispatch_medians_us"]):
        label = "VDIA" if idx == 1 else "CSR"
        cv = c["dispatch_medians_us"][idx]
        iv = i["dispatch_medians_us"][idx]
        ratio = iv / cv if cv > 0 else float("inf")
        print(f"{'Dispatch '+str(idx)+' ('+label+')':<20s} {cv:>10.1f}us {iv:>10.1f}us {ratio:>7.3f}x")
    ratio = i["total_us"] / c["total_us"] if c["total_us"] > 0 else float("inf")
    print(f"{'Total':<20s} {c['total_us']:>10.1f}us {i['total_us']:>10.1f}us {ratio:>7.3f}x")


if __name__ == "__main__":
    main()
