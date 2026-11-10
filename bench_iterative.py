"""GD / Jacobi time-to-convergence benchmark on the suitesparse-iter matrices.

Amortization experiment: per matrix, the best SABLE VBR+CSR configuration
(inspection + one solve) against the best fully-CSR baseline (one solve, no
inspection). suitesparse-iter/generate.py picks the matrix values so that
inspection + one VBR+CSR solve beats one CSR solve for both solvers. Every
configuration's iteration count is checked against a scipy reference and its
solution against the residual stopping rule.

    python3 bench_iterative.py                       # 12 matrices, vbr_csr + csr
    python3 bench_iterative.py eris1176 heart2 --bench 1

One solve (Jacobi / GD): eris1176 8 s / 19 s; heart2, heart3, exdata_1 3-7 min;
c-30 11 / 23 min; heart1 27 / 56 min; TSC_OPF_1047 54 / 117 min; TSC_OPF_300
72 min / 2.6 h; bundle1 4.7 / 10.3 h; c-45 4.9 / 10.6 h; net100 4 / 9 days;
net125 8 / 19 days. Multiply by --bench and by six
configurations per matrix. A scipy reference solve (4-5x a C iteration) runs
only where predicted iterations x nnz <= REFERENCE_MAX_WORK, i.e. the quick tier
(eris1176, heart2, heart3, exdata_1, c-30, heart1); elsewhere the C iteration
count is checked against generate.py's exact-arithmetic bounds in results/suitesparse_iter.json
and the solution against the residual rule. Configurations already in --output
are skipped, so a run can be interrupted and resumed; use a fresh --output
whenever the values change.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys

import numpy
import scipy.io

BASE = pathlib.Path(__file__).resolve().parent
for _path in (str(BASE), str(BASE / "find-submatrices")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from bench_suitesparse import FIRST_CORE, _executor_env, remove_outliers_deciles, set_ulimit
from sable import Matrix, Plan
from sable.build_config import CSRKernel, VBRKernel
from sable.extractors import BlockDetectorSkip, CSRConvertor
from sable.kernels import (
    GradientDescent,
    Jacobi,
    MKLCSRSpmv,
    MKLVBRSpmv,
    MixedVBRSpmv,
    NaiveCSRSpmv,
    NaiveVBRSpmv,
    SPV8CSRSpmv,
    UZPCSRSpmv,
)
from sable.tensor import DenseInput
from utils.fileio import parse_yaml_blocks, write_dense_values

MATRIX_DIR = BASE / "suitesparse-iter"
VALUES = BASE / "results" / "suitesparse_iter.json"
BLOCKS_DIR = BASE / "find-submatrices" / "results"
RESULTS_JSON = BASE / "results" / "iterative_eval.json"
MAX_ITERATIONS = 2000000000  # must fit the int loop counter in the generated C (2^31 - 1)
REFERENCE_MAX_WORK = 1.5e13  # predicted iterations x nnz above which the scipy reference is skipped
# Rounding differences between C and scipy over tens of millions of iterations can
# move the step where the residual crosses the threshold (c-30 Jacobi: 31,536,844 vs 843).
ITERATION_SLACK = 1e-7
SEED = 0

COMPOSITIONS = ("vbr_csr", "csr")
VBR_KERNELS = {
    VBRKernel.NAIVE: NaiveVBRSpmv,
    VBRKernel.MIXED: MixedVBRSpmv,
    VBRKernel.MKL: MKLVBRSpmv,
}
CSR_KERNELS = {
    CSRKernel.NAIVE: NaiveCSRSpmv,
    CSRKernel.MKL: MKLCSRSpmv,
    CSRKernel.UZP: UZPCSRSpmv,
    CSRKernel.SPV8: SPV8CSRSpmv,
}


def gd_reference(A, x0, b, alpha, delta, max_iterations):
    x = x0.copy()
    b_norm = float(b @ b)
    rn = numpy.inf
    iterations = 0
    while iterations < max_iterations and rn >= b_norm * delta * delta:
        r = numpy.asarray(A @ x) - b
        rn = float(r @ r)
        x = x - alpha * r
        iterations += 1
    return x, iterations


def jacobi_reference(A, x0, b, dinv, delta, max_iterations):
    x = x0.copy()
    b_norm = float(b @ b)
    rn = numpy.inf
    iterations = 0
    while iterations < max_iterations and rn >= b_norm * delta * delta:
        r = numpy.asarray(A @ x) - b
        rn = float(r @ r)
        x = x - dinv * r
        iterations += 1
    return x, iterations


def parse_result_values(output: str) -> numpy.ndarray:
    values = []
    skip_timing = True
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            skip_timing = False
            continue
        if skip_timing:
            continue
        values.append(float(line))
    return numpy.array(values)


def parse_iteration_counts(output: str) -> list[int]:
    match = re.search(r"Iterations: ([\d,]+)", output)
    if match is None:
        raise ValueError("Executor output has no 'Iterations:' line")
    return [int(v) for v in match.group(1).rstrip(",").split(",")]


def parse_dispatch_times(output: str) -> dict[int, list[float]]:
    times: dict[int, list[float]] = {}
    for line in output.split("\n"):
        match = re.match(r"Dispatch (\d+): (.+)", line.strip())
        if match:
            times[int(match.group(1))] = [
                float(v) for v in match.group(2).rstrip(",").split(",") if v.strip()
            ]
    return times


def run_binary(executor) -> str:
    command = [executor.binary_path]
    if not os.environ.get("SLURM_JOB_ID"):
        command = ["taskset", "-a", "-c", str(FIRST_CORE), *command]
    output = subprocess.check_output(
        command,
        cwd=executor.runtime_cwd or executor.artifact_dir,
        env=_executor_env(executor.compile_command or [], executor.runtime_env),
        preexec_fn=set_ulimit,
    )
    return output.decode("utf-8")


def eligible_matrices() -> list[str]:
    values = json.load(open(VALUES))
    return [name for name in sorted(values, key=lambda m: values[m]["nnz"])
            if (MATRIX_DIR / f"{name}.mtx").exists() and (BLOCKS_DIR / f"{name}.yaml").exists()]


def iteration_bounds(name: str, algorithm: str) -> tuple[float, float]:
    """generate.py's exact-arithmetic [lower, upper] iteration bounds for this solve."""
    lb, ub = json.load(open(VALUES))[name][algorithm]["iter_bounds"]
    return lb, ub


_REFERENCE_CACHE: dict[tuple[str, str], tuple[numpy.ndarray, int]] = {}


def reference(algorithm: str, name: str, A, x0, b, alpha, dinv, delta):
    key = (algorithm, name)
    if key not in _REFERENCE_CACHE:
        if algorithm == "gd":
            _REFERENCE_CACHE[key] = gd_reference(A, x0, b, alpha, delta, MAX_ITERATIONS)
        else:
            _REFERENCE_CACHE[key] = jacobi_reference(A, x0, b, dinv, delta, MAX_ITERATIONS)
    return _REFERENCE_CACHE[key]


def run_config(algorithm, composition, name, vbr_kernel, csr_kernel, bench, runs):
    A = scipy.io.mmread(MATRIX_DIR / f"{name}.mtx").tocsr()
    n = A.shape[1]
    alpha = 1.0 / float(numpy.abs(A).sum(axis=1).max())
    diag = A.diagonal()
    dinv = 1.0 / diag
    delta = GradientDescent.DELTA if algorithm == "gd" else Jacobi.DELTA

    rng = numpy.random.default_rng(SEED)
    x0 = rng.random(n)
    b = rng.random(n)

    tag_parts = [algorithm, composition]
    if vbr_kernel is not None:
        tag_parts.append(vbr_kernel.value)
    tag_parts += [csr_kernel.value, name]
    tag = "_".join(tag_parts)
    x0_path = write_dense_values(f"{tag}_x0_{n}.vector", x0.tolist())

    matrix = Matrix(A, name=tag)
    plan = Plan(matrix, artifact_dir=str(BASE / "Generated_Algorithms_C" / tag))
    plan.rhs(DenseInput.vector(x0_path, n))

    split = {}
    dispatch_labels = []
    if vbr_kernel is not None:
        blocks = parse_yaml_blocks(str(BLOCKS_DIR / f"{name}.yaml"))
        vbr = plan.extract(BlockDetectorSkip(blocks=blocks))
        assert len(vbr.val.values) > 0, "Expected a VBR region"
        plan.dispatch(vbr, VBR_KERNELS[vbr_kernel]())
        split["vbr_values"] = len(vbr.val.values)
        dispatch_labels.append("vbr")
    csr = plan.extract(CSRConvertor())
    assert csr.nnz > 0, "Expected a CSR residual"
    plan.dispatch(csr, CSR_KERNELS[csr_kernel]())
    split["csr_nnz"] = csr.nnz
    dispatch_labels += ["csr", "update"]

    accum = plan.accumulate()
    if algorithm == "gd":
        plan.dispatch(accum, GradientDescent(b, alpha))
    else:
        plan.dispatch(accum, Jacobi(b, diag))
    executor = plan.compile_loop(iters=MAX_ITERATIONS, filename=tag, bench=bench).build()
    if csr_kernel == CSRKernel.UZP:
        run_binary(executor)  # first UZP run tunes and prints z_polyhedrator noise
    lb, ub = iteration_bounds(name, algorithm)
    use_reference = lb * A.nnz <= REFERENCE_MAX_WORK
    x_ref, ref_iterations = None, None
    if use_reference:
        x_ref, ref_iterations = reference(algorithm, name, A, x0, b, alpha, dinv, delta)
        assert ref_iterations < MAX_ITERATIONS, f"reference did not converge ({ref_iterations})"
        assert math.floor(lb) <= ref_iterations <= math.ceil(ub), (ref_iterations, lb, ub)

    iterations: list[int] = []
    dispatch_times: dict[int, list[float]] = {}
    for _ in range(runs):
        output = run_binary(executor)
        run_iterations = parse_iteration_counts(output)
        assert len(set(run_iterations)) == 1, run_iterations
        if use_reference:
            slack = max(1, ITERATION_SLACK * ref_iterations)
            assert abs(run_iterations[0] - ref_iterations) <= slack, (run_iterations, ref_iterations)
        else:
            assert math.floor(lb) <= run_iterations[0] <= math.ceil(ub), (run_iterations[0], lb, ub)
        iterations.extend(run_iterations)
        x_generated = parse_result_values(output)
        if use_reference:
            numpy.testing.assert_allclose(x_generated, x_ref, rtol=1e-7, atol=1e-9)
        residual = numpy.asarray(A @ x_generated) - b
        assert float(residual @ residual) < float(b @ b) * delta * delta * 1.1
        for dispatch_id, times in parse_dispatch_times(output).items():
            dispatch_times.setdefault(dispatch_id, []).extend(times)

    mean_times = {}
    for dispatch_id, times in sorted(dispatch_times.items()):
        times_clean = remove_outliers_deciles(times)
        mean_times[f"dispatch_{dispatch_id}"] = statistics.mean(times_clean) if times_clean else 0
    total = sum(mean_times.values())
    return {
        "algorithm": algorithm,
        "composition": composition,
        "matrix_name": name,
        "kernels": {
            **({"vbr": vbr_kernel.value} if vbr_kernel is not None else {}),
            "csr": csr_kernel.value,
        },
        "matrix": {"rows": A.shape[0], "cols": A.shape[1], "nnz": int(A.nnz)},
        "split": split,
        "delta": delta,
        "alpha": alpha if algorithm == "gd" else None,
        "iterations": iterations[0],
        "reference_iterations": ref_iterations,
        "iteration_bounds": [lb, ub],
        "verified_against": "scipy reference" if use_reference else "iteration bounds",
        "dispatch_labels": {f"dispatch_{i + 1}": label for i, label in enumerate(dispatch_labels)},
        "dispatch_times_ns": {key: round(value, 1) for key, value in mean_times.items()},
        "total_time_ns": round(total, 1),
        "time_per_iteration_ns": round(total / iterations[0], 1) if iterations[0] else 0.0,
    }


def configs_for(composition, vbr_kernels, csr_kernels):
    if composition == "csr":
        return [(None, csr) for csr in csr_kernels]
    return [(vbr, csr) for vbr in vbr_kernels for csr in csr_kernels]


def config_key(entry):
    return (entry["algorithm"], entry["composition"], entry["matrix_name"],
            entry["kernels"].get("vbr"), entry["kernels"]["csr"])


def dump_json(obj, path: pathlib.Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate gradient descent and Jacobi iteration over the suitesparse-iter "
        "matrices: SABLE VBR+CSR configurations against fully-CSR baselines."
    )
    parser.add_argument("matrices", nargs="*", help="Restrict to these matrix names.")
    parser.add_argument("--algorithms", default="gd,jacobi")
    parser.add_argument("--compositions", default=",".join(COMPOSITIONS),
                        help="Comma-separated: vbr_csr, csr.")
    parser.add_argument("--vbr-kernels", default="blockmixed",
                        help="Comma-separated: blocknaive, blockmixed, blockmkl.")
    parser.add_argument("--csr-kernels", default="naive,mkl,uzp",
                        help="Comma-separated: naive, mkl, uzp, spv8.")
    parser.add_argument("--bench", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", default=str(RESULTS_JSON))
    args = parser.parse_args()

    algorithms = args.algorithms.split(",")
    compositions = args.compositions.split(",")
    unknown = [c for c in compositions if c not in COMPOSITIONS]
    if unknown:
        parser.error(f"unknown compositions {unknown}; choose from {COMPOSITIONS}")
    vbr_kernels = [VBRKernel(v) for v in args.vbr_kernels.split(",") if v]
    csr_kernels = [CSRKernel(v) for v in args.csr_kernels.split(",") if v]

    names = eligible_matrices()
    if args.matrices:
        names = [n for n in names if n in args.matrices]
    todo = [(algorithm, composition, name, vbr, csr)
            for composition in compositions
            for name in names
            for algorithm in algorithms
            for vbr, csr in configs_for(composition, vbr_kernels, csr_kernels)]
    print(f"{len(todo)} configurations to run", flush=True)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    results = json.load(open(output_path)) if output_path.exists() else []
    done = {config_key(e) for e in results if "error" not in e}

    for i, (algorithm, composition, name, vbr, csr) in enumerate(todo):
        key = (algorithm, composition, name, vbr.value if vbr else None, csr.value)
        label = (f"[{i + 1}/{len(todo)}] {algorithm}/{composition}/{name} "
                 f"vbr={vbr.value if vbr else '-'} csr={csr.value}")
        if key in done:
            print(f"{label}: already done, skipping", flush=True)
            continue
        try:
            entry = run_config(algorithm, composition, name, vbr, csr, args.bench, args.runs)
            print(f"{label}: {entry['iterations']} iters, "
                  f"{entry['time_per_iteration_ns'] / 1e3:.1f} us/iter", flush=True)
        except Exception as exc:
            entry = {
                "algorithm": algorithm, "composition": composition, "matrix_name": name,
                "kernels": {**({"vbr": vbr.value} if vbr else {}), "csr": csr.value},
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"{label}: ERROR {entry['error']}", flush=True)
        results = [e for e in results if config_key(e) != key]
        results.append(entry)
        dump_json(results, output_path)

    failures = sum(1 for e in results if "error" in e)
    print(f"\nDONE: {len(results) - failures} ok, {failures} errors -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
