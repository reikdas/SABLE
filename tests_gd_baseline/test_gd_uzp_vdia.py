from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

import numpy
import pytest
import scipy.io

BASE = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(BASE), str(BASE / "find-submatrices")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sable import Matrix, Plan
from sable.extractors import CSRConvertor
from sable.kernels import GradientDescent, UZPCSRSpmv
from sable.tensor import DenseInput
from utils.fileio import write_dense_values
from utils.utils import remove_outliers_deciles, set_ulimit
from find_matrices import get_matrix_info

BASELINE = "uzp"
RESULTS_JSON = BASE / "results" / f"algorithms_baseline_{BASELINE}_vdia.json"

MATRIX_NAMES = ["bcsstk28", "msc10848", "nd3k", "olafu", "thread"]
# These positive definite matrices are too ill-conditioned (kappa ~ 6e7..2e12)
# for fixed-step GD to reach DELTA=0.01 in any feasible budget, so run a fixed
# iteration budget and check the C loop tracks the Python reference exactly.
MAX_ITERATIONS = 1000
SEED = 0
# A rep is a 1000-iteration solve (~0.1-3 s); benchmark with 5 reps x 5 runs.
BENCH_REPS = 5


def load_suitesparse_matrix(name: str):
    """Load a SuiteSparse matrix through the shared ssgetpy cache."""
    info = get_matrix_info(name)
    if info is None:
        raise ValueError(f"Unknown SuiteSparse matrix: {name}")
    subdir, _ = info.localpath(format="MM", extract=True)
    mtx_path = os.path.join(subdir, f"{info.name}.mtx")
    if not os.path.exists(mtx_path):
        info.download(format="MM", extract=True)
    return scipy.io.mmread(mtx_path).tocsr()


def gd_reference(A, x0, b, alpha: float, delta: float, max_iterations: int):
    """Python implementation of the same loop; returns (solution, iterations)."""
    x = x0.copy()
    b_norm = float(b @ b)
    rn = numpy.inf
    iterations = 0
    while iterations < max_iterations and rn >= b_norm * delta * delta:
        y = A @ x
        r = numpy.asarray(y) - b
        rn = float(r @ r)
        x = x - alpha * r
        iterations += 1
    return x, iterations


def parse_result_values(output: str) -> numpy.ndarray:
    # The UZP kernel's setup shells out to uzp_prepare.sh, whose first-run
    # logs (blank lines included) precede the timing block on stdout -- so
    # anchor on the 'Iterations:' line instead of on the first blank line.
    lines = output.split("\n")
    for start, line in enumerate(lines):
        if line.startswith("Iterations:"):
            break
    else:
        raise ValueError("Executor output has no 'Iterations:' line")
    values = []
    for line in lines[start + 1 :]:
        line = line.strip()
        if line:
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
    """Execute the built benchmark binary once, pinned like bench_suitesparse."""
    command = [executor.binary_path]
    if not os.environ.get("SLURM_JOB_ID"):
        command = ["taskset", "-a", "-c", "0", *command]
    lib_dirs = [flag[2:] for flag in (executor.compile_command or []) if flag.startswith("-L")]
    env = os.environ.copy()
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    env.update(executor.runtime_env)
    output = subprocess.check_output(
        command,
        cwd=executor.runtime_cwd or executor.artifact_dir,
        env=env,
        preexec_fn=set_ulimit,
    )
    return output.decode("utf-8")


def run(matrix_name: str, bench: int = 5, runs: int = 1) -> dict:
    """Run the gradient descent baseline on one VDIA matrix; returns a result entry."""
    # -- Load the matrix; alpha = 1 / ||A||_inf ---------------------------
    A = load_suitesparse_matrix(matrix_name)
    n = A.shape[1]
    alpha = 1.0 / float(numpy.abs(A).sum(axis=1).max())
    delta = GradientDescent.DELTA

    # -- Random dense vectors x0 and b ------------------------------------
    rng = numpy.random.default_rng(SEED)
    x0 = rng.random(n)
    b = rng.random(n)

    filename = f"gd_spmv_vdia_{matrix_name}_{BASELINE}"
    x0_path = write_dense_values(f"{filename}_x0_{n}.vector", x0.tolist())

    # -- Build the plan: the whole matrix as one CSR dispatch + update ----
    matrix = Matrix(A, name=filename)
    plan = Plan(matrix, artifact_dir=str(BASE / "Generated_GD_Baseline_C" / filename))
    plan.rhs(DenseInput.vector(x0_path, n))

    csr = plan.extract(CSRConvertor())
    assert csr.nnz == int(A.nnz), "Expected the whole matrix in the CSR extraction"
    plan.dispatch(csr, UZPCSRSpmv())

    accum = plan.accumulate()
    plan.dispatch(accum, GradientDescent(b, alpha))

    # -- Compile once; run `runs` times pinned; verify each run ------------
    executor = plan.compile_loop(iters=MAX_ITERATIONS, filename=filename, bench=bench).build()
    x_ref, ref_iterations = gd_reference(A, x0, b, alpha, delta, MAX_ITERATIONS)

    iterations: list[int] = []
    dispatch_times: dict[int, list[float]] = {}
    for _ in range(runs):
        output = run_binary(executor)
        run_iterations = parse_iteration_counts(output)
        assert run_iterations == [ref_iterations] * bench, (run_iterations, ref_iterations)
        iterations.extend(run_iterations)
        # The update kernel writes x back into y, so the printed output is the solution.
        x_generated = parse_result_values(output)
        numpy.testing.assert_allclose(x_generated, x_ref, rtol=1e-7, atol=1e-9)
        residual = A @ x_generated - b
        assert float(residual @ residual) >= float(b @ b) * delta * delta
        for dispatch_id, times in parse_dispatch_times(output).items():
            dispatch_times.setdefault(dispatch_id, []).extend(times)

    # bench_suitesparse aggregation: pool every sample per dispatch across the
    # runs, drop values outside the 10th..90th percentiles, take the mean.
    mean_times = {}
    for dispatch_id, times in sorted(dispatch_times.items()):
        times_clean = remove_outliers_deciles(times)
        mean_times[f"dispatch_{dispatch_id}"] = statistics.mean(times_clean) if times_clean else 0
    total = sum(mean_times.values())
    return {
        "algorithm": "gradient_descent",
        "operation": "spmv",
        "baseline": BASELINE,
        "matrix_name": matrix_name,
        "matrix": {"rows": A.shape[0], "cols": A.shape[1], "nnz": int(A.nnz)},
        "split": {"csr_nnz": csr.nnz},
        "kernels": {
            f"dispatch_{i + 1}": type(dispatch.kernel).__name__
            for i, dispatch in enumerate(plan.dispatches)
        },
        "delta": delta,
        "alpha": alpha,
        "iterations": iterations,
        "reference_iterations": ref_iterations,
        "dispatch_times_ns": {key: round(value, 1) for key, value in mean_times.items()},
        "total_time_ns": round(total, 1),
        "time_per_iteration_ns": round(total / iterations[0], 1) if iterations[0] else 0.0,
    }


@pytest.mark.parametrize("matrix_name", MATRIX_NAMES)
def test_gd_spmv_vdia_uzp_baseline(matrix_name):
    """Gradient descent with the whole SpMV as UZP; hits the iteration cap unconverged."""
    entry = run(matrix_name, bench=1)
    assert entry["split"]["csr_nnz"] == entry["matrix"]["nnz"]
    assert entry["iterations"][0] == MAX_ITERATIONS
    assert entry["iterations"][0] == entry["reference_iterations"]


def main() -> int:
    entries = []
    for matrix_name in MATRIX_NAMES:
        print(f"[gd/spmv/{BASELINE}] {matrix_name}: running pipeline ({BENCH_REPS} reps x {BENCH_REPS} runs)...")
        entry = run(matrix_name, bench=BENCH_REPS, runs=BENCH_REPS)
        print(
            f"[gd/spmv/{BASELINE}] {matrix_name}: {entry['iterations'][0]} iterations "
            f"(matches Python reference), total {entry['total_time_ns'] / 1e6:.2f} ms, "
            f"{entry['time_per_iteration_ns'] / 1e3:.1f} us/iteration"
        )
        entries.append(entry)

    RESULTS_JSON.parent.mkdir(exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Results written to {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
