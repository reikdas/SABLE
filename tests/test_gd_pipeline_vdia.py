from __future__ import annotations

import os
import pathlib
import re
import statistics
import sys

import numpy
import pytest
import scipy.io

BASE = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(BASE), str(BASE / "find-submatrices")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sable import Matrix, Plan
from sable.extractors import BandExtractorSkip, CSRConvertor
from sable.kernels import GradientDescent, MKLCSRSpmv, MKLDIASpmv
from sable.tensor import DenseInput
from utils.fileio import parse_yaml_bands, write_dense_values
from find_matrices import get_matrix_info

MATRIX_NAMES = ["bcsstk28", "msc10848", "nd3k", "olafu", "thread"]
# These positive definite matrices are too ill-conditioned (kappa ~ 6e7..2e12)
# for fixed-step GD to reach DELTA=0.01 in any feasible budget, so run a fixed
# iteration budget and check the C loop tracks the Python reference exactly.
MAX_ITERATIONS = 1000
SEED = 0
BANDS_DIR = BASE / "find-submatrices" / "results_bands_075"


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


def run(matrix_name: str, bench: int = 5) -> dict:
    """Run the gradient descent pipeline with VDIA + CSR dispatches; returns a result entry."""
    # -- Load the matrix; alpha = 1 / ||A||_inf ---------------------------
    A = load_suitesparse_matrix(matrix_name)
    n = A.shape[1]
    alpha = 1.0 / float(numpy.abs(A).sum(axis=1).max())
    delta = GradientDescent.DELTA

    # -- Random dense vectors x0 and b ------------------------------------
    rng = numpy.random.default_rng(SEED)
    x0 = rng.random(n)
    b = rng.random(n)

    filename = f"gd_spmv_vdia_{matrix_name}"
    x0_path = write_dense_values(f"{filename}_x0_{n}.vector", x0.tolist())

    # -- Build the plan: VDIA bands + CSR residual + update ---------------
    matrix = Matrix(A, name=filename)
    plan = Plan(matrix, artifact_dir=str(BASE / "Generated_Algorithms_C" / filename))
    plan.rhs(DenseInput.vector(x0_path, n))

    bands = parse_yaml_bands(str(BANDS_DIR / f"{matrix_name}.yaml"))
    vdia = plan.extract(BandExtractorSkip(bands=bands))
    assert len(vdia.val.values) > 0, "Expected a VDIA region"
    plan.dispatch(vdia, MKLDIASpmv())

    csr = plan.extract(CSRConvertor())
    assert csr.nnz > 0, "Expected a CSR residual"
    plan.dispatch(csr, MKLCSRSpmv())
    split = {"vdia_values": len(vdia.val.values), "csr_nnz": csr.nnz}

    accum = plan.accumulate()
    plan.dispatch(accum, GradientDescent(b, alpha))

    # -- Compile, run, and verify against the Python reference ------------
    output = plan.compile_loop(iters=MAX_ITERATIONS, filename=filename, bench=bench).build().run()

    iterations = parse_iteration_counts(output)
    x_ref, ref_iterations = gd_reference(A, x0, b, alpha, delta, MAX_ITERATIONS)
    assert iterations == [ref_iterations] * bench, (iterations, ref_iterations)
    # The update kernel writes x back into y, so the printed output is the solution.
    x_generated = parse_result_values(output)
    numpy.testing.assert_allclose(x_generated, x_ref, rtol=1e-7, atol=1e-9)
    residual = A @ x_generated - b
    assert float(residual @ residual) >= float(b @ b) * delta * delta

    dispatch_times = parse_dispatch_times(output)
    mean_times = {
        f"dispatch_{dispatch_id}": statistics.mean(times)
        for dispatch_id, times in sorted(dispatch_times.items())
    }
    total = sum(mean_times.values())
    return {
        "algorithm": "gradient_descent",
        "operation": "spmv",
        "matrix_name": matrix_name,
        "matrix": {"rows": A.shape[0], "cols": A.shape[1], "nnz": int(A.nnz)},
        "split": split,
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
def test_gd_spmv_vdia_full_pipeline(matrix_name):
    """Gradient descent split into VDIA + CSR; hits the iteration cap unconverged."""
    entry = run(matrix_name, bench=1)
    assert entry["split"]["vdia_values"] > 0
    assert entry["split"]["csr_nnz"] > 0
    assert entry["iterations"][0] == MAX_ITERATIONS
    assert entry["iterations"][0] == entry["reference_iterations"]
