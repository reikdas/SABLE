from __future__ import annotations

import pathlib
import re
import statistics
import sys

import numpy
import scipy.io

BASE = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(BASE), str(BASE / "find-submatrices")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sable import Matrix, Plan
from sable.extractors import BlockDetectorSkip, CSRConvertor
from sable.kernels import Jacobi, MKLCSRSpmv, MixedVBRSpmv
from sable.tensor import DenseInput
from utils.fileio import parse_yaml_blocks, write_dense_values

# Jacobi needs a diagonally dominant operator: the raw SuiteSparse matrices either
# diverge (bundle1: rho(I - D^-1 A) = 1.87) or have zero diagonal entries, so this
# runs on the suitesparse-iter rewrite of eris1176 (generate.py), where Jacobi
# converges in 1,483,111 iterations (about 8 s compiled).
MATRIX_NAME = "eris1176"
MATRIX_PATH = BASE / "suitesparse-iter" / f"{MATRIX_NAME}.mtx"
MAX_ITERATIONS = 2000000
SEED = 0


def load_matrix():
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"{MATRIX_PATH} missing: run suitesparse-iter/generate.py {MATRIX_NAME}")
    return scipy.io.mmread(MATRIX_PATH).tocsr()


def jacobi_reference(A, x0, b, dinv, delta: float, max_iterations: int):
    """Python implementation of the same loop; returns (solution, iterations)."""
    x = x0.copy()
    b_norm = float(b @ b)
    rn = numpy.inf
    iterations = 0
    while iterations < max_iterations and rn >= b_norm * delta * delta:
        y = A @ x
        r = numpy.asarray(y) - b
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


def run(bench: int = 5) -> dict:
    """Run the full Jacobi pipeline on eris1176; returns a result entry."""
    # -- Load the matrix; the update divides by its diagonal ---------------
    A = load_matrix()
    n = A.shape[1]
    diag = A.diagonal()
    dinv = 1.0 / diag
    delta = Jacobi.DELTA

    # -- Random dense vectors x0 and b ------------------------------------
    rng = numpy.random.default_rng(SEED)
    x0 = rng.random(n)
    b = rng.random(n)

    filename = f"jacobi_spmv_{MATRIX_NAME}"
    x0_path = write_dense_values(f"{filename}_x0_{n}.vector", x0.tolist())

    # -- Build the plan: VBR blocks + CSR residual + update ---------------
    matrix = Matrix(A, name=filename)
    plan = Plan(matrix, artifact_dir=str(BASE / "Generated_Algorithms_C" / filename))
    plan.rhs(DenseInput.vector(x0_path, n))

    blocks = parse_yaml_blocks(
        str(BASE / "find-submatrices" / "results" / f"{MATRIX_NAME}.yaml")
    )
    vbr = plan.extract(BlockDetectorSkip(blocks=blocks))
    assert len(vbr.val.values) > 0, "Expected a VBR region"
    plan.dispatch(vbr, MixedVBRSpmv())

    csr = plan.extract(CSRConvertor())
    assert csr.nnz > 0, "Expected a CSR residual"
    plan.dispatch(csr, MKLCSRSpmv())
    split = {"vbr_values": len(vbr.val.values), "csr_nnz": csr.nnz}

    accum = plan.accumulate()
    plan.dispatch(accum, Jacobi(b, diag))

    # -- Compile, run, and verify against the Python reference ------------
    output = plan.compile_loop(iters=MAX_ITERATIONS, filename=filename, bench=bench).build().run()

    iterations = parse_iteration_counts(output)
    x_ref, ref_iterations = jacobi_reference(A, x0, b, dinv, delta, MAX_ITERATIONS)
    assert iterations == [ref_iterations] * bench, (iterations, ref_iterations)
    # The update kernel writes x back into y, so the printed output is the solution.
    x_generated = parse_result_values(output)
    numpy.testing.assert_allclose(x_generated, x_ref, rtol=1e-7, atol=1e-9)
    residual = A @ x_generated - b
    assert float(residual @ residual) < float(b @ b) * delta * delta * 1.1

    dispatch_times = parse_dispatch_times(output)
    mean_times = {
        f"dispatch_{dispatch_id}": statistics.mean(times)
        for dispatch_id, times in sorted(dispatch_times.items())
    }
    total = sum(mean_times.values())
    return {
        "algorithm": "jacobi",
        "operation": "spmv",
        "matrix_name": MATRIX_NAME,
        "matrix": {"rows": A.shape[0], "cols": A.shape[1], "nnz": int(A.nnz)},
        "split": split,
        "kernels": {
            f"dispatch_{i + 1}": type(dispatch.kernel).__name__
            for i, dispatch in enumerate(plan.dispatches)
        },
        "delta": delta,
        "iterations": iterations,
        "reference_iterations": ref_iterations,
        "dispatch_times_ns": {key: round(value, 1) for key, value in mean_times.items()},
        "total_time_ns": round(total, 1),
        "time_per_iteration_ns": round(total / iterations[0], 1) if iterations[0] else 0.0,
    }


def test_jacobi_spmv_eris1176_full_pipeline():
    """Jacobi on eris1176 split into VBR + CSR (one repetition)."""
    entry = run(bench=1)
    assert entry["split"]["vbr_values"] > 0
    assert entry["split"]["csr_nnz"] > 0
    assert 0 < entry["iterations"][0] < MAX_ITERATIONS
    assert entry["iterations"][0] == entry["reference_iterations"]
