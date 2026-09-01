from __future__ import annotations

import argparse
import json
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

from sable import Matrix, Plan
from sable.build_config import CSRKernel, VBRKernel, VDIAKernel
from sable.extractors import BandExtractorSkip, BlockDetectorSkip, CSRConvertor
from sable.kernels import (
    GradientDescent,
    Jacobi,
    MKLCSRSpmv,
    MKLDIASpmv,
    MKLVBRSpmv,
    MixedVBRSpmv,
    NaiveCSRSpmv,
    NaiveVBRSpmv,
    NaiveVDIASpmv,
    SPV8CSRSpmv,
    UZPCSRSpmv,
)
from sable.tensor import DenseInput
from utils.fileio import parse_yaml_bands, parse_yaml_blocks, write_dense_values
from utils.utils import remove_outliers_deciles, set_ulimit

MATRIX_DIR = BASE / "suitesparse-iter"
BLOCKS_DIR = BASE / "find-submatrices" / "results"
BANDS_DIR = BASE / "find-submatrices" / "results_bands_075"
RESULTS_JSON = BASE / "results" / "iterative_eval.json"
MAX_ITERATIONS = 1000000000
SEED = 0

# Paper benchmark suites (SABLE-paper appendix table:matrices and vdia_csr_table).
VBR_SUITE = """eris1176 c-30 lowThrust_3 jendrec1 orani678 vsp_c-30_data_data lp_fit2d lp_osa_07 cari
lowThrust_4 c-45 brainpc2 lowThrust_5 TSOPF_RS_b162_c1 lowThrust_6 lowThrust_7 lowThrust_8
lowThrust_9 lowThrust_10 lowThrust_11 lowThrust_12 lowThrust_13 TSOPF_RS_b39_c7 FX_March2010
lp_osa_14 c-57 lp_osa_30 TSOPF_FS_b162_c1 TSOPF_RS_b162_c3 heart2 heart3 TSOPF_RS_b39_c19
bundle1 TSOPF_RS_b162_c4 TSC_OPF_300 case39 connectus std1_Jac2 heart1 std1_Jac3
TSOPF_RS_b300_c1 Zd_Jac2 Zd_Jac6 Zd_Jac3 TSC_OPF_1047 net100 gupta1 exdata_1 net125
TSOPF_RS_b300_c2 net150 TSOPF_RS_b678_c1 TSOPF_RS_b300_c3 TSOPF_RS_b678_c2 TSOPF_RS_b2383""".split()

VDIA_SUITE = """vsp_c-30_data_data bcsstk28 cegb2802 cegb2919 heart2 heart3 nemeth19 nemeth20 olafu
nemeth21 msc10848 nemeth22 heart1 nemeth24 nemeth23 nemeth25 nemeth26 opt1 bcsstk32
TSC_OPF_1047 pkustk07 pkustk08 nd3k gupta3 ohne2 memchip circuit5M_dc CoupCons3D""".split()

VBR_KERNELS = {
    VBRKernel.NAIVE: NaiveVBRSpmv,
    VBRKernel.MIXED: MixedVBRSpmv,
    VBRKernel.MKL: MKLVBRSpmv,
}
VDIA_KERNELS = {
    VDIAKernel.NAIVE: NaiveVDIASpmv,
    VDIAKernel.MKL_DIA: MKLDIASpmv,
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


def eligible_matrices(composition: str) -> list[str]:
    if composition == "csr":
        suite = VBR_SUITE + [n for n in VDIA_SUITE if n not in VBR_SUITE]
    elif composition == "vbr_csr":
        suite = VBR_SUITE
    else:
        suite = VDIA_SUITE
    names = []
    for name in suite:
        if not (MATRIX_DIR / f"{name}.mtx").exists():
            continue
        if composition in ("vbr_csr", "vdia_vbr_csr") and not (BLOCKS_DIR / f"{name}.yaml").exists():
            continue
        if composition in ("vdia_csr", "vdia_vbr_csr") and not (BANDS_DIR / f"{name}.yaml").exists():
            continue
        names.append(name)
    return names


_REFERENCE_CACHE: dict[tuple[str, str], tuple[numpy.ndarray, int]] = {}


def reference(algorithm: str, name: str, A, x0, b, alpha, dinv, delta):
    key = (algorithm, name)
    if key not in _REFERENCE_CACHE:
        if algorithm == "gd":
            _REFERENCE_CACHE[key] = gd_reference(A, x0, b, alpha, delta, MAX_ITERATIONS)
        else:
            _REFERENCE_CACHE[key] = jacobi_reference(A, x0, b, dinv, delta, MAX_ITERATIONS)
    return _REFERENCE_CACHE[key]


def run_config(algorithm, composition, name, vdia_kernel, vbr_kernel, csr_kernel, bench, runs):
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
    if vdia_kernel is not None:
        tag_parts.append(vdia_kernel.value)
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
    if vdia_kernel is not None:
        dispatch_labels.append("vdia")
    if vbr_kernel is not None:
        dispatch_labels.append("vbr")
    dispatch_labels += ["csr", "update"]
    if vdia_kernel is not None:
        bands = parse_yaml_bands(str(BANDS_DIR / f"{name}.yaml"))
        vdia = plan.extract(BandExtractorSkip(bands=bands))
        assert len(vdia.val.values) > 0, "Expected a VDIA region"
        plan.dispatch(vdia, VDIA_KERNELS[vdia_kernel]())
        split["vdia_values"] = len(vdia.val.values)
    if vbr_kernel is not None:
        blocks = parse_yaml_blocks(str(BLOCKS_DIR / f"{name}.yaml"))
        vbr = plan.extract(BlockDetectorSkip(blocks=blocks))
        assert len(vbr.val.values) > 0, "Expected a VBR region"
        plan.dispatch(vbr, VBR_KERNELS[vbr_kernel]())
        split["vbr_values"] = len(vbr.val.values)
    csr = plan.extract(CSRConvertor())
    assert csr.nnz > 0, "Expected a CSR residual"
    plan.dispatch(csr, CSR_KERNELS[csr_kernel]())
    split["csr_nnz"] = csr.nnz

    accum = plan.accumulate()
    if algorithm == "gd":
        plan.dispatch(accum, GradientDescent(b, alpha))
    else:
        plan.dispatch(accum, Jacobi(b, diag))
    executor = plan.compile_loop(iters=MAX_ITERATIONS, filename=tag, bench=bench).build()
    if csr_kernel == CSRKernel.UZP:
        run_binary(executor)  # first UZP run tunes and prints z_polyhedrator noise
    x_ref, ref_iterations = reference(algorithm, name, A, x0, b, alpha, dinv, delta)
    assert ref_iterations < MAX_ITERATIONS, f"reference did not converge ({ref_iterations})"

    iterations: list[int] = []
    dispatch_times: dict[int, list[float]] = {}
    for _ in range(runs):
        output = run_binary(executor)
        run_iterations = parse_iteration_counts(output)
        assert run_iterations == [ref_iterations] * bench, (run_iterations, ref_iterations)
        iterations.extend(run_iterations)
        x_generated = parse_result_values(output)
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
            **({"vdia": vdia_kernel.value} if vdia_kernel is not None else {}),
            **({"vbr": vbr_kernel.value} if vbr_kernel is not None else {}),
            "csr": csr_kernel.value,
        },
        "matrix": {"rows": A.shape[0], "cols": A.shape[1], "nnz": int(A.nnz)},
        "split": split,
        "delta": delta,
        "alpha": alpha if algorithm == "gd" else None,
        "iterations": iterations[0],
        "reference_iterations": ref_iterations,
        "dispatch_labels": {f"dispatch_{i + 1}": label for i, label in enumerate(dispatch_labels)},
        "dispatch_times_ns": {key: round(value, 1) for key, value in mean_times.items()},
        "total_time_ns": round(total, 1),
        "time_per_iteration_ns": round(total / iterations[0], 1) if iterations[0] else 0.0,
    }


def configs_for(composition, vdia_kernels, vbr_kernels, csr_kernels):
    if composition == "csr":
        return [(None, None, csr) for csr in csr_kernels]
    if composition == "vbr_csr":
        return [(None, vbr, csr) for vbr in vbr_kernels for csr in csr_kernels]
    if composition == "vdia_csr":
        return [(vdia, None, csr) for vdia in vdia_kernels for csr in csr_kernels]
    return [(vdia, vbr, csr) for vdia in vdia_kernels for vbr in vbr_kernels for csr in csr_kernels]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate gradient descent and Jacobi iteration over the "
        "suitesparse-iter matrices for the paper's composed-format configurations."
    )
    parser.add_argument("matrices", nargs="*", help="Restrict to these matrix names.")
    parser.add_argument("--algorithms", default="gd,jacobi")
    parser.add_argument("--compositions", default="vbr_csr,vdia_csr,vdia_vbr_csr,csr")
    parser.add_argument("--vbr-kernels", default="blockmixed",
                        help="Comma-separated: blocknaive, blockmixed, blockmkl.")
    parser.add_argument("--vdia-kernels", default="bandnaive,bandmkl",
                        help="Comma-separated: bandnaive, bandmkl.")
    parser.add_argument("--csr-kernels", default="naive,mkl,uzp",
                        help="Comma-separated: naive, mkl, uzp, spv8 (spv8 excluded "
                        "by default: the vendored SpMV kernel is known-broken).")
    parser.add_argument("--bench", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", default=str(RESULTS_JSON))
    args = parser.parse_args()

    algorithms = args.algorithms.split(",")
    compositions = args.compositions.split(",")
    vbr_kernels = [VBRKernel(v) for v in args.vbr_kernels.split(",") if v]
    vdia_kernels = [VDIAKernel(v) for v in args.vdia_kernels.split(",") if v]
    csr_kernels = [CSRKernel(v) for v in args.csr_kernels.split(",") if v]

    todo = []
    for composition in compositions:
        names = eligible_matrices(composition)
        if args.matrices:
            names = [n for n in names if n in args.matrices]
        for name in names:
            for algorithm in algorithms:
                for vdia, vbr, csr in configs_for(composition, vdia_kernels, vbr_kernels, csr_kernels):
                    todo.append((algorithm, composition, name, vdia, vbr, csr))
    print(f"{len(todo)} configurations to run", flush=True)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    results = []
    if output_path.exists():
        results = json.load(open(output_path))
    done = {
        (e["algorithm"], e["composition"], e["matrix_name"],
         e["kernels"].get("vdia"), e["kernels"].get("vbr"), e["kernels"]["csr"])
        for e in results if "error" not in e
    }

    for i, (algorithm, composition, name, vdia, vbr, csr) in enumerate(todo):
        key = (algorithm, composition, name,
               vdia.value if vdia else None, vbr.value if vbr else None, csr.value)
        label = f"[{i + 1}/{len(todo)}] {algorithm}/{composition}/{name} " \
                f"vdia={vdia.value if vdia else '-'} vbr={vbr.value if vbr else '-'} csr={csr.value}"
        if key in done:
            print(f"{label}: already done, skipping", flush=True)
            continue
        try:
            entry = run_config(algorithm, composition, name, vdia, vbr, csr, args.bench, args.runs)
            print(f"{label}: {entry['iterations']} iters, "
                  f"{entry['time_per_iteration_ns'] / 1e3:.1f} us/iter", flush=True)
        except Exception as exc:
            entry = {
                "algorithm": algorithm, "composition": composition, "matrix_name": name,
                "kernels": {
                    **({"vdia": vdia.value} if vdia else {}),
                    **({"vbr": vbr.value} if vbr else {}),
                    "csr": csr.value,
                },
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"{label}: ERROR {entry['error']}", flush=True)
        results = [e for e in results
                   if (e["algorithm"], e["composition"], e["matrix_name"],
                       e["kernels"].get("vdia"), e["kernels"].get("vbr"),
                       e["kernels"]["csr"]) != key]
        results.append(entry)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    failures = sum(1 for e in results if "error" in e)
    print(f"\nDONE: {len(results) - failures} ok, {failures} errors -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
