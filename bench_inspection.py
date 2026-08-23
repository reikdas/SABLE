#!/usr/bin/env python3
"""Measure SABLE inspection (extraction) + codegen + gcc compile times.

Runs the real extraction pipeline (not the precomputed-YAML skip paths) on the
three matrix sets evaluated in the SABLE paper:

  vbr_csr       55 matrices: BlockDetector -> CSRConvertor
  vdia_csr      28 matrices: BandExtractor -> CSRConvertor
                (24 band-extractor matrices + 4 Fukaya et al. matrices)
  vdia_vbr_csr  24 matrices: BandExtractor -> BlockDetector -> CSRConvertor
                (block search runs on the residual left after band extraction)

For every matrix it records, into results/inspection/inspection_<set>_<op>.json:
  - VDIA band search time (plus to_csr/pack/residual breakdown and a
    found_at_seconds timestamp for every accepted band),
  - VBR block partitioner time (subprocess wall time, in-binary read/search
    split, timeout flag, and a found_at_seconds timestamp for every accepted
    block) -- the per-region timestamps let us evaluate post-hoc what a shorter
    search budget would have found,
  - CSR conversion time,
  - codegen time (writing the .c and .sabledata files),
  - gcc compile time of the generated code.

Results are written after every matrix; matrices that already have a
successful entry are skipped on re-runs, so the script is safe to restart.
Downloaded matrices and generated artifacts are never deleted.

Render tables with gen_inspection_table.py.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Optional

from scipy.io import mmread
from scipy.sparse import csr_matrix

from bench_suitesparse import (
    COMPILE_TIMEOUT,
    download_matrix_from_suitesparse,
    _csr_spmm_kernel,
    _csr_spmv_kernel,
    _vbr_spmm_kernel,
    _vbr_spmv_kernel,
    _vdia_spmm_kernel,
    _vdia_spmv_kernel,
)
from sable import Matrix, Operation, Plan
from sable.build_config import CSRKernel, VBRKernel, VDIAKernel
from sable.compiler import build_compile_command_for_plan
from sable.extractors import BandExtractor, BlockDetector, CSRConvertor
from sable.tensor import DenseInput, DenseLayout
from utils.fileio import write_dense_matrix, write_dense_vector

FILEPATH = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = FILEPATH / "results" / "inspection"
SPMM_NRHS = 512

# The 55 matrices of the VBR+CSR evaluation (paper Sec. Evaluation; the
# matrices of SABLE-paper/results/sable_spmv_blas_mkl.json).
VBR_CSR_55 = [
    "FX_March2010", "TSC_OPF_1047", "TSC_OPF_300", "TSOPF_FS_b162_c1",
    "TSOPF_RS_b162_c1", "TSOPF_RS_b162_c3", "TSOPF_RS_b162_c4",
    "TSOPF_RS_b2383", "TSOPF_RS_b300_c1", "TSOPF_RS_b300_c2",
    "TSOPF_RS_b300_c3", "TSOPF_RS_b39_c19", "TSOPF_RS_b39_c7",
    "TSOPF_RS_b678_c1", "TSOPF_RS_b678_c2", "Zd_Jac2", "Zd_Jac3", "Zd_Jac6",
    "brainpc2", "bundle1", "c-30", "c-45", "c-57", "cari", "case39",
    "connectus", "eris1176", "exdata_1", "gupta1", "heart1", "heart2",
    "heart3", "jendrec1", "lowThrust_10", "lowThrust_11", "lowThrust_12",
    "lowThrust_13", "lowThrust_3", "lowThrust_4", "lowThrust_5",
    "lowThrust_6", "lowThrust_7", "lowThrust_8", "lowThrust_9", "lp_fit2d",
    "lp_osa_07", "lp_osa_14", "lp_osa_30", "net100", "net125", "net150",
    "orani678", "std1_Jac2", "std1_Jac3", "vsp_c-30_data_data",
]

# The 24 matrices the band extractor identified (paper VDIA+VBR+CSR set;
# spmv_vdia_vbr_csr_d075.json minus the excluded 'thread' matrix).
VDIA_24 = [
    "TSC_OPF_1047", "bcsstk28", "bcsstk32", "cegb2802", "cegb2919", "gupta3",
    "heart1", "heart2", "heart3", "msc10848", "nd3k", "nemeth19", "nemeth20",
    "nemeth21", "nemeth22", "nemeth23", "nemeth24", "nemeth25", "nemeth26",
    "olafu", "opt1", "pkustk07", "pkustk08", "vsp_c-30_data_data",
]

# The 4 matrices from Fukaya et al. (2021) added to the VDIA+CSR evaluation.
FUKAYA_4 = ["circuit5M_dc", "memchip", "ohne2", "CoupCons3D"]

MATRIX_SETS = {
    "vbr_csr": VBR_CSR_55,
    "vdia_csr": VDIA_24 + FUKAYA_4,
    "vdia_vbr_csr": VDIA_24,
}


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def _generated_vector_path(size: int) -> str:
    dense_dir = os.environ.get("SABLE_DENSE_TENSOR_DIR") or str(FILEPATH / "Generated_dense_tensors")
    return os.path.abspath(os.path.join(dense_dir, f"generated_vector_{size}.vector"))


def _generated_matrix_path(rows: int, cols: int) -> str:
    dense_dir = os.environ.get("SABLE_DENSE_TENSOR_DIR") or str(FILEPATH / "Generated_dense_tensors")
    return os.path.abspath(os.path.join(dense_dir, f"generated_matrix_{rows}x{cols}.matrix"))


def _make_kernels(operation: Operation, args) -> Dict[str, Any]:
    if operation == Operation.SPMV:
        return {
            "vdia": _vdia_spmv_kernel(VDIAKernel(args.vdia_kernel)),
            "vbr": _vbr_spmv_kernel(VBRKernel(args.vbr_kernel)),
            "csr": _csr_spmv_kernel(CSRKernel(args.csr_kernel)),
        }
    return {
        "vdia": _vdia_spmm_kernel(VDIAKernel(args.vdia_kernel)),
        "vbr": _vbr_spmm_kernel(VBRKernel(args.vbr_kernel)),
        "csr": _csr_spmm_kernel(CSRKernel(args.csr_kernel)),
    }


def run_matrix(matrix_name: str, set_name: str, operation: Operation, args) -> Optional[Dict[str, Any]]:
    print(f"\n=== [{set_name}/{operation.value}] {matrix_name} ===")
    download_result = download_matrix_from_suitesparse(matrix_name)
    if download_result is None:
        print(f"  Failed to download {matrix_name}")
        return None
    mtx_path = download_result[0]

    print(f"  Loading {mtx_path} ...")
    load_start = time.perf_counter()
    A = csr_matrix(mmread(mtx_path))
    load_seconds = time.perf_counter() - load_start
    rows, cols = A.shape
    nnz = int(A.nnz)
    print(f"  Shape {rows} x {cols}, nnz {nnz} (load: {load_seconds:.1f}s)")

    matrix = Matrix(A, name=matrix_name)
    artifact_dir = str(pathlib.Path(args.artifact_dir) / f"{operation.value}_{set_name}")
    os.makedirs(artifact_dir, exist_ok=True)
    plan = Plan(matrix, artifact_dir=artifact_dir)

    if operation == Operation.SPMV:
        write_dense_vector(1.0, cols)
        plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    else:
        write_dense_matrix(1.0, cols, SPMM_NRHS)
        plan.rhs(
            DenseInput.matrix(
                _generated_matrix_path(cols, SPMM_NRHS),
                shape=(cols, SPMM_NRHS),
                layout=DenseLayout.ROW_MAJOR,
            )
        )

    kernels = _make_kernels(operation, args)
    phases: Dict[str, Any] = {}

    if set_name in ("vdia_csr", "vdia_vbr_csr"):
        print(f"  [VDIA] band search (min_density={args.band_min_density}) ...")
        extractor = BandExtractor(min_density=args.band_min_density)
        nnz_before = plan.residual.nnz
        vdia_fmt, wall = _timed(plan.extract, extractor)
        claimed = nnz_before - plan.residual.nnz
        phases["vdia"] = {
            "wall_seconds": wall,
            "claimed_nnz": claimed,
            "residual_nnz_after": plan.residual.nnz,
            **(extractor.last_timing or {}),
        }
        plan.dispatch(vdia_fmt, kernels["vdia"])
        print(
            f"  [VDIA] {len(phases['vdia'].get('bands', []))} band(s), "
            f"claimed {claimed}/{nnz} nnz, search "
            f"{phases['vdia'].get('search_seconds', 0):.1f}s (wall {wall:.1f}s)"
        )

    if set_name in ("vbr_csr", "vdia_vbr_csr"):
        print(
            f"  [VBR] block search (timeout={args.block_timeout:.0f}s, "
            f"threads={args.block_threads}) ..."
        )
        detector = BlockDetector(
            min_density=args.block_min_density,
            min_area=args.block_min_area,
            gamma=args.block_gamma,
            timeout_seconds=args.block_timeout,
            threads=args.block_threads,
        )
        nnz_before = plan.residual.nnz
        vbr_fmt, wall = _timed(plan.extract, detector)
        claimed = nnz_before - plan.residual.nnz
        phases["vbr"] = {
            "wall_seconds": wall,
            "claimed_nnz": claimed,
            "residual_nnz_after": plan.residual.nnz,
            **(detector.last_timing or {}),
        }
        plan.dispatch(vbr_fmt, kernels["vbr"])
        timeout_note = " [TIMEOUT]" if phases["vbr"].get("timeout") else ""
        print(
            f"  [VBR] {len(phases['vbr'].get('blocks', []))} block(s), "
            f"claimed {claimed}/{nnz} nnz, partitioner "
            f"{phases['vbr'].get('partitioner_seconds', 0):.1f}s (wall {wall:.1f}s){timeout_note}"
        )

    convertor = CSRConvertor()
    nnz_before = plan.residual.nnz
    csr_fmt, wall = _timed(plan.extract, convertor)
    phases["csr"] = {
        "wall_seconds": wall,
        "claimed_nnz": nnz_before,
        **(convertor.last_timing or {}),
    }
    plan.dispatch(csr_fmt, kernels["csr"])
    print(f"  [CSR] claimed {nnz_before}/{nnz} nnz, convert {phases['csr']['convert_seconds']:.2f}s")

    print("  [CodeGen] emitting C source and data ...")
    executor, codegen_wall = _timed(plan.compile, filename=matrix_name, bench=args.bench)
    phases["codegen"] = {
        "wall_seconds": codegen_wall,
        "codegen_time_ms": executor.codegen_time_ms,
    }
    print(f"  [CodeGen] {codegen_wall:.2f}s -> {executor.c_path}")

    output_path = os.path.abspath(os.path.join(artifact_dir, matrix_name))
    command = build_compile_command_for_plan(plan, executor.c_path, output_path)
    print(f"  [gcc] compiling (timeout={args.compile_timeout:.0f}s) ...")
    gcc_start = time.perf_counter()
    gcc_timed_out = False
    gcc_returncode: Optional[int] = None
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=args.compile_timeout)
        gcc_returncode = proc.returncode
        if proc.returncode != 0:
            print(f"  [gcc] FAILED:\n{proc.stderr[-2000:]}")
    except subprocess.TimeoutExpired:
        gcc_timed_out = True
        print("  [gcc] TIMEOUT")
    gcc_wall = time.perf_counter() - gcc_start
    phases["gcc"] = {
        "wall_seconds": gcc_wall,
        "returncode": gcc_returncode,
        "timed_out": gcc_timed_out,
        "command": command,
    }
    print(f"  [gcc] {gcc_wall:.1f}s")

    return {
        "matrix_name": matrix_name,
        "set": set_name,
        "operation": operation.value,
        "rows": rows,
        "cols": cols,
        "nnz": nnz,
        "load_seconds": load_seconds,
        "params": {
            "band_min_density": args.band_min_density,
            "block_min_density": args.block_min_density,
            "block_min_area": args.block_min_area,
            "block_gamma": args.block_gamma,
            "block_timeout_seconds": args.block_timeout,
            "block_threads": args.block_threads,
            "bench": args.bench,
        },
        "kernels": {
            "vdia": args.vdia_kernel,
            "vbr": args.vbr_kernel,
            "csr": args.csr_kernel,
        },
        "phases": phases,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _load_results(path: pathlib.Path) -> list:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_results(path: pathlib.Path, results: list) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure SABLE inspection/codegen/compile times on the paper matrix sets",
        epilog=(
            "Examples:\n"
            "  %(prog)s                                # all three sets, spmv\n"
            "  %(prog)s --sets vdia_vbr_csr            # one set\n"
            "  %(prog)s --sets vbr_csr --matrices eris1176 --block-timeout 600\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sets", type=str, default="vbr_csr,vdia_csr,vdia_vbr_csr",
                        help="Comma-separated sets: vbr_csr, vdia_csr, vdia_vbr_csr (default: all)")
    parser.add_argument("--matrices", nargs="*", default=None,
                        help="Restrict to these matrices (must belong to the selected sets)")
    parser.add_argument("--operation", type=str, default="spmv", choices=("spmv", "spmm"))
    parser.add_argument("--band-min-density", type=float, default=0.75,
                        help="Band extractor min density (default: 0.75, as in the paper)")
    parser.add_argument("--block-min-density", type=float, default=0.5)
    parser.add_argument("--block-min-area", type=int, default=2500)
    parser.add_argument("--block-gamma", type=float, default=1.5)
    parser.add_argument("--block-timeout", type=float, default=4.0 * 60.0 * 60.0,
                        help="Block partitioner time budget in seconds (default: 4h)")
    parser.add_argument("--block-threads", type=int, default=20)
    parser.add_argument("--vdia-kernel", type=str, default=VDIAKernel.NAIVE.value,
                        choices=[k.value for k in VDIAKernel])
    parser.add_argument("--vbr-kernel", type=str, default=VBRKernel.NAIVE.value,
                        choices=[k.value for k in VBRKernel])
    parser.add_argument("--csr-kernel", type=str, default=CSRKernel.NAIVE.value,
                        choices=[k.value for k in CSRKernel])
    parser.add_argument("--bench", type=int, default=5,
                        help="Benchmark iteration count baked into the generated binary (not run)")
    parser.add_argument("--compile-timeout", type=float, default=COMPILE_TIMEOUT,
                        help="gcc timeout in seconds (default: 4h)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifact-dir", type=str, default=str(FILEPATH / "Generated_Inspection_C"),
                        help="Where generated C/data/binaries are written (default: Generated_Inspection_C)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run matrices that already have a successful entry")
    args = parser.parse_args()

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    for set_name in sets:
        if set_name not in MATRIX_SETS:
            parser.error(f"Unknown set '{set_name}'. Valid: {sorted(MATRIX_SETS)}")

    operation = Operation(args.operation)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for set_name in sets:
        matrices = MATRIX_SETS[set_name]
        if args.matrices:
            requested = set(args.matrices)
            unknown = requested - set(matrices)
            if unknown:
                print(f"[{set_name}] Skipping matrices not in this set: {sorted(unknown)}")
            matrices = [m for m in matrices if m in requested]

        output_file = output_dir / f"inspection_{set_name}_{operation.value}.json"
        results = _load_results(output_file)
        done = {entry["matrix_name"] for entry in results if "error" not in entry}

        print(f"\n########## Set {set_name}: {len(matrices)} matrices "
              f"({len(done & set(matrices))} already done) -> {output_file}")

        for matrix_name in matrices:
            if matrix_name in done and not args.force:
                print(f"[{set_name}] {matrix_name}: already recorded, skipping (use --force to redo)")
                continue
            try:
                entry = run_matrix(matrix_name, set_name, operation, args)
                if entry is None:
                    entry = {"matrix_name": matrix_name, "set": set_name,
                             "operation": operation.value, "error": "download failed"}
            except Exception as exc:
                traceback.print_exc()
                entry = {"matrix_name": matrix_name, "set": set_name,
                         "operation": operation.value, "error": str(exc)}

            results = [r for r in results if r["matrix_name"] != matrix_name]
            results.append(entry)
            _save_results(output_file, results)
            print(f"[{set_name}] {matrix_name}: recorded -> {output_file}")

    print("\nAll requested sets processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
