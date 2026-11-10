#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import subprocess
import time
import traceback
from typing import Any, Dict, Optional

from scipy.io import mmread
from scipy.sparse import csr_matrix

from bench_suitesparse import (
    COMPILE_TIMEOUT,
    SPMM_NRHS,
    format_token,
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
from sable.extractors import BlockDetector, BlockDetectorSkip, CSRConvertor, SpMMBandExtractor, SpMVBandExtractor
from sable.tensor import DenseInput, DenseLayout
from utils.fileio import parse_yaml_blocks, write_dense_matrix, write_dense_vector

FILEPATH = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = FILEPATH / "results" / "inspection"

# The 55 matrices of the VBR+CSR evaluation (paper Sec. Evaluation; the
# matrices of results/spmv_vbr050-blockmixed_csr-mkl.json that the paper reports).
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

# The matrices each operation's band extractor selects bands for (the stored
# bands find-submatrices/scan_bands.py writes, find-submatrices/results_vdia<op>):
# SpMVBandExtractor 43, SpMMBandExtractor 49. Both VDIA sets use the
# operation's list.
VDIA_SPMV = [
    "Baumann", "G2_circuit", "G3_circuit", "Hardesty1", "Lin",
    "Spielman_k100", "Spielman_k200", "TSC_OPF_1047", "TSOPF_RS_b2052_c1",
    "TSOPF_RS_b678_c1", "TSOPF_RS_b678_c2", "apache1", "apache2", "atmosmodd",
    "atmosmodj", "atmosmodl", "atmosmodm", "bloweybl", "bodyy4", "bodyy5",
    "bodyy6", "chem_master1", "cont-201", "cont-300", "denormal", "dixmaanl",
    "ecology1", "ecology2", "exdata_1", "jnlbrng1", "mc2depi", "minsurfo",
    "mip1", "obstclae", "rajat09", "rajat10", "shyy161", "t2em", "tmt_sym",
    "tmt_unsym", "torsion1", "wang3", "wang4"
]
VDIA_SPMM = [
    "G2_circuit", "G3_circuit", "Spielman_k100", "Spielman_k200",
    "TSC_OPF_1047", "TSC_OPF_300", "TSOPF_RS_b162_c1", "TSOPF_RS_b162_c3",
    "TSOPF_RS_b162_c4", "TSOPF_RS_b2052_c1", "TSOPF_RS_b300_c1",
    "TSOPF_RS_b300_c2", "TSOPF_RS_b300_c3", "TSOPF_RS_b678_c1",
    "TSOPF_RS_b678_c2", "chem_master1", "ecology1", "ecology2", "exdata_1",
    "jnlbrng1", "mip1", "nemeth01", "nemeth02", "nemeth03", "nemeth04",
    "nemeth05", "nemeth06", "nemeth07", "nemeth08", "nemeth09", "nemeth10",
    "nemeth11", "nemeth12", "nemeth13", "nemeth14", "nemeth15", "nemeth16",
    "nemeth17", "nemeth18", "nemeth19", "nemeth20", "nemeth21", "nemeth22",
    "nemeth23", "nemeth24", "nemeth25", "nemeth26", "piston", "spmsrtls"
]
VDIA_MATRICES = {Operation.SPMV: VDIA_SPMV, Operation.SPMM: VDIA_SPMM}

# The blocks-first evaluation (three_format.py): its operation--matrix pairs,
# and the stored blocks it replays before searching the residual for bands.
THREE_FORMAT_EXTRACTIONS = FILEPATH / "find-submatrices" / "three_format"
STORED_BLOCKS = FILEPATH / "find-submatrices" / "results"

# The formats each set extracts, in extraction order; they name the output file.
# vbr_vdia_csr replays the stored blocks instead of running the partitioner.
SET_FORMATS = {
    "vbr_csr": ("vbr",),
    "vdia_csr": ("vdia",),
    "vdia_vbr_csr": ("vdia", "vbr"),
    "vbr_vdia_csr": ("vbr", "vdia"),
}


def matrix_set(set_name: str, operation: Operation) -> list[str]:
    if set_name == "vbr_csr":
        return VBR_CSR_55
    if set_name == "vbr_vdia_csr":
        composed = THREE_FORMAT_EXTRACTIONS / f"results_vbr050_vdia{operation.value}"
        return sorted(p.stem for p in composed.glob("*.yaml"))
    return VDIA_MATRICES[operation]


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def _make_kernels(operation: Operation, args) -> Dict[str, Any]:
    if operation == Operation.SPMV:
        return {
            "vdia": _vdia_spmv_kernel(VDIAKernel.NAIVE),
            "vbr": _vbr_spmv_kernel(VBRKernel(args.vbr_kernel)),
            "csr": _csr_spmv_kernel(CSRKernel(args.csr_kernel)),
        }
    return {
        "vdia": _vdia_spmm_kernel(VDIAKernel.NAIVE),
        "vbr": _vbr_spmm_kernel(VBRKernel(args.vbr_kernel)),
        "csr": _csr_spmm_kernel(CSRKernel(args.csr_kernel)),
    }


def run_vbr_phase(matrix_name: str, set_name: str, plan: Plan, kernels: Dict[str, Any],
                  phases: Dict[str, Any], args) -> None:
    nnz = plan.residual.nnz
    if set_name == "vbr_vdia_csr":
        blocks = parse_yaml_blocks(str(STORED_BLOCKS / f"{matrix_name}.yaml"))
        print(f"  [VBR] replaying {len(blocks)} stored block(s) ...")
        vbr_fmt, wall = _timed(plan.extract, BlockDetectorSkip(blocks))
        phases["vbr"] = {
            "wall_seconds": wall,
            "claimed_nnz": nnz - plan.residual.nnz,
            "replayed": True,
            "block_count": len(blocks),
        }
        plan.dispatch(vbr_fmt, kernels["vbr"])
        print(f"  [VBR] claimed {phases['vbr']['claimed_nnz']}/{nnz} nnz (wall {wall:.1f}s)")
        return
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
    vbr_fmt, wall = _timed(plan.extract, detector)
    claimed = nnz - plan.residual.nnz
    phases["vbr"] = {
        "wall_seconds": wall,
        "claimed_nnz": claimed,
        **(detector.last_timing or {}),
    }
    plan.dispatch(vbr_fmt, kernels["vbr"])
    timeout_note = " [TIMEOUT]" if phases["vbr"].get("timeout") else ""
    print(
        f"  [VBR] {len(phases['vbr'].get('blocks', []))} block(s), "
        f"claimed {claimed}/{nnz} nnz, partitioner "
        f"{phases['vbr'].get('partitioner_seconds', 0):.1f}s (wall {wall:.1f}s){timeout_note}"
    )


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
        plan.rhs(DenseInput.vector(write_dense_vector(1.0, cols), cols))
    else:
        plan.rhs(
            DenseInput.matrix(
                write_dense_matrix(1.0, cols, SPMM_NRHS),
                shape=(cols, SPMM_NRHS),
                layout=DenseLayout.ROW_MAJOR,
            )
        )

    kernels = _make_kernels(operation, args)
    phases: Dict[str, Any] = {}

    for kind in SET_FORMATS[set_name]:
        if kind == "vbr":
            run_vbr_phase(matrix_name, set_name, plan, kernels, phases, args)
            continue
        extractor = SpMVBandExtractor() if operation == Operation.SPMV else SpMMBandExtractor()
        print(f"  [VDIA] band search ({type(extractor).__name__}) ...")
        nnz_before = plan.residual.nnz
        vdia_fmt, wall = _timed(plan.extract, extractor)
        claimed = nnz_before - plan.residual.nnz
        phases["vdia"] = {
            "wall_seconds": wall,
            "claimed_nnz": claimed,
            **(extractor.last_timing or {}),
        }
        plan.dispatch(vdia_fmt, kernels["vdia"])
        print(
            f"  [VDIA] {len(phases['vdia'].get('bands', []))} band(s), "
            f"claimed {claimed}/{nnz} nnz, search "
            f"{phases['vdia'].get('search_seconds', 0):.1f}s (wall {wall:.1f}s)"
        )

    convertor = CSRConvertor()
    nnz_before = plan.residual.nnz
    csr_fmt, wall = _timed(plan.extract, convertor)
    phases["csr"] = {
        "wall_seconds": wall,
        "claimed_nnz": nnz_before,
        "convert_seconds": convertor.last_timing["convert_seconds"],
    }
    plan.dispatch(csr_fmt, kernels["csr"])
    print(f"  [CSR] claimed {nnz_before}/{nnz} nnz, convert {phases['csr']['convert_seconds']:.2f}s")

    print("  [CodeGen] emitting C source and data ...")
    executor, codegen_wall = _timed(plan.compile, filename=matrix_name, bench=args.bench)
    phases["codegen"] = {
        "wall_seconds": codegen_wall,
        # Emitting the C plus writing the staged data, as the docstring says.
        "codegen_time_ms": executor.codegen_time_ms + executor.staged_data_time_ms,
    }
    print(f"  [CodeGen] {codegen_wall:.2f}s -> {executor.c_path}")

    output_path = os.path.abspath(os.path.join(artifact_dir, matrix_name))
    command = build_compile_command_for_plan(plan, executor.c_path, output_path)
    print(f"  [gcc] compiling (timeout={args.compile_timeout:.0f}s) ...")
    gcc_start = time.perf_counter()
    # A failed compile keeps its phases but is recorded as an error, so the
    # next run retries it rather than reading its gcc time as a measurement.
    error = None
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=args.compile_timeout)
        if proc.returncode != 0:
            print(f"  [gcc] FAILED:\n{proc.stderr[-2000:]}")
            error = f"gcc failed with return code {proc.returncode}"
    except subprocess.TimeoutExpired:
        print("  [gcc] TIMEOUT")
        error = f"gcc timed out after {args.compile_timeout:.0f}s"
    gcc_wall = time.perf_counter() - gcc_start
    phases["gcc"] = {"wall_seconds": gcc_wall}
    print(f"  [gcc] {gcc_wall:.1f}s")

    # The operation, set and block density are in the file name, and the
    # matrix's shape and nonzeros are in the benchmark result files.
    entry = {
        "matrix_name": matrix_name,
        "load_seconds": load_seconds,
        "kernels": {
            "vdia": VDIAKernel.NAIVE.value,
            "vbr": args.vbr_kernel,
            "csr": args.csr_kernel,
        },
        "phases": phases,
    }
    if error:
        entry["error"] = error
    return entry


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
            "  %(prog)s --sets vbr_vdia_csr --operation spmm\n"
            "  %(prog)s --sets vbr_csr --matrices eris1176 --block-timeout 600\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sets", type=str, default="vbr_csr,vdia_csr,vbr_vdia_csr",
                        help="Comma-separated sets: vbr_csr, vdia_csr, vbr_vdia_csr (blocks first, "
                             "replaying the stored blocks), vdia_vbr_csr (bands first)")
    parser.add_argument("--matrices", nargs="*", default=None,
                        help="Restrict to these matrices (must belong to the selected sets)")
    parser.add_argument("--operation", type=str, default="spmv", choices=("spmv", "spmm"))
    parser.add_argument("--block-min-density", type=float, default=0.5)
    parser.add_argument("--block-min-area", type=int, default=2500)
    parser.add_argument("--block-gamma", type=float, default=1.5)
    parser.add_argument("--block-timeout", type=float, default=4.0 * 60.0 * 60.0,
                        help="Block partitioner time budget in seconds (default: 4h)")
    parser.add_argument("--block-threads", type=int, default=20)
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
        if set_name not in SET_FORMATS:
            parser.error(f"Unknown set '{set_name}'. Valid: {sorted(SET_FORMATS)}")

    operation = Operation(args.operation)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for set_name in sets:
        matrices = matrix_set(set_name, operation)
        if args.matrices:
            requested = set(args.matrices)
            unknown = requested - set(matrices)
            if unknown:
                print(f"[{set_name}] Skipping matrices not in this set: {sorted(unknown)}")
            matrices = [m for m in matrices if m in requested]

        # Bands have no density: their token names the operation, whose extractor found them.
        densities = {"vdia": operation.value, "vbr": args.block_min_density}
        extraction = "_".join(format_token(kind, densities[kind]) for kind in SET_FORMATS[set_name])
        output_file = output_dir / f"inspection_{operation.value}_{extraction}_csr.json"
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
                    entry = {"matrix_name": matrix_name, "error": "download failed"}
            except Exception as exc:
                traceback.print_exc()
                entry = {"matrix_name": matrix_name, "error": str(exc)}

            results = [r for r in results if r["matrix_name"] != matrix_name]
            results.append(entry)
            _save_results(output_file, results)
            print(f"[{set_name}] {matrix_name}: recorded -> {output_file}")

    print("\nAll requested sets processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
