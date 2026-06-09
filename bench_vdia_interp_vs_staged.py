#!/usr/bin/env python3
"""Interpreted vs staged (out-of-lined) vs staged (inlined) VDIA+VBR+CSR.

For each density-0.075 matrix and operation (SpMV, SpMM), three implementations
run the SAME decomposition on the same all-ones RHS, pinned to the same core:
VDIA = MKL DIA for SpMV but NAIVE for SpMM (MKL ddiamm loses badly for SpMM
because of the row-major<->column-major round-trip), VBR = mixed (per-block
MKL/naive), CSR = chosen backend.

  * interp        : interpreted/interp_vdia_vbr_csr_{spmv,spmm}.c -- walks the
                    indirection arrays at run time; CSR backend via argv.
  * staged (OOL)  : default SABLE codegen -- naive VBR blocks and (SpMV) MKL DIA
                    segments go through parameterized out-of-line helpers.
  * staged inline : every VBR block / VDIA segment emitted inline with literal
                    constants (constant-specialized).

Each binary times the three formats over BENCH iterations and prints
  Dispatch 1 = VDIA, Dispatch 2 = VBR, Dispatch 3 = CSR
We drop the first/last TRIM samples and mean the rest (SpMV 100, SpMM 30; trim 5).

Apples-to-apples check: the result vector of all three variants is compared
against a NumPy reference (A @ ones); the CSV records Values_Match (1/0) and a
warning is printed on any mismatch.

Output: one CSV per (operation x CSR backend) in <output-dir>/, columns:
  Matrix,
  Interp_{VDIA,VBR,CSR,Total}, StagedOOL_{...}, StagedInline_{...},
  Speedup_Interp_{...}   (= Interp / StagedOOL),
  Speedup_Inline_{...}   (= StagedInline / StagedOOL),
  Values_Match

Usage (run yourself, under tmux -- downloads + staged compiles are slow):
  python3 bench_vdia_interp_vs_staged.py
  python3 bench_vdia_interp_vs_staged.py --csr-backends naive --operation spmv --limit 3
  python3 bench_vdia_interp_vs_staged.py heart1 nemeth21
"""

import argparse
import csv
import json
import math
import os
import pathlib
import re
import subprocess
import sys

import numpy
from scipy.io import mmread
from scipy.sparse import csc_matrix

FILEPATH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(FILEPATH / "find-submatrices"))

import bench_suitesparse as bs
from interpreted.make_data import write_array
from sable import Matrix, Operation, Plan
from sable.build_config import CFLAGS, MKL_AVAILABLE, MKL_FLAGS
from sable.extractors import BandExtractorSkip, BlockDetectorSkip, CSRConvertor
from sable.kernels import (
    MixedVBRSpmm,
    MixedVBRSpmv,
    MKLCSRSpmm,
    MKLCSRSpmv,
    MKLDIASpmv,
    NaiveCSRSpmm,
    NaiveCSRSpmv,
    NaiveVDIASpmm,
)
from sable.kernels.vbr import (
    _emit_spmm_mkl_block,
    _emit_spmm_naive_block,
    _emit_spmv_mkl_block,
    _emit_spmv_naive_block,
    _should_use_mkl_for_block,
    _vbr_blocks,
)
from sable.kernels.vdia import _emit_mkl_idiag_decl, _mkl_idiag_name
from sable.tensor import DenseInput, DenseLayout
from utils.fileio import parse_yaml_bands, parse_yaml_blocks, write_dense_matrix, write_dense_vector
from utils.utils import set_ulimit

INTERP_DIR = FILEPATH / "interpreted"
RES_DIR = bs.RESULTS_DIR
BANDS_DIR = FILEPATH / "find-submatrices" / "results_bands"
SPMM_NRHS = bs.SPMM_NRHS

BENCH = {"spmv": 100, "spmm": 30}   # must match SABLE_BENCH in the interp C sources
TRIM = 5
DISPATCH_RE = re.compile(r"Dispatch (\d+): (.+)")
FORMAT_OF = {1: "vdia", 2: "vbr", 3: "csr"}
VALUE_RTOL, VALUE_ATOL = 1e-6, 1e-3


# ---------------------------------------------------------------------------
# Inlined (constant-specialized) staged kernels
# ---------------------------------------------------------------------------


class _InlineMixedVBRSpmv(MixedVBRSpmv):
    def emit_timed_calls(self, fmt, y, x, rhs):
        calls = []
        for r0, r1, c0, c1, off in _vbr_blocks(fmt):
            if _should_use_mkl_for_block(r1 - r0, c1 - c0):
                calls.append(_emit_spmv_mkl_block(fmt, y, x, r0, r1, c0, c1, off))
            else:
                calls.append(_emit_spmv_naive_block(fmt, y, x, r0, r1, c0, c1, off))
        return calls

    def emit_call(self, fmt, y, x, rhs):
        return self.emit_timed_calls(fmt, y, x, rhs)


class _InlineMixedVBRSpmm(MixedVBRSpmm):
    def emit_timed_calls(self, fmt, y, x, rhs):
        calls = []
        for r0, r1, c0, c1, off in _vbr_blocks(fmt):
            if _should_use_mkl_for_block(r1 - r0, c1 - c0):
                calls.append(_emit_spmm_mkl_block(fmt, y, x, rhs, r0, r1, c0, c1, off))
            else:
                calls.append(_emit_spmm_naive_block(fmt, y, x, rhs, r0, r1, c0, c1, off))
        return calls

    def emit_call(self, fmt, y, x, rhs):
        return self.emit_timed_calls(fmt, y, x, rhs)


def _inline_mkl_dia_spmv_segment(fmt, y, x, seg):
    row0 = fmt.seg_row_start[seg]
    nrows = fmt.seg_nrows[seg]
    ndiags = fmt.seg_ndiags[seg]
    idiag_off = fmt.seg_idiag_ptr[seg]
    val_off = fmt.seg_val_ptr[seg]
    idiag = _mkl_idiag_name(fmt)
    return f"""\
{{
MKL_INT mkl_m = {nrows};
MKL_INT mkl_k = {fmt.ncols};
MKL_INT mkl_lval = {nrows};
MKL_INT mkl_ndiag = {ndiags};
double mkl_alpha = 1.0;
double mkl_beta = 1.0;
char mkl_transa = 'N';
char mkl_matdescra[6] = {{'G', ' ', ' ', 'C', ' ', ' '}};
mkl_ddiamv(&mkl_transa, &mkl_m, &mkl_k, &mkl_alpha, mkl_matdescra,
           &{fmt.val}[{val_off}], &mkl_lval, (MKL_INT *)&{idiag}[{idiag_off}], &mkl_ndiag,
           &{x}[0], &mkl_beta, &{y}[{row0}]);
}}
"""


class _InlineMKLDIASpmv(MKLDIASpmv):
    def emit_helpers(self, fmt, rhs):
        return _emit_mkl_idiag_decl(fmt)

    def emit_timed_calls(self, fmt, y, x, rhs):
        return [_inline_mkl_dia_spmv_segment(fmt, y, x, s) for s in range(fmt.nsegments)]

    def emit_call(self, fmt, y, x, rhs):
        return self.emit_timed_calls(fmt, y, x, rhs)


# Inline naive VDIA SpMM: each segment's naive diagonal loop emitted inline with
# literal bounds (vs the default NaiveVDIASpmm, which uses one out-of-line helper).
def _inline_naive_vdia_spmm_segment(fmt, y, x, seg, nrhs):
    row0 = fmt.seg_row_start[seg]
    nrows = fmt.seg_nrows[seg]
    ndiags = fmt.seg_ndiags[seg]
    idiag_off = fmt.seg_idiag_ptr[seg]
    val_off = fmt.seg_val_ptr[seg]
    return f"""\
for (int row = 0; row < {nrows}; row++) {{
    for (int d = 0; d < {ndiags}; d++) {{
        int col = {row0} + row + {fmt.idiag}[{idiag_off} + d];
        if (0 <= col && col < {fmt.ncols}) {{
            double a = {fmt.val}[{val_off} + d * {nrows} + row];
            for (int k = 0; k < {nrhs}; k++) {{
                {y}[({row0} + row) * {nrhs} + k] += a * {x}[col * {nrhs} + k];
            }}
        }}
    }}
}}
"""


class _InlineNaiveVDIASpmm(NaiveVDIASpmm):
    def emit_timed_calls(self, fmt, y, x, rhs):
        nrhs = rhs.shape[1]
        return [_inline_naive_vdia_spmm_segment(fmt, y, x, s, nrhs) for s in range(fmt.nsegments)]

    def emit_call(self, fmt, y, x, rhs):
        return self.emit_timed_calls(fmt, y, x, rhs)


# OOL = shared parameterized helpers; inline = per-segment/block constant-specialized.
# Per-op VDIA dispatch: SpMV = MKL DIA, SpMM = naive (MKL ddiamm loses for SpMM).
# Alias the kernels whose default SABLE lowering already matches the strategy.
_OOLMixedVBRSpmv = MixedVBRSpmv
_OOLMixedVBRSpmm = MixedVBRSpmm
_OOLMKLDIASpmv = MKLDIASpmv          # default MKL DIA SpMV is out-of-line
_OOLNaiveVDIASpmm = NaiveVDIASpmm    # default naive VDIA SpMM is out-of-line

VBR_KERNEL = {"spmv": _OOLMixedVBRSpmv, "spmm": _OOLMixedVBRSpmm}
INLINE_VBR_KERNEL = {"spmv": _InlineMixedVBRSpmv, "spmm": _InlineMixedVBRSpmm}
VDIA_KERNEL = {"spmv": _OOLMKLDIASpmv, "spmm": _OOLNaiveVDIASpmm}
INLINE_VDIA_KERNEL = {"spmv": _InlineMKLDIASpmv, "spmm": _InlineNaiveVDIASpmm}
CSR_KERNEL = {
    "spmv": {"naive": NaiveCSRSpmv, "mkl": MKLCSRSpmv},
    "spmm": {"naive": NaiveCSRSpmm, "mkl": MKLCSRSpmm},
}

CSV_COLUMNS = (
    ["Matrix"]
    + [f"Interp_{c}" for c in ("VDIA", "VBR", "CSR", "Total")]
    + [f"StagedOOL_{c}" for c in ("VDIA", "VBR", "CSR", "Total")]
    + [f"StagedInline_{c}" for c in ("VDIA", "VBR", "CSR", "Total")]
    + [f"Speedup_Interp_{c}" for c in ("VDIA", "VBR", "CSR", "Total")]
    + [f"Speedup_Inline_{c}" for c in ("VDIA", "VBR", "CSR", "Total")]
    + [f"Speedup_InterpVsInline_{c}" for c in ("VDIA", "VBR", "CSR", "Total")]
    + ["Values_Match"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _d075_matrices_with_bands():
    import glob

    names = set()
    for path in glob.glob(str(FILEPATH / "results" / "*_d075.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if isinstance(data, list):
            names.update(e.get("matrix_name") for e in data if e.get("matrix_name"))
    return sorted(
        n for n in names
        if (RES_DIR / f"{n}.yaml").exists() and (BANDS_DIR / f"{n}.yaml").exists()
    )


def _trimmed_mean(vals, bench):
    assert len(vals) == bench, f"expected {bench} samples, got {len(vals)}"
    kept = vals[TRIM:len(vals) - TRIM]
    return sum(kept) / len(kept)


def _parse_dispatches(output):
    out = {}
    for line in output.splitlines():
        m = DISPATCH_RE.match(line)
        if m:
            out[int(m.group(1))] = [float(v) for v in m.group(2).rstrip(",").split(",") if v]
    return out


def _means(output, bench):
    d = _parse_dispatches(output)
    m = {FORMAT_OF[i]: _trimmed_mean(d[i], bench) for i in (1, 2, 3)}
    m["overall"] = m["vdia"] + m["vbr"] + m["csr"]
    return m


def _parse_y(output):
    ys = []
    for line in output.splitlines():
        s = line.strip()
        if not s or "Dispatch" in line or "Intel" in line:
            continue
        try:
            ys.append(float(s))
        except ValueError:
            continue
    return numpy.array(ys)


def _set_rhs(plan, A, op):
    if op == Operation.SPMV:
        write_dense_vector(1.0, A.shape[1])
        path = bs._generated_vector_path(A.shape[1])
        plan.rhs(DenseInput.vector(path, A.shape[1]))
    else:
        write_dense_matrix(1.0, A.shape[1], SPMM_NRHS)
        path = bs._generated_matrix_path(A.shape[1], SPMM_NRHS)
        plan.rhs(DenseInput.matrix(path, shape=(A.shape[1], SPMM_NRHS), layout=DenseLayout.ROW_MAJOR))
    return path


def _build_staged(name, A, bands, blocks, op, csr_backend, variant, outdir):
    opv = op.value
    if variant == "inline":
        kd, kb = INLINE_VDIA_KERNEL[opv](), INLINE_VBR_KERNEL[opv]()
    else:
        kd, kb = VDIA_KERNEL[opv](), VBR_KERNEL[opv]()
    kc = CSR_KERNEL[opv][csr_backend]()
    plan = Plan(Matrix(A, name=name), artifact_dir=str(outdir))
    rhs_path = _set_rhs(plan, A, op)
    plan.dispatch(plan.extract(BandExtractorSkip(bands)), kd)
    plan.dispatch(plan.extract(BlockDetectorSkip(blocks)), kb)
    plan.dispatch(plan.extract(CSRConvertor()), kc)
    executor = plan.compile(filename=f"{name}_{opv}_{csr_backend}_{variant}", bench=BENCH[opv])
    return executor, rhs_path


def _write_interp_sabledata(path, vdia, vbr, csr):
    with open(path, "w") as f:
        write_array(f, "seg_row_start", vdia.seg_row_start)
        write_array(f, "seg_nrows", vdia.seg_nrows)
        write_array(f, "seg_ndiags", vdia.seg_ndiags)
        write_array(f, "seg_idiag_ptr", vdia.seg_idiag_ptr)
        write_array(f, "seg_val_ptr", vdia.seg_val_ptr)
        write_array(f, "vdia_idiag", vdia.idiag.values)
        write_array(f, "vdia_val", vdia.val.values)
        write_array(f, "rpntr", vbr.rpntr)
        write_array(f, "cpntr", vbr.cpntr)
        write_array(f, "bpntrb", vbr.bpntrb)
        write_array(f, "bindx", vbr.bindx)
        write_array(f, "indx", vbr.indx)
        write_array(f, "vbr_val", vbr.val.values)
        write_array(f, "csr_indptr", csr.indptr.values)
        write_array(f, "csr_indices", csr.indices.values)
        write_array(f, "csr_val", csr.values.values)


def _compile_interp(op, outdir):
    src = INTERP_DIR / ("interp_vdia_vbr_csr_spmv.c" if op == Operation.SPMV else "interp_vdia_vbr_csr_spmm.c")
    binary = outdir / src.stem
    command = ["gcc", str(src), "-o", str(binary)] + list(CFLAGS) + list(MKL_FLAGS)
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=bs.COMPILE_TIMEOUT)
    return binary, command


# Pin all benchmark binaries to a single, fixed core.  Avoid core 0, which
# carries most of the OS interrupt / kernel-thread noise.  Override with
# SABLE_BENCH_CORE.
BENCH_CORE = int(os.environ.get("SABLE_BENCH_CORE", "4"))

# Force single-threaded execution so a pinned core is not oversubscribed by an
# MKL / OpenMP thread pool busy-spinning on the one allowed CPU.  Applied
# identically to the interpreted and staged runs for an apples-to-apples timing.
_THREAD_ENV = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_THREADING_LAYER": "GNU",
}


def _taskset_prefix():
    return ["taskset", "-a", "-c", str(BENCH_CORE)]


def _run(cmd, cwd, env):
    return subprocess.check_output(cmd, cwd=cwd, env=env, preexec_fn=set_ulimit).decode("utf-8")


def _staged_run(executor):
    env = bs._executor_env(executor.compile_command or [], executor.runtime_env)
    env = dict(env) if env is not None else os.environ.copy()
    env.update(_THREAD_ENV)
    return _run(_taskset_prefix() + [executor.binary_path], executor.artifact_dir, env)


def _interp_env(command):
    env = os.environ.copy()
    lib_dirs = [flag[2:] for flag in command if flag.startswith("-L")]
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    env.update(_THREAD_ENV)
    return env


def _reference(A, op):
    if op == Operation.SPMV:
        return A.dot(numpy.ones(A.shape[1]))
    return A.dot(numpy.ones((A.shape[1], SPMM_NRHS))).reshape(-1)


def bench_matrix_op(op, name, A, bands, blocks, backends, outdir):
    """Run all three variants for each backend. Returns (results, values_match).

    results[backend] = {"interp":means, "ool":means, "inline":means}
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bench = BENCH[op.value]
    data_path = outdir / f"{name}.sabledata"
    interp_binary = interp_cmd = None
    results = {}
    y_caps = {}

    for bi, backend in enumerate(backends):
        ex_ool, rhs_path = _build_staged(name, A, bands, blocks, op, backend, "ool", outdir / f"ool_{backend}")
        if bs.compile_frontend_executor(ex_ool) is None:
            raise RuntimeError(f"staged OOL compile failed ({op.value}, csr={backend})")
        out_ool = _staged_run(ex_ool)

        ex_inl, _ = _build_staged(name, A, bands, blocks, op, backend, "inline", outdir / f"inline_{backend}")
        if bs.compile_frontend_executor(ex_inl) is None:
            raise RuntimeError(f"staged inline compile failed ({op.value}, csr={backend})")
        out_inl = _staged_run(ex_inl)

        if interp_binary is None:
            vdia, vbr, csr = (d.fmt for d in ex_ool.plan.dispatches)
            _write_interp_sabledata(str(data_path), vdia, vbr, csr)
            interp_binary, interp_cmd = _compile_interp(op, outdir)
        out_interp = _run(
            _taskset_prefix() + [str(interp_binary), str(data_path), str(rhs_path), backend],
            str(outdir), _interp_env(interp_cmd),
        )

        results[backend] = {
            "ool": _means(out_ool, bench),
            "inline": _means(out_inl, bench),
            "interp": _means(out_interp, bench),
        }
        if bi == 0:
            y_caps["ool"] = _parse_y(out_ool)
            y_caps["inline"] = _parse_y(out_inl)
            y_caps["interp"] = _parse_y(_run(
                _taskset_prefix() + [str(interp_binary), str(data_path), str(rhs_path), backend, "y"],
                str(outdir), _interp_env(interp_cmd),
            ))

    ref = _reference(A, op)
    diffs = {}
    match = True
    for v in ("interp", "ool", "inline"):
        y = y_caps.get(v)
        if y is None or y.shape != ref.shape:
            match = False
            diffs[v] = float("nan")
            continue
        diffs[v] = float(numpy.max(numpy.abs(y - ref)))
        if not numpy.allclose(y, ref, rtol=VALUE_RTOL, atol=VALUE_ATOL):
            match = False
    return results, match, diffs


def _csv_row(name, res, values_match):
    interp, ool, inline = res["interp"], res["ool"], res["inline"]
    row = {"Matrix": name, "Values_Match": 1 if values_match else 0}
    for fmt, col in (("vdia", "VDIA"), ("vbr", "VBR"), ("csr", "CSR"), ("overall", "Total")):
        row[f"Interp_{col}"] = round(interp[fmt], 2)
        row[f"StagedOOL_{col}"] = round(ool[fmt], 2)
        row[f"StagedInline_{col}"] = round(inline[fmt], 2)
        row[f"Speedup_Interp_{col}"] = round(interp[fmt] / ool[fmt], 4) if ool[fmt] > 0 else 0.0
        row[f"Speedup_Inline_{col}"] = round(inline[fmt] / ool[fmt], 4) if ool[fmt] > 0 else 0.0
        row[f"Speedup_InterpVsInline_{col}"] = round(interp[fmt] / inline[fmt], 4) if inline[fmt] > 0 else 0.0
    return row


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="interp vs staged-OOL vs staged-inline VDIA(MKL)+VBR(mixed)+CSR -> CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("matrices", nargs="*")
    parser.add_argument("--operation", default="spmv,spmm")
    parser.add_argument("--csr-backends", default="naive,mkl", help="comma list of: naive, mkl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="results/interp_vs_staged")
    args = parser.parse_args()

    if not MKL_AVAILABLE:
        parser.error("MKL not detected; VDIA=MKL, mixed VBR, and MKL CSR all require it.")

    operations = [Operation(o.strip()) for o in args.operation.split(",")]
    backends = [b.strip() for b in args.csr_backends.split(",") if b.strip()]
    matrices = args.matrices or _d075_matrices_with_bands()
    if args.limit is not None:
        matrices = matrices[: args.limit]

    csv_dir = pathlib.Path(args.output_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    codegen_root = pathlib.Path(os.environ.get("SABLE_CODEGEN_DIR") or str(FILEPATH)) / "Generated_vdia_interp_vs_staged"

    print(f"{len(matrices)} matrices x {[o.value for o in operations]} x csr{backends}; "
          f"VDIA=mkl VBR=mixed; bench spmv={BENCH['spmv']} spmm={BENCH['spmm']} trim {TRIM}")

    rows = {(op.value, b): [] for op in operations for b in backends}

    for name in matrices:
        if not (RES_DIR / f"{name}.yaml").exists() or not (BANDS_DIR / f"{name}.yaml").exists():
            print(f"\n{name}: missing VBR or band yaml, skipping")
            continue
        print(f"\nProcessing {name}...")
        dl = bs.download_matrix_from_suitesparse(name)
        if dl is None:
            print("  download failed, skipping")
            continue
        mtx_path, info, tar_path, subdir = dl
        try:
            A = csc_matrix(mmread(mtx_path), copy=False)
            blocks = parse_yaml_blocks(str(RES_DIR / f"{name}.yaml"))
            bands = parse_yaml_bands(str(BANDS_DIR / f"{name}.yaml"))
            print(f"  {A.shape[0]}x{A.shape[1]}, nnz={A.nnz}, {len(blocks)} blocks, {len(bands)} bands")
            for op in operations:
                res_by_backend, match, diffs = bench_matrix_op(
                    op, name, A, bands, blocks, backends, codegen_root / f"{op.value}_{name}"
                )
                if not match:
                    print(f"  !! VALUES MISMATCH ({op.value}): max|y-ref| "
                          f"interp={diffs.get('interp')}, ool={diffs.get('ool')}, inline={diffs.get('inline')}")
                for backend in backends:
                    row = _csv_row(name, res_by_backend[backend], match)
                    rows[(op.value, backend)].append(row)
                    _write_csv(csv_dir / f"{op.value}_vbr-mixed_vdia-mkl_csr-{backend}.csv", rows[(op.value, backend)])
                    print(f"  [{op.value} csr={backend}] interp/OOL={row['Speedup_Interp_Total']:.2f}x  "
                          f"inline/OOL={row['Speedup_Inline_Total']:.2f}x  values_match={row['Values_Match']}")
        except Exception as exc:
            print(f"  Error processing {name}: {exc}")
            import traceback
            traceback.print_exc()
        # Downloaded matrices are intentionally kept (not cleaned up) so reruns
        # are fast and the inputs stay available for inspection.

    print("\n" + "=" * 72)
    print("Summary: geomean ratio (interp/OOL and inline/OOL); >1 = slower than staged-OOL")
    print("=" * 72)
    for (opv, backend), rlist in rows.items():
        if not rlist:
            continue
        nmatch = sum(r["Values_Match"] for r in rlist)
        print(f"  {opv} csr={backend} ({len(rlist)} matrices, {nmatch} value-matched):")
        for label, prefix in (("interp/OOL", "Speedup_Interp_"), ("inline/OOL", "Speedup_Inline_")):
            cells = []
            for col in ("VDIA", "VBR", "CSR", "Total"):
                vals = [r[f"{prefix}{col}"] for r in rlist if r[f"{prefix}{col}"] > 0]
                geo = math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
                cells.append(f"{col}={geo:.2f}")
            print(f"      {label:<11} " + "  ".join(cells))
    print(f"\nCSVs written to {csv_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
