#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import sys
from typing import Any

from scipy.io import mmread
from scipy.sparse import csr_matrix

FILEPATH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(FILEPATH / "find-submatrices"))

import bench_suitesparse as bs
from sable import Matrix, Operation, Plan
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
    SPV8CSRSpmv,
)
from sable.kernels.vbr import _vbr_blocks
from sable.tensor import DenseInput, DenseLayout
from utils.fileio import parse_yaml_bands, parse_yaml_blocks, write_dense_matrix, write_dense_vector

RESULTS = FILEPATH / "results"
MEASUREMENTS = RESULTS / "vdia_overlap_measurements.json"

# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

BLOCKS_DIR = FILEPATH / "find-submatrices" / "results"
BANDS_DIR = FILEPATH / "find-submatrices" / "results_bands_075"
CODEGEN_ROOT = pathlib.Path(os.environ.get("SABLE_CODEGEN_DIR") or str(FILEPATH)) / "Generated_OverlapFix_C"

SPMM_NRHS = bs.SPMM_NRHS
BENCH = {"spmv": 30, "spmm": 10}  # iterations per invocation == number of invocations


AFFECTED_CORRECTED = ["heart1", "heart2", "heart3", "pkustk07", "pkustk08"]

AFFECTED_ABLATION = ["heart1", "heart2", "heart3", "nd3k", "nemeth22", "pkustk07"]

CSR_KERNELS = {
    "spmv": {"naive": NaiveCSRSpmv, "mkl": MKLCSRSpmv, "spv8": SPV8CSRSpmv},
    "spmm": {"naive": NaiveCSRSpmm, "mkl": MKLCSRSpmm},
}


def block_band_overlap(block, band) -> bool:
    """True iff the block rectangle intersects the band's ribbon cells."""
    r0, r1, c0, c1 = block
    d = band["diag_offset"]
    for seg in band["segments"]:
        s0, s1 = seg["rows"]
        lo_bw, up_bw = seg["bandwidth"]
        if max(r0, s0, c0 - d - up_bw) < min(r1, s1, c1 - d + lo_bw):
            return True
    return False


def classify(matrix_name: str) -> dict[str, Any]:
    blocks = parse_yaml_blocks(str(BLOCKS_DIR / f"{matrix_name}.yaml"))
    bands = parse_yaml_bands(str(BANDS_DIR / f"{matrix_name}.yaml"))
    block_overlaps = [any(block_band_overlap(b, bd) for bd in bands) for b in blocks]
    band_overlaps = [any(block_band_overlap(b, bd) for b in blocks) for bd in bands]
    return {
        "blocks": blocks,
        "bands": bands,
        "kept_blocks": [b for b, o in zip(blocks, block_overlaps) if not o],
        "removed_blocks": [b for b, o in zip(blocks, block_overlaps) if o],
        "kept_bands": [bd for bd, o in zip(bands, band_overlaps) if not o],
        "removed_bands": [bd for bd, o in zip(bands, band_overlaps) if o],
    }


def _band_segment_mapping(bands: list[dict]) -> list[int]:
    """VDIA part k (1-based) times segment k-1; map each segment to its band index."""
    mapping = []
    for band_idx, band in enumerate(bands):
        for _ in band["segments"]:
            mapping.append(band_idx)
    return mapping


def _packed_block_mapping(vbr_fmt, yaml_blocks) -> list[dict[str, Any]]:
    """Map each packed VBR part (in emission order) to the YAML blocks containing it."""
    mapping = []
    for r0, r1, c0, c1, _ in _vbr_blocks(vbr_fmt):
        containing = [
            i
            for i, (br0, br1, bc0, bc1) in enumerate(yaml_blocks)
            if br0 <= r0 and r1 <= br1 and bc0 <= c0 and c1 <= bc1
        ]
        mapping.append({"cell": [int(r0), int(r1), int(c0), int(c1)], "yaml_blocks": containing})
    return mapping


def build_and_measure(matrix_name, A, variant, regions, operation, backend_name):
    op = operation.value
    matrix = Matrix(A, name=matrix_name)
    artifact_dir = str(CODEGEN_ROOT / f"{variant}_{op}_{backend_name}")
    os.makedirs(artifact_dir, exist_ok=True)

    plan = Plan(matrix, artifact_dir=artifact_dir)
    if operation == Operation.SPMV:
        write_dense_vector(1.0, matrix.ncols)
        plan.rhs(DenseInput.vector(bs._generated_vector_path(matrix.ncols), matrix.ncols))
        vdia_kernel, vbr_kernel = MKLDIASpmv(), MixedVBRSpmv()
    else:
        write_dense_matrix(1.0, matrix.ncols, SPMM_NRHS)
        plan.rhs(
            DenseInput.matrix(
                bs._generated_matrix_path(matrix.ncols, SPMM_NRHS),
                shape=(matrix.ncols, SPMM_NRHS),
                layout=DenseLayout.ROW_MAJOR,
            )
        )
        vdia_kernel, vbr_kernel = NaiveVDIASpmm(), MixedVBRSpmm()
    csr_kernel = CSR_KERNELS[op][backend_name]()

    bands_v = regions["bands_v"]
    blocks_v = regions["blocks_v"]
    if variant == "abl":
        vbr = plan.extract(BlockDetectorSkip(blocks_v))
        vdia = plan.extract(BandExtractorSkip(bands_v))
        plan.dispatch(vbr, vbr_kernel)
        plan.dispatch(vdia, vdia_kernel)
        dispatch_roles = {"vbr": 1, "vdia": 2, "csr": 3}
    else:
        vdia = plan.extract(BandExtractorSkip(bands_v))
        vbr = plan.extract(BlockDetectorSkip(blocks_v))
        plan.dispatch(vdia, vdia_kernel)
        plan.dispatch(vbr, vbr_kernel)
        dispatch_roles = {"vdia": 1, "vbr": 2, "csr": 3}
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, csr_kernel)

    residual_nnz = csr.nnz
    part_block_mapping = _packed_block_mapping(vbr, blocks_v)
    segment_band_mapping = _band_segment_mapping(bands_v)

    executor = plan.compile(filename=f"{matrix_name}_{variant}", bench=BENCH[op])
    dispatch_times, dispatch_part_times, compile_time_ns = bs.eval_frontend_executor_timings(
        executor, BENCH[op], threads=1
    )
    if not dispatch_times:
        return None

    return {
        "matrix_name": matrix_name,
        "variant": variant,
        "operation": op,
        "backend": backend_name,
        "dispatch_roles": dispatch_roles,
        "dispatch_times_ns": {str(k): v for k, v in sorted(dispatch_times.items())},
        "dispatch_part_times_ns": dict(sorted(dispatch_part_times.items())),
        "part_block_mapping": part_block_mapping,
        "segment_band_mapping": segment_band_mapping,
        "residual_nnz": int(residual_nnz),
        "compile_time_s": compile_time_ns / 1e9,
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

SWEEP_FILES = {
    ("spmv", "naive"): "sable_spmv_mkl_naive.json",
    ("spmv", "mkl"): "sable_spmv_mkl_mkl.json",
    ("spmv", "spv8"): "sable_spmv_mkl_spv8.json",
    ("spmm", "naive"): "sable_spmm_mkl_naive.json",
    ("spmm", "mkl"): "sable_spmm_mkl_mkl.json",
}

BACKENDS = {"spmv": ["naive", "mkl", "spv8"], "spmm": ["naive", "mkl"]}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def match_recorded_blocks(matrix_name, op, backend, yaml_blocks, block_full_nnz):
    """Greedily match recorded sweep blocks to YAML blocks by dimensions.

    The old sweep counted per-block nnz slightly differently, so nnz is only a
    tie-breaker (nearest count) among blocks with identical dimensions.
    Returns recorded time_ns per YAML block index (None if unmatched).
    """
    sweep = load_json(RESULTS / SWEEP_FILES[(op, backend)])
    entry = next(e for e in sweep if e["matrix_name"] == matrix_name)
    recorded = list(entry["timing"]["1 thread"]["individual_dense_block_timings"].values())
    times = [None] * len(yaml_blocks)
    used = [False] * len(recorded)
    for i, (r0, r1, c0, c1) in enumerate(yaml_blocks):
        dims = (r1 - r0, c1 - c0)
        candidates = [
            j for j, rec in enumerate(recorded)
            if not used[j] and (rec["rows"], rec["cols"]) == dims
        ]
        if not candidates:
            continue
        j = min(candidates, key=lambda j: abs(recorded[j]["nnz"] - block_full_nnz[i]))
        times[i] = recorded[j]["time_ns"]
        used[j] = True
    return times


def fresh_csr_us(raw, matrix, variant, op, backend):
    entry = raw[f"{matrix}|{variant}|{op}|{backend}"]
    csr_id = str(entry["dispatch_roles"]["csr"])
    return entry["dispatch_times_ns"][csr_id] / 1000.0


def buggy_removed_band_us(seg_times_all, matrix, op, removed_segment_indices, n_segments):
    """Sum the published-era per-segment VDIA times of the removed segments."""
    seg_times = seg_times_all[f"{matrix}|{op}"]
    if len(seg_times) != n_segments:
        raise RuntimeError(
            f"{matrix}/{op}: {len(seg_times)} recorded segments, {n_segments} in the band YAML")
    return sum(seg_times[k] for k in removed_segment_indices) / 1000.0, seg_times


def recompute_best(per_backend):
    best_v = min(per_backend.items(), key=lambda kv: kv[1]["vdia_us"])
    best_b = min(per_backend.items(), key=lambda kv: kv[1]["baseline_us"])
    return {
        "best_vdia_us": best_v[1]["vdia_us"],
        "best_vdia_backend": best_v[0],
        "best_baseline_us": best_b[1]["baseline_us"],
        "best_baseline_backend": best_b[0],
        "speedup": round(best_b[1]["baseline_us"] / best_v[1]["vdia_us"], 4),
    }


def measure(args) -> int:
    variants = [v.strip() for v in args.variants.split(",")]
    operations = [Operation(o.strip()) for o in args.operations.split(",")]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS / "overlap_fix_raw.json"
    results: dict[str, Any] = {}
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)

    matrix_variants: dict[str, list[str]] = {}
    for m in AFFECTED_CORRECTED:
        matrix_variants.setdefault(m, []).extend(["old", "new"])
    for m in AFFECTED_ABLATION:
        entry = matrix_variants.setdefault(m, [])
        if "old" not in entry:
            entry.append("old")
        entry.append("abl")

    matrices = args.matrices or sorted(matrix_variants)

    for matrix_name in matrices:
        wanted = [v for v in matrix_variants.get(matrix_name, []) if v in variants]
        if not wanted:
            continue

        info = classify(matrix_name)
        download = bs.download_matrix_from_suitesparse(matrix_name)
        if download is None:
            print(f"[{matrix_name}] download failed, skipping")
            continue
        mtx_path = download[0]
        print(f"[{matrix_name}] loading {mtx_path}")
        A = csr_matrix(mmread(mtx_path))

        for variant in wanted:
            if variant == "abl":
                regions = {"bands_v": info["kept_bands"], "blocks_v": info["blocks"]}
            elif variant == "new":
                regions = {"bands_v": info["bands"], "blocks_v": info["kept_blocks"]}
            else:
                regions = {"bands_v": info["bands"], "blocks_v": info["blocks"]}

            for operation in operations:
                op = operation.value
                backends = list(CSR_KERNELS[op])
                if args.backends:
                    backends = [b for b in backends if b in args.backends.split(",")]
                for backend_name in backends:
                    key = f"{matrix_name}|{variant}|{op}|{backend_name}"
                    if key in results and not args.force:
                        print(f"  [{key}] already recorded, skipping")
                        continue
                    print(f"  [{key}] measuring ...")
                    entry = build_and_measure(matrix_name, A, variant, regions, operation, backend_name)
                    if entry is None:
                        print(f"  [{key}] FAILED")
                        continue
                    results[key] = entry
                    with open(out_file, "w") as f:
                        json.dump(results, f, indent=1)
                    csr_id = entry["dispatch_roles"]["csr"]
                    csr_ns = entry["dispatch_times_ns"].get(str(csr_id), 0.0)
                    print(f"  [{key}] csr={csr_ns / 1000.0:.1f}us "
                          f"total={sum(entry['dispatch_times_ns'].values()) / 1000.0:.1f}us")

    print(f"\nRaw results in {out_file}")
    return 0


def aggregate(args) -> int:
    meas = load_json(MEASUREMENTS)
    raw = meas["fresh_measurements"]
    nnz_info = meas["block_nnz_analysis"]
    seg_times_all = meas["prefix_vdia_segment_times_ns"]
    report = []

    # The pristine published totals come from the measurements file, never from
    # the files this script overwrites, so the correction is never applied twice.
    published = meas["published_totals"]

    corrected = {"spmv": [], "spmm": []}
    ablation = {"spmv": [], "spmm": []}

    for op in ("spmv", "spmm"):
        for pub in published[op]:
            m = pub["matrix_name"]
            info = classify(m)
            n_blocks = len(info["blocks"])
            n_removed_blocks = len(info["removed_blocks"])
            n_removed_bands = len(info["removed_bands"])

            # ---------------- corrected VDIA+VBR+CSR ----------------
            if info["kept_blocks"]:
                if m not in AFFECTED_CORRECTED:
                    corrected[op].append(pub)
                else:
                    yaml_blocks = info["blocks"]
                    blk_info = nnz_info[m]["blocks"]
                    full_nnz = [b["full_nnz"] for b in blk_info]
                    removed_nonempty = [
                        i for i, b in enumerate(blk_info)
                        if b["overlaps"] and b["postband_nnz"] > 0
                    ]
                    entry = dict(pub)
                    entry["per_backend"] = {}
                    for be in BACKENDS[op]:
                        if be not in pub["per_backend"]:
                            continue
                        rec_times = match_recorded_blocks(m, op, be, yaml_blocks, full_nnz)
                        if any(rec_times[i] is None for i in removed_nonempty):
                            raise RuntimeError(f"unmatched recorded block for {m}/{op}/{be}")
                        removed_vbr_us = sum(rec_times[i] for i in removed_nonempty) / 1000.0
                        csr_old = fresh_csr_us(raw, m, "old", op, be)
                        csr_new = fresh_csr_us(raw, m, "new", op, be)
                        pub_be = pub["per_backend"][be]
                        new_total = pub_be["vdia_us"] - removed_vbr_us - csr_old + csr_new
                        entry["per_backend"][be] = {
                            "vdia_us": round(new_total, 2),
                            "baseline_us": pub_be["baseline_us"],
                            "speedup": round(pub_be["baseline_us"] / new_total, 4)
                            if new_total > 0 else 0.0,
                        }
                        report.append({
                            "table": "vdia_vbr_csr", "matrix": m, "op": op, "backend": be,
                            "published_us": pub_be["vdia_us"],
                            "removed_vbr_us": round(removed_vbr_us, 2),
                            "csr_old_us": round(csr_old, 2),
                            "csr_new_us": round(csr_new, 2),
                            "corrected_us": round(new_total, 2),
                            "n_blocks": n_blocks,
                            "n_blocks_removed": n_removed_blocks,
                        })
                    entry["best_vs_best"] = recompute_best(entry["per_backend"])
                    corrected[op].append(entry)

            # ---------------- VBR+VDIA+CSR ablation ----------------
            if info["kept_bands"]:
                kept_nnz = sum(s["nnz"] for bd in info["kept_bands"] for s in bd["segments"])
                kept_segments = sum(len(bd["segments"]) for bd in info["kept_bands"])
                if m not in AFFECTED_ABLATION:
                    entry = dict(pub)
                    ablation[op].append(entry)
                else:
                    yaml_blocks = info["blocks"]
                    blk_info = nnz_info[m]["blocks"]
                    full_nnz = [b["full_nnz"] for b in blk_info]
                    revived = [
                        i for i, b in enumerate(blk_info)
                        if b["overlaps"] and b["postband_nnz"] == 0
                    ]
                    n_segments = sum(len(bd["segments"]) for bd in info["bands"])
                    removed_seg_idx = []
                    seg_idx = 0
                    removed_ids = {id(bd) for bd in info["removed_bands"]}
                    for bd in info["bands"]:
                        for _ in bd["segments"]:
                            if id(bd) in removed_ids:
                                removed_seg_idx.append(seg_idx)
                            seg_idx += 1
                    removed_band_us, _ = buggy_removed_band_us(
                        seg_times_all, m, op, removed_seg_idx, n_segments)

                    entry = dict(pub)
                    entry["band_nnz_pct"] = kept_nnz / pub["nnz"] * 100
                    entry["n_segments"] = kept_segments
                    entry["per_backend"] = {}
                    for be in BACKENDS[op]:
                        if be not in pub["per_backend"]:
                            continue
                        rec_times = match_recorded_blocks(m, op, be, yaml_blocks, full_nnz)
                        if any(rec_times[i] is None for i in revived):
                            raise RuntimeError(f"unmatched revived block for {m}/{op}/{be}")
                        revived_vbr_us = sum(rec_times[i] for i in revived) / 1000.0
                        csr_old = fresh_csr_us(raw, m, "old", op, be)
                        csr_abl = fresh_csr_us(raw, m, "abl", op, be)
                        pub_be = pub["per_backend"][be]
                        abl_total = (pub_be["vdia_us"] + revived_vbr_us
                                     - removed_band_us - csr_old + csr_abl)
                        entry["per_backend"][be] = {
                            "vdia_us": round(abl_total, 2),
                            "baseline_us": pub_be["baseline_us"],
                            "speedup": round(pub_be["baseline_us"] / abl_total, 4)
                            if abl_total > 0 else 0.0,
                        }
                        report.append({
                            "table": "vbr_vdia_csr", "matrix": m, "op": op, "backend": be,
                            "published_us": pub_be["vdia_us"],
                            "revived_vbr_us": round(revived_vbr_us, 2),
                            "removed_band_vdia_us": round(removed_band_us, 2),
                            "csr_old_us": round(csr_old, 2),
                            "csr_abl_us": round(csr_abl, 2),
                            "ablation_us": round(abl_total, 2),
                            "n_bands_removed": n_removed_bands,
                        })
                    entry["best_vs_best"] = recompute_best(entry["per_backend"])
                    ablation[op].append(entry)

    for op in ("spmv", "spmm"):
        with open(RESULTS / f"{op}_vdia_vbr_csr_d075.json", "w") as f:
            json.dump(corrected[op], f, indent=1)
        with open(RESULTS / f"{op}_vbr_vdia_csr_d075.json", "w") as f:
            json.dump(ablation[op], f, indent=1)
        print(f"{op}: corrected {len(corrected[op])} matrices, "
              f"ablation {len(ablation[op])} matrices")

    for r in report:
        print(json.dumps(r))
    return 0

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", nargs="?", default="all",
                        choices=("measure", "aggregate", "all"),
                        help="measure the benchmarks, aggregate them, or both (default: all)")
    parser.add_argument("--matrices", nargs="*", default=None)
    parser.add_argument("--variants", type=str, default="old,new,abl")
    parser.add_argument("--operations", type=str, default="spmv,spmm")
    parser.add_argument("--backends", type=str, default=None,
                        help="Restrict CSR backends (default: all per operation)")
    parser.add_argument("--force", action="store_true",
                        help="Re-measure runs that are already recorded")
    args = parser.parse_args()

    if args.phase in ("measure", "all"):
        rc = measure(args)
        if rc:
            return rc
    if args.phase in ("aggregate", "all"):
        return aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
