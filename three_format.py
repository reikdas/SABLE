"""Blocks and bands together (VBR+VDIA+CSR, blocks first) against the faster
of the two-format programs VBR+CSR and VDIA+CSR on the same matrix.

  three_format.py extract [--workers N] [matrix ...]
      For every matrix with stored blocks (find-submatrices/results/<m>.yaml)
      that pass the block extractor's coverage cutoff (BLOCK_MIN_COVERAGE, the
      matrices VBR+CSR is evaluated on) and each operation: replay the blocks,
      run the operation's band extractor on what they leave, and keep the
      matrix when those bands pass the band extractor's coverage cutoff on that
      residual. The choice is static: nothing is timed. Writes, under
      find-submatrices/three_format/:
        results_vbr050_vdia<op>/<m>.yaml  the stored blocks, then the bands of the residual
        results_vdia<op>/<m>.yaml         the bands of the whole matrix, recorded even when
                                          they miss the cutoff, so that VDIA+CSR is timed on
                                          every candidate
      A pair is a candidate exactly when its results_vbr050_vdia<op>/<m>.yaml
      exists; three_format.sh, bench_inspection.py, and the paper's
      composed_results.py list those directories.
  three_format.py analyze [results_dir] [--allow-unchecked]
      Tabulate three_format.sh's results (default results/three_format/).

Speedups use the same CSR kernel on every side; "best/best" lets each program
pick its fastest CSR kernel. Records whose output check failed or is missing
are left out (--allow-unchecked keeps records without a check).

Reachability: if CSR time were proportional to the nonzeros it handles, a
FMT+CSR program could be at most 1/r faster than CSR alone (r: the fraction
left to CSR), even with FMT free, and the composition could be at most
T2 / (fmt + csr2 * rc / r2) faster than a two-format program (T2, with CSR
part csr2 on a residual r2), even with the added format free (fmt: the
composition's own time for the format the two share; rc: its residual). A
speedup above its bound is marked '!': the nonzeros moved out of CSR do not
explain it. csr% is the composition's CSR part over csr2 * rc / r2 for the
faster two-format program.
"""
import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys

import yaml
from scipy.io import mmread
from scipy.sparse import csr_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "find-submatrices"))
import bench_suitesparse as bench  # noqa: E402
from find_vdia import SearchTimeout, band_candidates, select_bands  # noqa: E402
from sable import Matrix, Plan  # noqa: E402
from sable.extractors import BlockDetectorSkip  # noqa: E402
from scan_bands import PARAMETERS  # noqa: E402
from utils.fileio import parse_yaml_blocks  # noqa: E402

FILEPATH = pathlib.Path(__file__).resolve().parent
STORED_BLOCKS = FILEPATH / "find-submatrices" / "results"
EXTRACTIONS = FILEPATH / "find-submatrices" / "three_format"
RESULTS = FILEPATH / "results" / "three_format"
VARIANTS = {"vbr": "vbr050-blockmixed", "vdia": "vdia-bandnaive", "vbr_vdia": "vbr050-blockmixed_vdia-bandnaive"}
CSR_KERNELS = {"spmv": ("naive", "mkl", "spv8", "uzp"), "spmm": ("naive", "mkl", "spreg")}
NOISE = 1.03
BLOCK_MIN_COVERAGE = 0.08


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def _bands(csr, op, cutoff):
    """The operation's bands of `csr` and their selection record; below the
    coverage cutoff they are kept only when `cutoff` is False."""
    slot_cost, min_coverage = PARAMETERS[op]
    info = {}
    try:
        candidates = band_candidates(csr)
    except SearchTimeout:
        return [], {"choice": "timeout", "min_coverage": min_coverage}
    regions = select_bands(csr, slot_cost, 0.0, candidates=candidates, info=info)
    info.update(min_coverage=min_coverage, choice="bands" if info["coverage"] >= min_coverage else "below cutoff")
    if cutoff and info["choice"] != "bands":
        regions = []
    return [r.to_dict() for r in regions], info


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(bench._plain(record), f, sort_keys=False)


def candidates():
    """The (operation, matrix) pairs with a composed extraction, in a fixed order."""
    return [(op, p.stem) for op in PARAMETERS
            for p in sorted((EXTRACTIONS / f"results_vbr050_vdia{op}").glob("*.yaml"))]


def extract_one(name):
    found = bench.download_matrix_from_suitesparse(name)
    if found is None:
        return {"matrix": name, "error": "not found"}
    A = csr_matrix(mmread(found[0]))
    A.eliminate_zeros()
    A.sort_indices()
    blocks = parse_yaml_blocks(str(STORED_BLOCKS / f"{name}.yaml"))
    plan = Plan(Matrix(A, name=name), artifact_dir=str(EXTRACTIONS))
    plan.extract(BlockDetectorSkip(blocks))
    residual = plan.residual.to_csr()
    residual.sort_indices()
    block_coverage = 1 - residual.nnz / A.nnz
    out = {"matrix": name, "nnz": int(A.nnz), "block_coverage": round(block_coverage, 4)}
    for op in PARAMETERS:
        for kind in ("vdia", "vbr050_vdia"):
            (EXTRACTIONS / f"results_{kind}{op}" / f"{name}.yaml").unlink(missing_ok=True)
        if block_coverage < BLOCK_MIN_COVERAGE:
            out[op] = {"choice": "blocks below cutoff"}
            continue
        resid_bands, resid_info = _bands(residual, op, cutoff=True)
        out[op] = resid_info
        if not resid_bands:
            continue
        full_bands, full_info = _bands(A, op, cutoff=False)
        _write(EXTRACTIONS / f"results_vbr050_vdia{op}" / f"{name}.yaml",
               {"order": ["vbr", "vdia"], "block_source": f"find-submatrices/results/{name}.yaml",
                "band_extractor": op, "band_selection": resid_info, "bands": resid_bands,
                "blocks": [{"rows": [r0, r1], "cols": [c0, c1]} for r0, r1, c0, c1 in blocks]})
        _write(EXTRACTIONS / f"results_vdia{op}" / f"{name}.yaml",
               {"order": ["vdia"], "band_extractor": op, "band_selection": full_info, "bands": full_bands,
                "blocks": []})
    return out


def extract(args):
    names = args.matrices or sorted(p.stem for p in STORED_BLOCKS.glob("*.yaml"))
    with mp.Pool(args.workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(extract_one, names))
    for r in rows:
        if "error" in r:
            print(f"{r['matrix']}: {r['error']}")
    rows = [r for r in rows if "error" not in r]
    for op in PARAMETERS:
        picked = sorted((r for r in rows if r[op]["choice"] == "bands"), key=lambda r: r["nnz"])
        for r in picked:
            share = r[op]["coverage"] * (1 - r["block_coverage"])
            print(f"{op} {r['matrix']:20s} blocks {r['block_coverage']:6.1%}  bands {r[op]['coverage']:6.1%} of the "
                  f"residual ({share:5.1%} of the matrix)")
    print(f"{len(candidates())} candidates in {EXTRACTIONS}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def load(results_dir, allow_unchecked):
    data, bad = {}, []
    for op, kernels in CSR_KERNELS.items():
        for variant, token in VARIANTS.items():
            for csr in kernels:
                path = results_dir / f"{op}_{token}_csr-{csr}.json"
                if not path.exists():
                    continue
                for e in json.loads(path.read_text()):
                    t = e["timing"]["1 thread"]
                    key = (op, e["matrix_name"], variant, csr)
                    checks = [t.get("verification"), t.get("csr_baseline_verification")]
                    if (not all((c or {}).get("values_match", allow_unchecked and c is None) for c in checks)
                            or t.get("total_time_ns", 0) <= 0):
                        bad.append(key)
                        continue
                    parts = {}
                    for k, v in t["dispatch_times"].items():
                        label = e["dispatch_labels"][k]
                        parts[label] = parts.get(label, 0.0) + v["time_ns"] / 1e3
                    data[key] = {"total": t["total_time_ns"] / 1e3, "csr_only": t["csr_baseline_time_ns"] / 1e3,
                                 "parts": parts, "resid": e["nnz"]["residual_nnz_perc"] / 100}
    return data, bad


def bound_vs(two, comp, added):
    """Largest speedup of `comp` over the two-format record `two` if the `added`
    format were free, the other format took its time in `comp`, and CSR took
    `two`'s time per residual nonzero."""
    kept = sum(v for k, v in comp["parts"].items() if k not in ("csr", added))
    return two["total"] / (kept + two["parts"]["csr"] * comp["resid"] / two["resid"])


def geomean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def analyze(args):
    data, bad = load(pathlib.Path(args.results_dir), args.allow_unchecked)
    configs = candidates()
    if bad:
        print(f"Left out, output check failed or no timing ({len(bad)}):", sorted(set(bad)))
    summary = []
    for op in CSR_KERNELS:
        print(f"\n===== {op.upper()}  (us; VBR+VDIA+CSR over the faster two-format program, "
              "bound in brackets, ! = above the bound)")
        print(f"{'matrix':18s} {'csr':5s} {'CSR':>9s} {'VBR+CSR':>9s} {'VDIA+CSR':>9s} {'3-fmt':>9s} | "
              f"{'vs VBR':>13s} {'vs VDIA':>13s} {'vs faster':>9s} {'csr%':>5s} | two-format checks")
        for cop, m in configs:
            if cop != op:
                continue
            per = {}
            for c in CSR_KERNELS[op]:
                r = {v: data.get((op, m, v, c)) for v in VARIANTS}
                if not all(r.values()):
                    continue
                v, d, comp = r["vbr"], r["vdia"], r["vbr_vdia"]
                s_v, s_d = v["total"] / comp["total"], d["total"] / comp["total"]
                b_v, b_d = bound_vs(v, comp, "vdia"), bound_vs(d, comp, "vbr")
                faster = v if v["total"] <= d["total"] else d
                csr_pct = comp["parts"]["csr"] / (faster["parts"]["csr"] * comp["resid"] / faster["resid"])
                checks = [f"{name}+CSR {rec['csr_only'] / rec['total']:.2f} > 1/r {1 / rec['resid']:.2f}!"
                          for name, rec in (("VBR", v), ("VDIA", d)) if rec["csr_only"] / rec["total"] > 1 / rec["resid"]]
                print(f"{m:18s} {c:5s} {v['csr_only']:9.0f} {v['total']:9.0f} {d['total']:9.0f} {comp['total']:9.0f} | "
                      f"{s_v:5.2f} [{b_v:4.2f}]{'!' if s_v > b_v else ' '} {s_d:5.2f} [{b_d:4.2f}]{'!' if s_d > b_d else ' '} "
                      f"{min(s_v, s_d):9.2f} {csr_pct:5.0%} | {'; '.join(checks)}")
                per[c] = (min(v["total"], d["total"]), comp["total"], s_v <= b_v and s_d <= b_d)
            if not per:
                print(f"{m:18s} no complete kernel")
                continue
            best = min(p[0] for p in per.values()) / min(p[1] for p in per.values())
            holds = all(p[0] / p[1] >= NOISE and p[2] for p in per.values())
            summary.append((op, m, best, holds, sorted(per)))
            print(f"{m:18s} best/best {best:.3f}; {'holds' if holds else 'does not hold'} on every kernel "
                  f"({', '.join(sorted(per))})")

    print(f"\n===== VBR+VDIA+CSR over the faster two-format program, best CSR kernel on each side; "
          f"'holds': >= {NOISE} with every kernel, each within its bounds")
    for op in CSR_KERNELS:
        rows = [s for s in summary if s[0] == op]
        if not rows:
            continue
        wins = [s for s in rows if s[2] > 1.0]
        holds = [s for s in rows if s[3]]
        print(f"{op}: {len(rows)} candidates, geomean {geomean([s[2] for s in rows]):.3f}; {len(wins)} faster; "
              f"{len(holds)} hold: {', '.join(f'{s[1]} {s[2]:.2f}' for s in holds)}")
    holds = [s for s in summary if s[3]]
    print(f"all: {len(summary)} (operation, matrix) pairs on {len({s[1] for s in summary})} matrices, geomean "
          f"{geomean([s[2] for s in summary]):.3f}; {len(holds)} hold, on {len({s[1] for s in holds})} matrices")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("matrices", nargs="*")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(run=extract)
    p = sub.add_parser("analyze")
    p.add_argument("results_dir", nargs="?", default=str(RESULTS))
    p.add_argument("--allow-unchecked", action="store_true")
    p.set_defaults(run=analyze)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
