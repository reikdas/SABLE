"""Generate suitesparse-iter/: value-only SPD rewrites of the paper's VBR+CSR
matrices such that inspection + ONE VBR+CSR solve beats ONE CSR-only solve
(which has no inspection stage) for both gradient descent and Jacobi.

    python3 generate.py            # all matrices (~1.5 h, mostly scipy verification)
    python3 generate.py heart2     # regenerate one

The SuiteSparse originals are read from the ssgetpy cache (~/.ssgetpy) and
downloaded into it when missing; the generated .mtx files are not tracked.

Construction (sparsity pattern frozen, values replaced):

    M = I - gamma * N,   N = D^{-1/2} B D^{-1/2},  B = Sinkhorn-balanced sym(|offdiag(A)|)

rho(N) = 1 exactly with Perron vector v_i = sqrt(d_i), so lambda_min(M) = eta =
1 - gamma and the slow mode overlaps the pipeline's random b. With m0 =
|<v_hat, r0>|, r0 = b - M x0, the residual stopping rule ||r|| < delta ||b|| gives
exact-arithmetic iteration bounds (alpha = 1/||M||_inf is the pipeline's GD step):

    Jacobi:  ln(m0/(delta||b||)) / (-ln gamma)          <= k <= ln(||r0||/(delta||b||)) / (-ln gamma)
    GD:      ln(m0/(delta||b||)) / (-ln(1 - alpha*eta)) <= k <= ln(||r0||/(delta||b||)) / (-ln(1 - alpha*eta))

Measured C runs land on the lower bound, which is what is targeted. Sinkhorn
balancing keeps ||M||_inf ~ 2 so alpha ~ 0.45 (hub rows would otherwise inflate GD
counts ~30x). Per matrix and solver the break-even count is

    k_be = inspection_s / (t_csr - t_vbr_csr)

from the measured best per-iteration times in results/iterative_calibration.json
(frozen bench_iterative runs on earlier values of these matrices; per-iteration
times are pattern-determined) and the VBR+CSR inspection total from
results/inspection/inspection_spmv_vbr050_csr.json. eta is set so the
BINDING solver's lower bound is MARGIN * k_be; the counts are coupled (k_gd ~
k_jacobi / alpha), so the other solver overshoots. If the slowest solver's upper
bound would exceed the bench iteration cap (32-bit C loop counter), eta is
raised until it fits and the entry is flagged cap_limited (margin below MARGIN,
still above break-even). End-to-end speedup at k = m*k_be is m*S/(S - 1 + m),
S = t_csr/t_vbr_csr; MARGIN = 2 delivers about half of S - 1.

Suite: the paper's 55 VBR matrices minus 40 inadmissible for a value-only SPD
rewrite (8 rectangular, 13 with an incomplete stored diagonal, 19 pattern-
unsymmetric) leaves 15; DROPPED lists the 3 with no reachable break-even.
Writes <name>.mtx and, per matrix and solver, the iteration bounds and the
per-iteration times they were calibrated against to results/suitesparse_iter.json;
scipy verifies the full solve where it fits FULL_RUN_BUDGET_S.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "find-submatrices"))
from find_matrices import get_matrix_info

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(OUT_DIR)
RESULTS = os.path.join(REPO, "results")
VALUES = os.path.join(RESULTS, "suitesparse_iter.json")
INSPECTION = os.path.join(RESULTS, "inspection", "inspection_spmv_vbr050_csr.json")
CALIBRATION = os.path.join(RESULTS, "iterative_calibration.json")

DELTA = 0.01
SEED = 0
MARGIN = 2.0
ITERATION_CAP = 2_000_000_000  # bench_iterative.MAX_ITERATIONS; the C loop counter is an int
CAP_FRACTION = 0.95            # slowest solver's upper bound must stay below this * cap
FULL_RUN_BUDGET_S = 1800       # full scipy solve verification budget per matrix per solver
PROBE_ITERS = 10000
SOLVERS = ("gd", "jacobi")

MATRICES = ["eris1176", "c-30", "c-45", "heart2", "heart3", "bundle1", "TSC_OPF_300",
            "heart1", "TSC_OPF_1047", "net100", "exdata_1", "net125"]
DROPPED = {
    "gupta1": "VBR+CSR slower than CSR per iteration for both solvers (S = 0.93 GD, 0.96 Jacobi)",
    "net150": "VBR+CSR slower than CSR for GD (S = 0.97); Jacobi S = 1.003 needs 4e9 iterations",
    "c-57": "S = 1.02 GD / 1.01 Jacobi: break-even needs 2.5e9-4e9 iterations, above the cap",
}


def dump_json(obj, path):
    """Atomic write: a crash mid-dump must not leave a truncated file behind."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def best_per_iteration(entries, name, algo, composition):
    cands = [e for e in entries if e["matrix_name"] == name and e["algorithm"] == algo
             and e["composition"] == composition]
    if not cands:
        return None
    best = min(cands, key=lambda e: e["total_time_ns"] / e["iterations"])
    return best["total_time_ns"] / best["iterations"] / 1e9, best["kernels"]


def breakeven_targets(names):
    """{name: {gd: {...}, jacobi: {...}}} from measured timings."""
    insp = {e["matrix_name"]: e["phases"] for e in json.load(open(INSPECTION)) if "error" not in e}
    entries = [e for e in json.load(open(CALIBRATION)) if "error" not in e]
    targets = {}
    for name in names:
        p = insp[name]
        total = (p["vbr"]["partitioner_seconds"] + p["csr"]["convert_seconds"]
                 + p["codegen"]["wall_seconds"] + p["gcc"]["wall_seconds"])
        need = {}
        for algo in SOLVERS:
            vc = best_per_iteration(entries, name, algo, "vbr_csr")
            cs = best_per_iteration(entries, name, algo, "csr")
            if not (vc and cs):
                raise RuntimeError(f"no timings for {name}/{algo}")
            speedup = cs[0] / vc[0]
            assert speedup > 1.0, (name, algo, speedup)
            need[algo] = {
                "per_iteration_s": {"vbr_csr": vc[0], "csr": cs[0]},
                "kernels": {"vbr_csr": vc[1], "csr": cs[1]},
                "iters_breakeven": total / (cs[0] - vc[0]),
            }
        targets[name] = need
    return targets


def build_N(name):
    """Normalized adjacency N (rho = 1), its Perron vector v, and n."""
    info = get_matrix_info(name)
    subdir, _ = info.localpath(format="MM", extract=True)
    path = os.path.join(subdir, f"{info.name}.mtx")
    if not os.path.exists(path):
        print(f"[{name}] downloading original ({info.nnz} nnz) into the ssgetpy cache...", flush=True)
        info.download(format="MM", extract=True)
    A = scipy.io.mmread(path).tocsr()
    n = A.shape[0]
    Babs = abs(A) + abs(A.T)
    Babs.setdiag(0)
    Babs.eliminate_zeros()
    w = np.ones(n)
    mask = np.asarray(Babs.sum(axis=1)).ravel() > 0
    for _ in range(500):
        rs = w * (Babs @ w)
        dev = np.abs(np.log(rs[mask]))
        if dev.max() < 0.2:
            break
        w[mask] *= rs[mask] ** -0.25  # damped symmetric Sinkhorn sweep
    Babs = (sp.diags(w) @ Babs @ sp.diags(w)).tocsr()
    rows_sums = np.asarray(Babs.sum(axis=1)).ravel()
    s = np.where(rows_sums > 0, 1.0 / np.sqrt(np.maximum(rows_sums, 1e-300)), 1.0)
    S = sp.diags(s)
    N = (S @ Babs @ S).tocsr()
    N = ((N + N.T) * 0.5).tocsr()
    v = np.where(rows_sums > 0, np.sqrt(rows_sums), 0.0)
    return N, v, n


def solver_run(M, dinv, x0, b, alpha, cap, algo):
    """Exact replica of the pipeline's solver loops (residual-based stop)."""
    x = x0.copy()
    b_norm = float(b @ b)
    rn = np.inf
    it = 0
    while it < cap and rn >= b_norm * DELTA * DELTA:
        r = M @ x - b
        rn = float(r @ r)
        x = x - (alpha * r if algo == "gd" else r * dinv)
        it += 1
    return it, rn < b_norm * DELTA * DELTA


def eta_for_count(count, log_term, algo, nrow):
    """eta at which the solver's bound log_term / rate(eta) equals `count`."""
    rate = log_term / count
    if algo == "jacobi":
        return 1.0 - np.exp(-rate)  # rate_jacobi = -ln(gamma)
    # rate_gd = -ln(1 - alpha*eta), alpha = 1 / (1 + gamma*nrow): fixed point in eta.
    eta = 0.0
    for _ in range(100):
        alpha = 1.0 / (1.0 + (1.0 - eta) * nrow)
        new = -np.expm1(-rate) / alpha
        if abs(new - eta) <= 1e-13 * new:
            return new
        eta = new
    raise RuntimeError("GD eta fixed point did not converge")


def calibrate(need, log_lb, log_ub, nrow):
    etas = {a: eta_for_count(MARGIN * need[a]["iters_breakeven"], log_lb, a, nrow) for a in SOLVERS}
    binding = min(etas, key=etas.get)
    eta = etas[binding]
    eta_cap = max(eta_for_count(CAP_FRACTION * ITERATION_CAP, log_ub, a, nrow) for a in SOLVERS)
    cap_limited = bool(eta_cap > eta)
    return max(eta, eta_cap), binding, cap_limited


def process(name, need):
    t_start = time.time()
    N, v, n = build_N(name)
    nv = np.linalg.norm(v)
    eig_err = np.linalg.norm(N @ v - v) / nv
    assert eig_err < 1e-10, (name, eig_err)
    v_hat = v / nv
    nrow = float(np.asarray(N.sum(axis=1)).ravel().max())

    rng = np.random.default_rng(SEED)
    x0 = rng.random(n)
    b = rng.random(n)
    thresh = np.linalg.norm(b) * DELTA
    r0 = b - (x0 - N @ x0)  # gamma ~ 1: m0 and ||r0|| are insensitive to eta
    m0 = abs(float(v_hat @ r0))
    log_lb = np.log(m0 / thresh)
    log_ub = np.log(np.linalg.norm(r0) / thresh)
    assert log_lb > 0, (name, m0, thresh)

    eta, binding, cap_limited = calibrate(need, log_lb, log_ub, nrow)
    gamma = 1.0 - eta
    path = os.path.join(OUT_DIR, f"{name}.mtx")
    scipy.io.mmwrite(path, (sp.eye(n, format="csr") - gamma * N).tocsr(), symmetry="symmetric")
    M = scipy.io.mmread(path).tocsr()  # verify exactly what the pipeline will read
    alpha = 1.0 / float(np.asarray(abs(M).sum(axis=1)).max())
    dinv = 1.0 / M.diagonal()
    rates = {"jacobi": -np.log(gamma), "gd": -np.log1p(-alpha * eta)}

    entry = {"nnz": int(M.nnz)}
    margins = {}
    for algo in SOLVERS:
        lb, ub = log_lb / rates[algo], log_ub / rates[algo]
        k_be = need[algo]["iters_breakeven"]
        assert lb >= k_be, (name, algo, lb, k_be)
        assert ub <= CAP_FRACTION * ITERATION_CAP * (1 + 1e-9), (name, algo, ub)
        if algo == binding and not cap_limited:
            assert abs(lb - MARGIN * k_be) <= 1e-6 * lb, (name, algo, lb, MARGIN * k_be)
        entry[algo] = {
            "iter_bounds": [float(lb), float(ub)],
            "per_iteration_s": need[algo]["per_iteration_s"],
            "kernels": need[algo]["kernels"],
        }
        margins[algo] = lb / k_be

    verified = {}
    per_iter = None
    for algo in ("jacobi", "gd"):
        lb, ub = entry[algo]["iter_bounds"]
        if per_iter is None:
            t0 = time.time()
            it_probe, _ = solver_run(M, dinv, x0, b, alpha, 100, algo)
            per_iter = (time.time() - t0) / max(it_probe, 1)
        if ub * per_iter <= FULL_RUN_BUDGET_S:
            it, ok = solver_run(M, dinv, x0, b, alpha, int(4 * ub) + 10, algo)
            assert ok, (name, algo, "did not converge in scipy verification")
            assert it >= need[algo]["iters_breakeven"], (name, algo, it)
            assert lb * 0.999 <= it <= ub * 1.001, (name, algo, it, lb, ub)
            verified[algo] = it
        else:
            it, _ = solver_run(M, dinv, x0, b, alpha, PROBE_ITERS, algo)
            assert it == PROBE_ITERS, (name, algo, "converged way too early", it)
    print(f"{name}: eta={eta:.3e} binding={binding}{' CAP-LIMITED' if cap_limited else ''}"
          + "".join(f" {algo} {entry[algo]['iter_bounds'][0]:,.0f} ({margins[algo]:.2f}x be,"
                    f" verified={verified.get(algo)})" for algo in SOLVERS)
          + f" ({time.time() - t_start:.1f}s)", flush=True)
    return entry


def main():
    names = sys.argv[1:] or MATRICES
    unknown = [n for n in names if n not in MATRICES]
    if unknown:
        raise SystemExit(f"not in MATRICES: {unknown}")
    targets = breakeven_targets(names)
    values = json.load(open(VALUES)) if os.path.exists(VALUES) else {}
    for name in DROPPED:
        path = os.path.join(OUT_DIR, f"{name}.mtx")
        if os.path.exists(path):
            os.remove(path)
            print(f"dropped {name}: removed {path}", flush=True)
        values.pop(name, None)

    for name in names:
        values[name] = process(name, targets[name])
        dump_json(values, VALUES)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
