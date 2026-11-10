import pathlib
import sys

import numpy
import pytest
import scipy.sparse

from sable import Matrix, Plan
from sable.extractors import BandExtractorSkip, CSRConvertor, SpMMBandExtractor, SpMVBandExtractor
from sable.kernels import NaiveCSRSpmm, NaiveCSRSpmv, NaiveVDIASpmm, NaiveVDIASpmv
from sable.tensor import DenseInput, DenseLayout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "find-submatrices"))
import find_vdia  # noqa: E402

N_ROWS = 203
# Listed out of row order, with segments that leave the matrix at the top and
# the bottom, one shorter than a 4-row tile and a 32-row block, rows shared by
# no band, and a 41-diagonal segment.
BANDS = [
    {"diag_offset": 6, "segments": [{"rows": [195, 203], "bandwidth": [4, 4]}]},
    {"diag_offset": -1, "segments": [{"rows": [0, 37], "bandwidth": [2, 2]}]},
    {"diag_offset": 0, "segments": [{"rows": [40, 43], "bandwidth": [1, 1]}, {"rows": [60, 150], "bandwidth": [0, 0]}]},
    {"diag_offset": 0, "segments": [{"rows": [150, 190], "bandwidth": [20, 20]}]},
]
OVERLAPPING = [{"diag_offset": 0, "segments": [{"rows": [0, 30], "bandwidth": [1, 1]}]},
               {"diag_offset": 3, "segments": [{"rows": [20, 40], "bandwidth": [0, 0]}]}]
# Diagonals over all rows: a tridiagonal band, couplings at +-29 as on a
# 7 x 29 grid and a run of two at 60-61, which leave 3 whole 32-row blocks
# inside the matrix; adding -150 and 120 leaves none.
STENCIL_OFFSETS = [-29, -1, 0, 1, 29, 60, 61]
FAR_STENCIL_OFFSETS = [-150] + STENCIL_OFFSETS + [120]
# select_bands' parameters for each operation's kernel.
SPMV = (find_vdia.spmv_slot_cost, find_vdia.SPMV_MIN_COVERAGE)
SPMM = (find_vdia.spmm_slot_cost, find_vdia.SPMM_MIN_COVERAGE)
# The VDIA kernels emit AVX-512 intrinsics, so their programs build and run only on CPUs with AVX-512F.
HAS_AVX512F = "avx512f" in pathlib.Path("/proc/cpuinfo").read_text().split()
requires_avx512 = pytest.mark.skipif(not HAS_AVX512F, reason="the VDIA kernels need AVX-512F")


def _matrix() -> scipy.sparse.csr_matrix:
    rng = numpy.random.default_rng(7)
    rows, cols = [], []
    for band in BANDS + OVERLAPPING:
        for seg in band["segments"]:
            lower, upper = seg["bandwidth"]
            for delta in range(-lower, upper + 1):
                r = numpy.arange(*seg["rows"])
                c = r + band["diag_offset"] + delta
                keep = (c >= 0) & (c < N_ROWS) & (rng.random(r.size) < 0.8)
                rows.append(r[keep])
                cols.append(c[keep])
    rows.append(rng.integers(0, N_ROWS, 600))
    cols.append(rng.integers(0, N_ROWS, 600))
    r, c = numpy.concatenate(rows), numpy.concatenate(cols)
    A = scipy.sparse.csr_matrix((rng.standard_normal(r.size), (r, c)), shape=(N_ROWS, N_ROWS))
    A.sum_duplicates()
    A.data[A.indptr[55]:A.indptr[56]] = 0.0              # an empty row between segments
    A.eliminate_zeros()
    A.sort_indices()
    return A


def _stencil(offsets, n=N_ROWS) -> scipy.sparse.csr_matrix:
    """Full diagonals at `offsets` plus scattered nonzeros."""
    rng = numpy.random.default_rng(11)
    rows, cols = [], []
    for offset in offsets:
        r = numpy.arange(max(0, -offset), min(n, n - offset))
        rows.append(r)
        cols.append(r + offset)
    rows.append(rng.integers(0, n, 300))
    cols.append(rng.integers(0, n, 300))
    r, c = numpy.concatenate(rows), numpy.concatenate(cols)
    A = scipy.sparse.csr_matrix((rng.standard_normal(r.size), (r, c)), shape=(n, n))
    A.sum_duplicates()
    A.sort_indices()
    return A


def _rhs(tmp_path, nrhs):
    if nrhs is None:
        X = 1.0 + 0.001 * (numpy.arange(N_ROWS) % 97)
        path = tmp_path / "x.vector"
        path.write_text(",".join(repr(v) for v in X) + "\n")
        return X, DenseInput.vector(str(path), N_ROWS)
    X = 1.0 + 0.001 * (numpy.arange(N_ROWS * nrhs) % 97)
    path = tmp_path / "x.matrix"
    path.write_text(",".join(repr(v) for v in X) + "\n")
    return X.reshape(N_ROWS, nrhs), DenseInput.matrix(str(path), shape=(N_ROWS, nrhs), layout=DenseLayout.ROW_MAJOR)


def _result(output: str, shape) -> numpy.ndarray:
    lines = output.splitlines()
    start = lines.index("") + 1
    return numpy.array([float(v) for v in lines[start:] if v.strip()]).reshape(shape)


def _run(tmp_path, bands, vdia_kernel, csr_kernel, nrhs, A=None):
    A = _matrix() if A is None else A
    X, rhs = _rhs(tmp_path, nrhs)
    plan = Plan(Matrix(A, name="vdia"), artifact_dir=str(tmp_path))
    plan.rhs(rhs)
    plan.dispatch(plan.extract(BandExtractorSkip(bands=bands)), vdia_kernel)
    plan.dispatch(plan.extract(CSRConvertor()), csr_kernel)
    executor = plan.compile(filename="vdia", bench=2)
    source = pathlib.Path(executor.c_path).read_text()
    output = executor.build().run()
    return A, X, source, output


@requires_avx512
def test_vdia_spmv_matches_scipy(tmp_path):
    A, x, source, output = _run(tmp_path, BANDS, NaiveVDIASpmv(), NaiveCSRSpmv(), None)
    assert source.count("vdia_val_spmv(y, x, vdia_val);") == 1          # one call for all segments
    assert "Dispatch 1 Part" not in output
    numpy.testing.assert_allclose(_result(output, (N_ROWS,)), A @ x, rtol=1e-10, atol=1e-10)


@requires_avx512
@pytest.mark.parametrize("nrhs", [64, 20, 45])      # whole 32-column tiles, none, and a remainder
def test_vdia_spmm_matches_scipy(tmp_path, nrhs):
    A, X, source, output = _run(tmp_path, BANDS, NaiveVDIASpmm(), NaiveCSRSpmm(), nrhs)
    assert source.count("vdia_val_spmm(y, x, vdia_val, vdia_val_pad,") == 1
    assert "Dispatch 1 Part" not in output
    numpy.testing.assert_allclose(_result(output, (N_ROWS, nrhs)), A @ X, rtol=1e-10, atol=1e-10)


@requires_avx512
@pytest.mark.parametrize("op", ["spmv", "spmm"])
def test_vdia_kernels_accept_segments_sharing_rows(tmp_path, op):
    kernel, csr, nrhs = (NaiveVDIASpmv(), NaiveCSRSpmv(), None) if op == "spmv" else (NaiveVDIASpmm(), NaiveCSRSpmm(), 32)
    A, X, _, output = _run(tmp_path, OVERLAPPING, kernel, csr, nrhs)
    numpy.testing.assert_allclose(_result(output, X.shape), A @ X, rtol=1e-10, atol=1e-10)


def _runs_over_all_rows(offsets) -> list[dict]:
    runs: list[list[int]] = []
    for d in sorted(offsets):
        if runs and d == runs[-1][-1] + 1:
            runs[-1].append(d)
        else:
            runs.append([d])
    return [{"diag_offset": run[0], "segments": [{"rows": [0, N_ROWS], "bandwidth": [0, len(run) - 1]}]}
            for run in runs]


@requires_avx512
@pytest.mark.parametrize("offsets", [STENCIL_OFFSETS, FAR_STENCIL_OFFSETS])
def test_spmv_kernel_computes_diagonals_sharing_rows(tmp_path, offsets):
    A = _stencil(offsets)
    _, x, _, output = _run(tmp_path, _runs_over_all_rows(offsets), NaiveVDIASpmv(), NaiveCSRSpmv(), None, A=A)
    numpy.testing.assert_allclose(_result(output, (N_ROWS,)), A @ x, rtol=1e-10, atol=1e-10)


def _shifted_band(n=6000) -> scipy.sparse.csr_matrix:
    """A full tridiagonal band centred on 0, then on +300, then on -400, plus scattered nonzeros."""
    rng = numpy.random.default_rng(5)
    rows, cols = [], []
    for (r0, r1), centre in (((0, 1500), 0), ((1500, 3900), 300), ((3900, n), -400)):
        for delta in (-1, 0, 1):
            r = numpy.arange(r0, r1)
            c = r + centre + delta
            keep = (c >= 0) & (c < n)
            rows.append(r[keep])
            cols.append(c[keep])
    rows.append(rng.integers(0, n, 300))
    cols.append(rng.integers(0, n, 300))
    r, c = numpy.concatenate(rows), numpy.concatenate(cols)
    A = scipy.sparse.csr_matrix((numpy.ones(r.size), (r, c)), shape=(n, n))
    A.sum_duplicates()
    A.sort_indices()
    return A


def test_select_bands_follows_a_band_that_moves():
    A = _shifted_band()
    info = {}
    regions = find_vdia.select_bands(A, *SPMV, info=info)
    assert info["choice"] == "bands" and info["coverage"] >= 0.9
    spans = {(r.diag_offset - s.lower_bw, r.diag_offset + s.upper_bw) for r in regions for s in r.segments}
    assert spans == {(-1, 1), (299, 301), (-401, -399)}
    rows = sorted((s.row_start, s.row_end) for r in regions for s in r.segments)
    assert all(a[1] <= b[0] for a, b in zip(rows, rows[1:]))     # row-disjoint


def test_select_bands_finds_bands_far_from_the_diagonal():
    # Full tridiagonal pieces centred at +1000 and +1500.
    n = 6000
    rows, cols = [], []
    for (r0, r1), centre in (((0, 2500), 1000), ((2500, 4400), 1500)):
        for delta in (-1, 0, 1):
            r = numpy.arange(r0, r1)
            c = r + centre + delta
            keep = c < n
            rows.append(r[keep])
            cols.append(c[keep])
    r, c = numpy.concatenate(rows), numpy.concatenate(cols)
    A = scipy.sparse.csr_matrix((numpy.ones(r.size), (r, c)), shape=(n, n))
    info = {}
    find_vdia.select_bands(A, *SPMV, info=info)
    assert info["choice"] == "bands" and info["coverage"] > 0.99


@pytest.mark.skipif(find_vdia._njit is None, reason="numba not installed")
@pytest.mark.parametrize("min_density", [0.75, 0.45])
def test_compiled_band_search_matches_python(monkeypatch, min_density):
    found = 0
    for A in (_matrix(), _dense_band(fill=0.85)):
        A.sort_indices()
        monkeypatch.setattr(find_vdia, "USE_COMPILED", False)
        python = [r.to_dict() for r in find_vdia.find_vdia_regions(A, min_density=min_density, min_savings=0.0)]
        monkeypatch.setattr(find_vdia, "USE_COMPILED", True)
        compiled = [r.to_dict() for r in find_vdia.find_vdia_regions(A, min_density=min_density, min_savings=0.0)]
        for region in python + compiled:
            region.pop("found_at_seconds", None)
        assert compiled == python
        found += len(python)
    assert found                                        # the comparison covered real bands


def _dense_band(n=2000, half=15, fill=1.0, seed=0):
    rng = numpy.random.default_rng(seed)
    r = numpy.repeat(numpy.arange(n), 2 * half + 1)
    c = r + numpy.tile(numpy.arange(-half, half + 1), n)
    keep = (c >= 0) & (c < n) & (rng.random(r.size) < fill)
    return scipy.sparse.csr_matrix((numpy.ones(int(keep.sum())), (r[keep], c[keep])), shape=(n, n))


def test_select_bands_parameters_decide_a_partly_filled_band():
    # A 31-diagonal band 70% full pays for SpMM (break-even 0.53 at that width) but not for SpMV (0.90).
    A = _dense_band(fill=0.7)
    spmv, spmm = {}, {}
    assert find_vdia.select_bands(A, *SPMV, info=spmv) == []
    assert spmv["choice"] == "none"
    assert find_vdia.select_bands(A, *SPMM, info=spmm)
    assert spmm["choice"] == "bands" and spmm["coverage"] >= find_vdia.SPMM_MIN_COVERAGE
    full = {}
    assert find_vdia.select_bands(_dense_band(), *SPMV, info=full)
    assert full["coverage"] > 0.99 and full["density"] >= find_vdia.SPMV_SLOT_COST


@pytest.mark.parametrize("params", [SPMV, SPMM], ids=["spmv", "spmm"])
def test_select_bands_declines_a_matrix_without_bands(params):
    A = scipy.sparse.random(3000, 3000, density=0.002, random_state=numpy.random.default_rng(3), format="csr")
    A.sort_indices()
    info = {}
    assert find_vdia.select_bands(A, *params, info=info) == []
    assert info["choice"] == "none"


def test_select_bands_declines_when_the_search_runs_out_of_time():
    info = {}
    assert find_vdia.select_bands(_dense_band(), *SPMV, timeout_seconds=0.0, info=info) == []
    assert info["choice"] == "timeout"


@pytest.mark.parametrize("extractor", [SpMVBandExtractor, SpMMBandExtractor])
def test_frontend_band_extractors_claim_the_band(extractor):
    A = _dense_band(n=600)
    vdia, residual = extractor().extract(Matrix(A, name="band"))
    assert vdia.nsegments >= 1
    assert residual.nnz < 0.05 * A.nnz
