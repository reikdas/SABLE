from __future__ import annotations

from sable.formats import VDIA
from sable.kernels.base import SpmmKernel, SpmvKernel
from sable.tensor import DenseLayout


def _empty_list() -> list[str]:
    return []


def _empty_dict() -> dict[str, str]:
    return {}


# ---------------------------------------------------------------------------
# Segment table
# ---------------------------------------------------------------------------
# Both kernels make one call per dispatch, which loops over a file-scope table
# of the segments, rather than one call (and one timed part) per segment.


def _segment_table(fmt: VDIA) -> list[tuple[int, int, int, int, int]]:
    """(row_start, nrows, first diagonal, ndiags, value offset) per segment.

    The kernels index a segment's diagonals as first + d, so its diagonals must
    be consecutive, as the band packer writes them.
    """
    table = []
    for seg in range(fmt.nsegments):
        ptr, ndiags = fmt.seg_idiag_ptr[seg], fmt.seg_ndiags[seg]
        diags = fmt.idiag.values[ptr:ptr + ndiags]
        if ndiags < 1 or list(diags) != list(range(diags[0], diags[0] + ndiags)):
            raise ValueError(f"VDIA kernels need consecutive diagonals; segment {seg} has {list(diags)}")
        table.append((fmt.seg_row_start[seg], fmt.seg_nrows[seg], diags[0], ndiags, fmt.seg_val_ptr[seg]))
    return table


def _emit_segment_table(fmt: VDIA, table: list[tuple[int, int, int, int, int]]) -> str:
    columns = list(zip(*table)) if table else [(), (), (), (), ()]
    return "".join(
        f"static const long {fmt.val}_seg_{name}[] = {{{', '.join(map(str, column)) or '0'}}};\n"
        for name, column in zip(("row0", "nrows", "off0", "nd", "voff"), columns))


# ---------------------------------------------------------------------------
# SpMV
# ---------------------------------------------------------------------------
# Rows whose diagonals all stay inside the matrix are computed 32 at a time in
# four AVX-512 accumulators: for each diagonal, 32 contiguous values times 32
# contiguous entries of x, so there are no index loads and no gathers. The
# rows where a diagonal leaves the matrix (at most the band width at each end
# of a segment) take a guarded scalar loop; fewer than 32 remaining rows take
# 8-row blocks and a masked tail.

_SPMV_LIBRARY = r"""
#ifndef SABLE_VDIA_SPMV
#define SABLE_VDIA_SPMV
static void sable_vdia_spmv_guarded(const double *dv, long r0, long nr, long off0, long nd, long a, long b,
        long m, const double *x, double *y) {
    for (long r = a; r < b; r++) {
        double t = 0.0;
        for (long d = 0; d < nd; d++) {
            long c = r + off0 + d;
            if (c >= 0 && c < m) t += dv[d * nr + (r - r0)] * x[c];
        }
        y[r] += t;
    }
}
static void sable_vdia_spmv_segment(const double *dv, long r0, long nr, long off0, long nd, long m,
        const double *x, double *y) {
    long e = r0 + nr, lo = r0 > -off0 ? r0 : -off0, hi = m - off0 - nd + 1;
    if (hi > e) hi = e;
    if (lo > e) lo = e;
    if (hi < lo) hi = lo;
    sable_vdia_spmv_guarded(dv, r0, nr, off0, nd, r0, lo, m, x, y);
    long r = lo;
    for (; r + 32 <= hi; r += 32) {
        __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd(), a2 = _mm512_setzero_pd(), a3 = _mm512_setzero_pd();
        const double *vp = dv + (r - r0), *xp = x + r + off0;
        for (long d = 0; d < nd; d++, vp += nr, xp++) {
            a0 = _mm512_fmadd_pd(_mm512_loadu_pd(vp), _mm512_loadu_pd(xp), a0);
            a1 = _mm512_fmadd_pd(_mm512_loadu_pd(vp + 8), _mm512_loadu_pd(xp + 8), a1);
            a2 = _mm512_fmadd_pd(_mm512_loadu_pd(vp + 16), _mm512_loadu_pd(xp + 16), a2);
            a3 = _mm512_fmadd_pd(_mm512_loadu_pd(vp + 24), _mm512_loadu_pd(xp + 24), a3);
        }
        double *yp = y + r;
        _mm512_storeu_pd(yp, _mm512_add_pd(_mm512_loadu_pd(yp), a0));
        _mm512_storeu_pd(yp + 8, _mm512_add_pd(_mm512_loadu_pd(yp + 8), a1));
        _mm512_storeu_pd(yp + 16, _mm512_add_pd(_mm512_loadu_pd(yp + 16), a2));
        _mm512_storeu_pd(yp + 24, _mm512_add_pd(_mm512_loadu_pd(yp + 24), a3));
    }
    for (; r + 8 <= hi; r += 8) {
        __m512d a0 = _mm512_setzero_pd();
        const double *vp = dv + (r - r0), *xp = x + r + off0;
        for (long d = 0; d < nd; d++, vp += nr, xp++)
            a0 = _mm512_fmadd_pd(_mm512_loadu_pd(vp), _mm512_loadu_pd(xp), a0);
        _mm512_storeu_pd(y + r, _mm512_add_pd(_mm512_loadu_pd(y + r), a0));
    }
    if (r < hi) {
        __mmask8 k = (__mmask8)((1u << (hi - r)) - 1u);
        __m512d a0 = _mm512_setzero_pd();
        const double *vp = dv + (r - r0), *xp = x + r + off0;
        for (long d = 0; d < nd; d++, vp += nr, xp++)
            a0 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(k, vp), _mm512_maskz_loadu_pd(k, xp), a0);
        _mm512_mask_storeu_pd(y + r, k, _mm512_add_pd(_mm512_maskz_loadu_pd(k, y + r), a0));
    }
    sable_vdia_spmv_guarded(dv, r0, nr, off0, nd, hi, e, m, x, y);
}
#endif
"""


class NaiveVDIASpmv(SpmvKernel):
    """VDIA SpMV: adds the band to y over the segment rows (AVX-512)."""

    accepts = VDIA

    def emit_includes(self) -> list[str]:
        return ["#include <immintrin.h>"]

    def emit_helpers(self, fmt: VDIA, rhs) -> str:
        table = _segment_table(fmt)
        p = fmt.val
        return f"""\
{_SPMV_LIBRARY}
{_emit_segment_table(fmt, table)}
static void {p}_spmv(double *y, const double *x, const double *val) {{
    for (long s = 0; s < {len(table)}; s++)
        sable_vdia_spmv_segment(val + {p}_seg_voff[s], {p}_seg_row0[s], {p}_seg_nrows[s], {p}_seg_off0[s],
                                {p}_seg_nd[s], {fmt.ncols}, x, y);
}}
"""

    def emit_call(self, fmt: VDIA, y: str, x: str, rhs) -> str:
        return f"{fmt.val}_spmv({y}, {x}, {fmt.val});\n"

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


# ---------------------------------------------------------------------------
# SpMM
# ---------------------------------------------------------------------------
# A tile computes R rows x 32 right-hand-side columns in 4R AVX-512
# accumulators and adds them to y. The "reuse" tile walks the x rows the
# tile's diagonals touch and loads each 32-column slice of x once for all R
# rows; it reads a copy of the segment's values padded with R-1 zero diagonals
# on each side (made at set-up), so the R rows share one loop. The "direct"
# tile loops over diagonals and loads x per row; it takes the rows left over
# at the end of a segment. Columns beyond the last multiple of 32 take a
# scalar loop.

_TILE_ROWS = 4
_TILE_COLS = 32
_PAD = _TILE_ROWS - 1


def _tile_function(kind: str, rows: int) -> str:
    acc = lambda r, k: f"a{r}{k}"  # noqa: E731
    lines = [
        f"static void sable_vdia_tile_{kind}{rows}(const double *dv, const double *dvp, long stride, long rloc,",
        "        long off0, long nd, long r0, long m, const double *x, double *y, long N, long NC) {",
        f"    for (long cb = 0; cb < NC; cb += {_TILE_COLS}) {{",
    ]
    for r in range(rows):
        lines.append("        __m512d " + ", ".join(
            f"{acc(r, k)} = _mm512_loadu_pd(y + (r0 + {r}) * N + cb + {8 * k})" for k in range(4)) + ";")
    if kind == "reuse":
        lines += [
            f"        long jlo = r0 + off0, jhi = r0 + {rows - 1} + off0 + nd - 1;",
            "        if (jlo < 0) jlo = 0;",
            "        if (jhi > m - 1) jhi = m - 1;",
            "        for (long j = jlo; j <= jhi; j++) {",
            "            const double *xr = x + j * N + cb;",
            "            __m512d x0 = _mm512_loadu_pd(xr), x1 = _mm512_loadu_pd(xr + 8), "
            "x2 = _mm512_loadu_pd(xr + 16), x3 = _mm512_loadu_pd(xr + 24);",
            f"            const double *d0 = dvp + (j - r0 - off0 + {rows - 1}) * stride + rloc;",
            "            __m512d b;",
        ]
        for r in range(rows):
            lines.append(f"            b = _mm512_set1_pd(d0[{r} - {r} * stride]); " + " ".join(
                f"{acc(r, k)} = _mm512_fmadd_pd(b, x{k}, {acc(r, k)});" for k in range(4)))
        lines.append("        }")
    else:
        lines += [
            "        for (long d = 0; d < nd; d++) {",
            "            const double *d0 = dv + d * stride + rloc;",
            "            long j0 = r0 + off0 + d;",
        ]
        for r in range(rows):
            lines.append(
                f"            if (j0 + {r} >= 0 && j0 + {r} < m) {{ const double *xr = x + (j0 + {r}) * N + cb; "
                f"__m512d b = _mm512_set1_pd(d0[{r}]); " + " ".join(
                    f"{acc(r, k)} = _mm512_fmadd_pd(b, _mm512_loadu_pd(xr + {8 * k}), {acc(r, k)});"
                    for k in range(4)) + " }")
        lines.append("        }")
    lines.append("        " + " ".join(
        f"_mm512_storeu_pd(y + (r0 + {r}) * N + cb + {8 * k}, {acc(r, k)});" for r in range(rows) for k in range(4)))
    lines += ["    }", "}", ""]
    return "\n".join(lines)


_SPMM_LIBRARY = f"""
#ifndef SABLE_VDIA_SPMM
#define SABLE_VDIA_SPMM
{_tile_function("reuse", _TILE_ROWS)}
{_tile_function("direct", 1)}
static void sable_vdia_spmm_segment(const double *dv, const double *dvp, long r0, long nr, long off0, long nd,
        long m, const double *x, double *y, long N) {{
    long NC = N - N % {_TILE_COLS}, r = r0;
    if (NC) {{
        for (; r + {_TILE_ROWS} <= r0 + nr; r += {_TILE_ROWS})
            sable_vdia_tile_reuse{_TILE_ROWS}(dv, dvp, nr, r - r0, off0, nd, r, m, x, y, N, NC);
        for (; r < r0 + nr; r++)
            sable_vdia_tile_direct1(dv, dvp, nr, r - r0, off0, nd, r, m, x, y, N, NC);
    }}
    if (NC < N)
        for (long row = 0; row < nr; row++)
            for (long d = 0; d < nd; d++) {{
                long col = r0 + row + off0 + d;
                if (col < 0 || col >= m) continue;
                double a = dv[d * nr + row];
                for (long c = NC; c < N; c++) y[(r0 + row) * N + c] += a * x[col * N + c];
            }}
}}
#endif
"""


class NaiveVDIASpmm(SpmmKernel):
    """VDIA SpMM with register tiles: adds the band to y over the segment rows.

    The right-hand side must be row-major; the code is AVX-512.
    """

    accepts = VDIA

    def _nrhs(self, rhs) -> int:
        if rhs.layout not in (None, DenseLayout.ROW_MAJOR):
            raise ValueError("NaiveVDIASpmm needs a row-major right-hand side")
        return rhs.shape[1]

    def emit_includes(self) -> list[str]:
        return ["#include <immintrin.h>"]

    def emit_helpers(self, fmt: VDIA, rhs) -> str:
        self._nrhs(rhs)
        table = _segment_table(fmt)
        p = fmt.val
        return f"""\
{_SPMM_LIBRARY}
{_emit_segment_table(fmt, table)}
static void {p}_spmm(double *y, const double *x, const double *val, double *const *pad, long N) {{
    for (long s = 0; s < {len(table)}; s++)
        sable_vdia_spmm_segment(val + {p}_seg_voff[s], pad[s], {p}_seg_row0[s], {p}_seg_nrows[s], {p}_seg_off0[s],
                                {p}_seg_nd[s], {fmt.ncols}, x, y, N);
}}
"""

    def emit_setup(self, fmt: VDIA, rhs) -> str:
        p, n = fmt.val, len(_segment_table(fmt))
        return f"""\
double **{p}_pad = (double **)malloc(sizeof(double *) * {n + 1});
assert({p}_pad != NULL);
for (long s = 0; s < {n}; s++) {{
    size_t cells = (size_t)({p}_seg_nd[s] + {2 * _PAD}) * {p}_seg_nrows[s];
    {p}_pad[s] = (double *)aligned_alloc(64, (cells * sizeof(double) + 63) / 64 * 64);
    assert({p}_pad[s] != NULL);
    memset({p}_pad[s], 0, cells * sizeof(double));
    memcpy({p}_pad[s] + {_PAD} * {p}_seg_nrows[s], {fmt.val} + {p}_seg_voff[s],
           (size_t){p}_seg_nd[s] * {p}_seg_nrows[s] * sizeof(double));
}}
"""

    def emit_call(self, fmt: VDIA, y: str, x: str, rhs) -> str:
        return f"{fmt.val}_spmm({y}, {x}, {fmt.val}, {fmt.val}_pad, {self._nrhs(rhs)});\n"

    def emit_teardown(self, fmt: VDIA, rhs) -> str:
        p = fmt.val
        return f"for (long s = 0; s < {len(_segment_table(fmt))}; s++) free({p}_pad[s]);\nfree({p}_pad);\n"

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)
