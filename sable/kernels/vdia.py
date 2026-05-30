from __future__ import annotations

from sable.build_config import MKL_FLAGS
from sable.codegen import OutOfLineCode, out_of_line
from sable.formats import VDIA
from sable.kernels.base import SpmmKernel, SpmvKernel


def _empty_list() -> list[str]:
    return []


def _empty_dict() -> dict[str, str]:
    return {}


def _mkl_compile_flags() -> list[str]:
    return [flag for flag in MKL_FLAGS if flag.startswith("-I")]


def _mkl_link_flags() -> list[str]:
    return [flag for flag in MKL_FLAGS if not flag.startswith("-I")]


# ---------------------------------------------------------------------------
# Naive SpMV out-of-line helpers
# ---------------------------------------------------------------------------

def _spmv_naive_body(fmt: VDIA, y: str, x: str) -> str:
    return f"""\
for (int d = 0; d < ndiags; d++) {{
    int diag = {fmt.idiag}[idiag_off + d];
    for (int row = 0; row < nrows; row++) {{
        int col = row0 + row + diag;
        if (0 <= col && col < {fmt.ncols}) {{
            {y}[row0 + row] += {fmt.val}[val_off + d * nrows + row] * {x}[col];
        }}
    }}
}}
"""


def _spmv_naive_helper_name(fmt: VDIA) -> str:
    return f"{fmt.val}_spmv_naive_segment"


_SPMV_NAIVE_PARAMS = ["int row0", "int nrows", "int ndiags", "int idiag_off", "int val_off"]


def _spmv_naive_args(fmt: VDIA, seg_idx: int) -> list[int]:
    return [
        fmt.seg_row_start[seg_idx],
        fmt.seg_nrows[seg_idx],
        fmt.seg_ndiags[seg_idx],
        fmt.seg_idiag_ptr[seg_idx],
        fmt.seg_val_ptr[seg_idx],
    ]


# ---------------------------------------------------------------------------
# Naive SpMM out-of-line helpers
# ---------------------------------------------------------------------------

def _spmm_naive_body(fmt: VDIA, y: str, x: str) -> str:
    return f"""\
for (int row = 0; row < nrows; row++) {{
    for (int d = 0; d < ndiags; d++) {{
        int col = row0 + row + {fmt.idiag}[idiag_off + d];
        if (0 <= col && col < {fmt.ncols}) {{
            double a = {fmt.val}[val_off + d * nrows + row];
            for (int rhs_col = 0; rhs_col < nrhs; rhs_col++) {{
                {y}[(row0 + row) * nrhs + rhs_col] += a * {x}[col * nrhs + rhs_col];
            }}
        }}
    }}
}}
"""


def _spmm_naive_helper_name(fmt: VDIA) -> str:
    return f"{fmt.val}_spmm_naive_segment"


_SPMM_NAIVE_PARAMS = ["int row0", "int nrows", "int ndiags", "int idiag_off", "int val_off", "int nrhs"]


def _spmm_naive_args(fmt: VDIA, seg_idx: int, nrhs: int) -> list[int]:
    return [
        fmt.seg_row_start[seg_idx],
        fmt.seg_nrows[seg_idx],
        fmt.seg_ndiags[seg_idx],
        fmt.seg_idiag_ptr[seg_idx],
        fmt.seg_val_ptr[seg_idx],
        nrhs,
    ]


# ---------------------------------------------------------------------------
# MKL DIA SpMV out-of-line helpers
# ---------------------------------------------------------------------------

def _spmv_mkl_body(fmt: VDIA, y: str, x: str) -> str:
    return f"""\
{{
MKL_INT mkl_m = nrows;
MKL_INT mkl_k = {fmt.ncols};
MKL_INT mkl_lval = nrows;
MKL_INT mkl_ndiag = ndiags;
double mkl_alpha = 1.0;
double mkl_beta = 1.0;
char mkl_transa = 'N';
char mkl_matdescra[6] = {{'G', ' ', ' ', 'C', ' ', ' '}};
mkl_ddiamv(&mkl_transa, &mkl_m, &mkl_k, &mkl_alpha, mkl_matdescra,
           &{fmt.val}[val_off], &mkl_lval, (MKL_INT *)&{fmt.idiag}[idiag_off], &mkl_ndiag,
           &{x}[0], &mkl_beta, &{y}[row0]);
}}
"""


def _spmv_mkl_helper_name(fmt: VDIA) -> str:
    return f"{fmt.val}_spmv_mkl_dia_segment"


_SPMV_MKL_PARAMS = ["int row0", "int nrows", "int ndiags", "int idiag_off", "int val_off"]


def _spmv_mkl_args(fmt: VDIA, seg_idx: int) -> list[int]:
    return [
        fmt.seg_row_start[seg_idx],
        fmt.seg_nrows[seg_idx],
        fmt.seg_ndiags[seg_idx],
        fmt.seg_idiag_ptr[seg_idx],
        fmt.seg_val_ptr[seg_idx],
    ]


# ---------------------------------------------------------------------------
# MKL DIA SpMM out-of-line helpers
# ---------------------------------------------------------------------------

def _spmm_mkl_body(fmt: VDIA, y: str, x: str) -> str:
    return f"""\
{{
MKL_INT mkl_m = nrows;
MKL_INT mkl_n = nrhs;
MKL_INT mkl_k = {fmt.ncols};
MKL_INT mkl_lval = nrows;
MKL_INT mkl_ndiag = ndiags;
MKL_INT mkl_ldb = nrhs;
MKL_INT mkl_ldc = nrhs;
double mkl_alpha = 1.0;
double mkl_beta = 1.0;
char mkl_transa = 'N';
char mkl_matdescra[6] = {{'G', ' ', ' ', 'C', ' ', ' '}};
mkl_ddiamm(&mkl_transa, &mkl_m, &mkl_n, &mkl_k, &mkl_alpha, mkl_matdescra,
           &{fmt.val}[val_off], &mkl_lval, (MKL_INT *)&{fmt.idiag}[idiag_off], &mkl_ndiag,
           &{x}[0], &mkl_ldb, &mkl_beta, &{y}[row0 * nrhs], &mkl_ldc);
}}
"""


def _spmm_mkl_helper_name(fmt: VDIA) -> str:
    return f"{fmt.val}_spmm_mkl_dia_segment"


_SPMM_MKL_PARAMS = ["int row0", "int nrows", "int ndiags", "int idiag_off", "int val_off", "int nrhs"]


def _spmm_mkl_args(fmt: VDIA, seg_idx: int, nrhs: int) -> list[int]:
    return [
        fmt.seg_row_start[seg_idx],
        fmt.seg_nrows[seg_idx],
        fmt.seg_ndiags[seg_idx],
        fmt.seg_idiag_ptr[seg_idx],
        fmt.seg_val_ptr[seg_idx],
        nrhs,
    ]


class _BaseVDIAKernel:
    pass


class NaiveVDIASpmv(_BaseVDIAKernel, SpmvKernel):
    accepts = VDIA

    def emit_includes(self) -> list[str]:
        return []

    def emit_helpers(self, fmt: VDIA, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VDIA, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return [
            out_of_line(
                _spmv_naive_body(fmt, y, x),
                name=_spmv_naive_helper_name(fmt),
                parameters=_SPMV_NAIVE_PARAMS,
                arguments=_spmv_naive_args(fmt, i),
            )
            for i in range(fmt.nsegments)
        ]

    def emit_call(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VDIA, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


class NaiveVDIASpmm(_BaseVDIAKernel, SpmmKernel):
    accepts = VDIA

    def emit_includes(self) -> list[str]:
        return []

    def emit_helpers(self, fmt: VDIA, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VDIA, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        nrhs = rhs.shape[1]
        return [
            out_of_line(
                _spmm_naive_body(fmt, y, x),
                name=_spmm_naive_helper_name(fmt),
                parameters=_SPMM_NAIVE_PARAMS,
                arguments=_spmm_naive_args(fmt, i, nrhs),
            )
            for i in range(fmt.nsegments)
        ]

    def emit_call(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VDIA, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


class MKLDIASpmv(_BaseVDIAKernel, SpmvKernel):
    accepts = VDIA

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>"]

    def emit_helpers(self, fmt: VDIA, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VDIA, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return [
            out_of_line(
                _spmv_mkl_body(fmt, y, x),
                name=_spmv_mkl_helper_name(fmt),
                parameters=_SPMV_MKL_PARAMS,
                arguments=_spmv_mkl_args(fmt, i),
            )
            for i in range(fmt.nsegments)
        ]

    def emit_call(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VDIA, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class MKLDIASpmm(_BaseVDIAKernel, SpmmKernel):
    accepts = VDIA

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>"]

    def emit_helpers(self, fmt: VDIA, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VDIA, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        nrhs = rhs.shape[1]
        return [
            out_of_line(
                _spmm_mkl_body(fmt, y, x),
                name=_spmm_mkl_helper_name(fmt),
                parameters=_SPMM_MKL_PARAMS,
                arguments=_spmm_mkl_args(fmt, i, nrhs),
            )
            for i in range(fmt.nsegments)
        ]

    def emit_call(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VDIA, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


