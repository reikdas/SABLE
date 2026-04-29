from __future__ import annotations

from sable.build_config import MKL_FLAGS
from sable.formats import VBR
from sable.kernels.base import SpmmKernel, SpmvKernel


def _mkl_compile_flags() -> list[str]:
    return [flag for flag in MKL_FLAGS if flag.startswith("-I")]


def _mkl_link_flags() -> list[str]:
    return [flag for flag in MKL_FLAGS if not flag.startswith("-I")]


def _empty_list() -> list[str]:
    return []


def _empty_dict() -> dict[str, str]:
    return {}


def _should_use_mkl_for_block(rows: int, cols: int) -> bool:
    min_dim = min(rows, cols)
    max_dim = max(rows, cols)
    if min_dim < 8:
        return False
    return max_dim / min_dim <= 100


def _dense_blocks(fmt: VBR):
    dense_count = 0
    nnz_block = 0
    sparse_blocks = set(fmt.ublocks.values)
    for block_row in range(len(fmt.rpntr.values) - 1):
        start = fmt.bpntrb.values[block_row]
        end = fmt.bpntre.values[block_row]
        if start == -1:
            continue
        valid_cols = fmt.bindx.values[start:end]
        for block_col in range(len(fmt.cpntr.values) - 1):
            if block_col not in valid_cols:
                continue
            if nnz_block not in sparse_blocks:
                yield (
                    fmt.rpntr.values[block_row],
                    fmt.rpntr.values[block_row + 1],
                    fmt.cpntr.values[block_col],
                    fmt.cpntr.values[block_col + 1],
                    fmt.indx.values[dense_count],
                )
                dense_count += 1
            nnz_block += 1


def _emit_spmv_naive_block(fmt: VBR, y: str, x: str, r0: int, r1: int, c0: int, c1: int, offset: int) -> str:
    return f"""\
for (int j = {c0}; j < {c1}; j++) {{
    for (int i = {r0}; i < {r1}; i++) {{
        {y}[i] += {fmt.val}[{offset} + (j - {c0}) * ({r1} - {r0}) + (i - {r0})] * {x}[j];
    }}
}}
"""


def _emit_spmv_mkl_block(fmt: VBR, y: str, x: str, r0: int, r1: int, c0: int, c1: int, offset: int) -> str:
    return f"""\
cblas_dgemv(CblasColMajor, CblasNoTrans,
    {r1} - {r0}, {c1} - {c0},
    1.0,
    &{fmt.val}[{offset}], {r1} - {r0},
    &{x}[{c0}], 1,
    1.0,
    &{y}[{r0}], 1);
"""


def _emit_spmm_naive_block(fmt: VBR, y: str, x: str, rhs, r0: int, r1: int, c0: int, c1: int, offset: int) -> str:
    nrhs = rhs.shape[1]
    return f"""\
for (int i = {r0}; i < {r1}; i++) {{
    for (int j = {c0}; j < {c1}; j++) {{
        double a = {fmt.val}[{offset} + (j - {c0}) * ({r1} - {r0}) + (i - {r0})];
        for (int k = 0; k < {nrhs}; k++) {{
            {y}[i * {nrhs} + k] += a * {x}[j * {nrhs} + k];
        }}
    }}
}}
"""


def _emit_spmm_mkl_block(fmt: VBR, y: str, x: str, rhs, r0: int, r1: int, c0: int, c1: int, offset: int) -> str:
    nrhs = rhs.shape[1]
    return f"""\
cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    {r1} - {r0}, {nrhs}, {c1} - {c0},
    1.0,
    &{fmt.val}[{offset}], {r1} - {r0},
    &{x}[{c0} * {nrhs}], {nrhs},
    1.0,
    &{y}[{r0} * {nrhs}], {nrhs});
"""


class _BaseVBRKernel:
    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""


class NaiveVBRSpmv(SpmvKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return []

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str]:
        return [_emit_spmv_naive_block(fmt, y, x, *block) for block in _dense_blocks(fmt)]

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> str:
        return "".join(self.emit_timed_calls(fmt, y, x, rhs))

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


class MixedVBRSpmv(SpmvKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_cblas.h>"]

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str]:
        calls = []
        for block in _dense_blocks(fmt):
            r0, r1, c0, c1, offset = block
            if _should_use_mkl_for_block(r1 - r0, c1 - c0):
                calls.append(_emit_spmv_mkl_block(fmt, y, x, r0, r1, c0, c1, offset))
            else:
                calls.append(_emit_spmv_naive_block(fmt, y, x, r0, r1, c0, c1, offset))
        return calls

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> str:
        return "".join(self.emit_timed_calls(fmt, y, x, rhs))

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class MKLVBRSpmv(SpmvKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_cblas.h>"]

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str]:
        return [_emit_spmv_mkl_block(fmt, y, x, *block) for block in _dense_blocks(fmt)]

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> str:
        return "".join(self.emit_timed_calls(fmt, y, x, rhs))

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class NaiveVBRSpmm(SpmmKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return []

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str]:
        return [_emit_spmm_naive_block(fmt, y, x, rhs, *block) for block in _dense_blocks(fmt)]

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> str:
        return "".join(self.emit_timed_calls(fmt, y, x, rhs))

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


class MixedVBRSpmm(SpmmKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_cblas.h>"]

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str]:
        calls = []
        for block in _dense_blocks(fmt):
            r0, r1, c0, c1, offset = block
            if _should_use_mkl_for_block(r1 - r0, c1 - c0):
                calls.append(_emit_spmm_mkl_block(fmt, y, x, rhs, r0, r1, c0, c1, offset))
            else:
                calls.append(_emit_spmm_naive_block(fmt, y, x, rhs, r0, r1, c0, c1, offset))
        return calls

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> str:
        return "".join(self.emit_timed_calls(fmt, y, x, rhs))

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class MKLVBRSpmm(SpmmKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_cblas.h>"]

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str]:
        return [_emit_spmm_mkl_block(fmt, y, x, rhs, *block) for block in _dense_blocks(fmt)]

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> str:
        return "".join(self.emit_timed_calls(fmt, y, x, rhs))

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}
