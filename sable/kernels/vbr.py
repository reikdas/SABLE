from __future__ import annotations

from sable.build_config import MKL_FLAGS
from sable.codegen import OutOfLineCode, out_of_line
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


def _vbr_blocks(fmt: VBR):
    # VBR stores only packed dense blocks; bpntrb is a CSR-style row pointer, so
    # row `block_row`'s packed blocks are bindx[bpntrb[x]:bpntrb[x+1]] and the
    # k-th packed block reads its values at val offset indx[k].
    for block_row in range(len(fmt.rpntr) - 1):
        for k in range(fmt.bpntrb[block_row], fmt.bpntrb[block_row + 1]):
            block_col = fmt.bindx[k]
            yield (
                fmt.rpntr[block_row],
                fmt.rpntr[block_row + 1],
                fmt.cpntr[block_col],
                fmt.cpntr[block_col + 1],
                fmt.indx[k],
            )


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
    return _emit_spmm_naive_block_loop(fmt, y, x, r0, r1, c0, c1, offset, nrhs)


def _emit_spmm_naive_block_loop(fmt: VBR, y: str, x: str, r0, r1, c0, c1, offset, nrhs) -> str:
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


def _vbr_loop_helper_name(fmt: VBR, operation: str, mode: str) -> str:
    return f"{fmt.val}_{operation}_{mode}_block"


def _vbr_loop_parameters(operation: str) -> list[str]:
    parameters = ["int r0", "int r1", "int c0", "int c1", "int offset"]
    if operation == "spmm":
        parameters.append("int nrhs")
    return parameters


def _vbr_loop_arguments(operation: str, block: tuple[int, int, int, int, int], rhs) -> list[int]:
    arguments = list(block)
    if operation == "spmm":
        arguments.append(rhs.shape[1])
    return arguments


def _vbr_part_modes(fmt: VBR, mode: str) -> list[tuple[str, tuple[int, int, int, int, int]]]:
    parts = []
    for block in _vbr_blocks(fmt):
        r0, r1, c0, c1, _ = block
        if mode == "mixed":
            part_mode = "mkl" if _should_use_mkl_for_block(r1 - r0, c1 - c0) else "naive"
        else:
            part_mode = mode
        parts.append((part_mode, block))
    return parts


def _emit_vbr_part(
    fmt: VBR,
    y: str,
    x: str,
    rhs,
    operation: str,
    part_mode: str,
    block: tuple[int, int, int, int, int],
) -> str:
    r0, r1, c0, c1, offset = block
    if operation == "spmv":
        if part_mode == "mkl":
            return _emit_spmv_mkl_block(fmt, y, x, r0, r1, c0, c1, offset)
        return _emit_spmv_naive_block(fmt, y, x, r0, r1, c0, c1, offset)
    if part_mode == "mkl":
        return _emit_spmm_mkl_block(fmt, y, x, rhs, r0, r1, c0, c1, offset)
    return _emit_spmm_naive_block(fmt, y, x, rhs, r0, r1, c0, c1, offset)


def _emit_vbr_loop_part(fmt: VBR, y: str, x: str, operation: str) -> str:
    if operation == "spmv":
        return _emit_spmv_naive_block(fmt, y, x, "r0", "r1", "c0", "c1", "offset")
    return _emit_spmm_naive_block_loop(fmt, y, x, "r0", "r1", "c0", "c1", "offset", "nrhs")


def _emit_vbr_timed_calls(
    fmt: VBR,
    y: str,
    x: str,
    rhs,
    operation: str,
    mode: str,
) -> list[str | OutOfLineCode]:
    calls = []
    for part_mode, block in _vbr_part_modes(fmt, mode):
        if part_mode == "mkl":
            calls.append(_emit_vbr_part(fmt, y, x, rhs, operation, part_mode, block))
            continue
        body = _emit_vbr_loop_part(fmt, y, x, operation)
        calls.append(
            out_of_line(
                body,
                name=_vbr_loop_helper_name(fmt, operation, part_mode),
                parameters=_vbr_loop_parameters(operation),
                arguments=_vbr_loop_arguments(operation, block, rhs),
            )
        )
    return calls


class _BaseVBRKernel:
    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""


class NaiveVBRSpmv(_BaseVBRKernel, SpmvKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return []

    def emit_helpers(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return _emit_vbr_timed_calls(fmt, y, x, rhs, "spmv", "naive")

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


class MixedVBRSpmv(_BaseVBRKernel, SpmvKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_cblas.h>"]

    def emit_helpers(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return _emit_vbr_timed_calls(fmt, y, x, rhs, "spmv", "mixed")

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class MKLVBRSpmv(_BaseVBRKernel, SpmvKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_cblas.h>"]

    def emit_helpers(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return _emit_vbr_timed_calls(fmt, y, x, rhs, "spmv", "mkl")

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class NaiveVBRSpmm(_BaseVBRKernel, SpmmKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return []

    def emit_helpers(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return _emit_vbr_timed_calls(fmt, y, x, rhs, "spmm", "naive")

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


class MixedVBRSpmm(_BaseVBRKernel, SpmmKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_cblas.h>"]

    def emit_helpers(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return _emit_vbr_timed_calls(fmt, y, x, rhs, "spmm", "mixed")

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class MKLVBRSpmm(_BaseVBRKernel, SpmmKernel):
    accepts = VBR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_cblas.h>"]

    def emit_helpers(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_setup(self, fmt: VBR, rhs) -> str:
        return ""

    def emit_timed_calls(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return _emit_vbr_timed_calls(fmt, y, x, rhs, "spmm", "mkl")

    def emit_call(self, fmt: VBR, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VBR, rhs) -> str:
        return ""

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}
