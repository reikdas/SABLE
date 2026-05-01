from __future__ import annotations

from sable.codegen import OutOfLineCode, out_of_line
from sable.formats import VDIA
from sable.kernels.base import SpmmKernel, SpmvKernel


def _empty_list() -> list[str]:
    return []


def _empty_dict() -> dict[str, str]:
    return {}


def _vdia_segments(fmt: VDIA):
    for region, diag_center in enumerate(fmt.diag_offsets):
        for segment in range(fmt.region_ptr[region], fmt.region_ptr[region + 1]):
            yield (
                diag_center,
                fmt.row_start[segment],
                fmt.row_end[segment],
                fmt.lower_bw[segment],
                fmt.upper_bw[segment],
                fmt.data_ptr[segment],
            )


def _vdia_loop_helper_name(fmt: VDIA, operation: str) -> str:
    return f"{fmt.data}_{operation}_naive_segment"


def _vdia_loop_parameters(operation: str) -> list[str]:
    parameters = [
        "int ncols",
        "int diag_center",
        "int row0",
        "int row1",
        "int lower",
        "int upper",
        "int base",
    ]
    if operation == "spmm":
        parameters.append("int nrhs")
    return parameters


def _vdia_loop_arguments(operation: str, fmt: VDIA, segment: tuple[int, int, int, int, int, int], rhs) -> list[int]:
    arguments = [fmt.ncols, *segment]
    if operation == "spmm":
        arguments.append(rhs.shape[1])
    return arguments


def _emit_spmv_naive_vdia_loop(fmt: VDIA, y: str, x: str) -> str:
    return f"""\
int rows = row1 - row0;
int ndiags = lower + upper + 1;
int diag_start = diag_center - lower;
for (int local_diag = 0; local_diag < ndiags; local_diag++) {{
    int diag = diag_start + local_diag;
    for (int row = row0; row < row1; row++) {{
        int col = row + diag;
        if (0 <= col && col < ncols) {{
            {y}[row] += {fmt.data}[base + local_diag * rows + (row - row0)] * {x}[col];
        }}
    }}
}}
"""


def _emit_spmm_naive_vdia_loop(fmt: VDIA, y: str, x: str) -> str:
    return f"""\
int rows = row1 - row0;
int ndiags = lower + upper + 1;
int diag_start = diag_center - lower;
for (int local_diag = 0; local_diag < ndiags; local_diag++) {{
    int diag = diag_start + local_diag;
    for (int row = row0; row < row1; row++) {{
        int col = row + diag;
        if (0 <= col && col < ncols) {{
            double a = {fmt.data}[base + local_diag * rows + (row - row0)];
            for (int rhs_col = 0; rhs_col < nrhs; rhs_col++) {{
                {y}[row * nrhs + rhs_col] += a * {x}[col * nrhs + rhs_col];
            }}
        }}
    }}
}}
"""


def _emit_vdia_timed_calls(fmt: VDIA, y: str, x: str, rhs, operation: str) -> list[OutOfLineCode]:
    if operation == "spmv":
        body = _emit_spmv_naive_vdia_loop(fmt, y, x)
    else:
        body = _emit_spmm_naive_vdia_loop(fmt, y, x)
    return [
        out_of_line(
            body,
            name=_vdia_loop_helper_name(fmt, operation),
            parameters=_vdia_loop_parameters(operation),
            arguments=_vdia_loop_arguments(operation, fmt, segment, rhs),
        )
        for segment in _vdia_segments(fmt)
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
        return _emit_vdia_timed_calls(fmt, y, x, rhs, "spmv")

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
        return _emit_vdia_timed_calls(fmt, y, x, rhs, "spmm")

    def emit_call(self, fmt: VDIA, y: str, x: str, rhs) -> list[str | OutOfLineCode]:
        return self.emit_timed_calls(fmt, y, x, rhs)

    def emit_teardown(self, fmt: VDIA, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)
