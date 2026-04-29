from __future__ import annotations

from enum import Enum


class Operation(str, Enum):
    SPMV = "spmv"
    SPMM = "spmm"


OPERATION_RHS_RANK = {
    Operation.SPMV: 1,
    Operation.SPMM: 2,
}


def rhs_rank_for_operation(operation: Operation | str) -> int:
    return OPERATION_RHS_RANK[Operation(operation)]


def kernel_operation(kernel: object) -> Operation | None:
    operation = getattr(kernel, "operation", None)
    if operation is not None:
        return Operation(operation)
    return None
