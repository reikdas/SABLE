from __future__ import annotations

from sable.operation import Operation


class SpmvKernel:
    operation = Operation.SPMV


class SpmmKernel:
    operation = Operation.SPMM
