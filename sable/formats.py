from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class Rep(Generic[T]):
    values: list[T]
    label: str | None = None
    c_name: str | None = None

    def __str__(self) -> str:
        return self.c_name or self.label or ""


@dataclass
class Format:
    nrows: int
    ncols: int


@dataclass
class CSR(Format):
    nnz: int
    indptr: Rep[int]
    indices: Rep[int]
    values: Rep[float]


@dataclass
class VBR(Format):
    val: Rep[float]
    indx: Rep[int]
    bindx: Rep[int]
    rpntr: Rep[int]
    cpntr: Rep[int]
    bpntrb: Rep[int]
    bpntre: Rep[int]
    # Block metadata used to skip entries handled by sparse residual kernels.
    ublocks: Rep[int]
    blocks: list[tuple[int, int, int, int]]


@dataclass
class VDIA(Format):
    nregions: int
    nsegments: int
    region_ptr: Rep[int]
    diag_offsets: Rep[int]
    row_start: Rep[int]
    row_end: Rep[int]
    lower_bw: Rep[int]
    upper_bw: Rep[int]
    data_ptr: Rep[int]
    data: Rep[float]
