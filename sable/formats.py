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
    indx: list[int]
    bindx: list[int]
    rpntr: list[int]
    cpntr: list[int]
    bpntrb: list[int]
    bpntre: list[int]
    ublocks: list[int]
    blocks: list[tuple[int, int, int, int]]


@dataclass
class VDIA(Format):
    nregions: int
    nsegments: int
    region_ptr: list[int]
    diag_offsets: list[int]
    row_start: list[int]
    row_end: list[int]
    lower_bw: list[int]
    upper_bw: list[int]
    data_ptr: list[int]
    data: Rep[float]
