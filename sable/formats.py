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
    blocks: list[tuple[int, int, int, int]]


@dataclass
class VDIA(Format):
    nsegments: int
    seg_row_start: list[int]
    seg_nrows: list[int]
    seg_ndiags: list[int]
    seg_val_ptr: list[int]
    seg_idiag_ptr: list[int]
    val: Rep[float]
    idiag: Rep[int]
