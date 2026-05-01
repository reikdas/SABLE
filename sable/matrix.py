from __future__ import annotations

import pathlib
from dataclasses import dataclass

import scipy.io
import scipy.sparse

from .formats import CSR, Format, VBR, VDIA


def _as_csr(data) -> scipy.sparse.csr_matrix:
    if isinstance(data, (str, pathlib.Path)):
        return scipy.io.mmread(str(data)).tocsr()
    return scipy.sparse.csr_matrix(data)


@dataclass
class ResidualMatrix:
    _data: scipy.sparse.csr_matrix
    name: str = "matrix"

    @property
    def nrows(self) -> int:
        return int(self._data.shape[0])

    @property
    def ncols(self) -> int:
        return int(self._data.shape[1])

    @property
    def nnz(self) -> int:
        return int(self._data.nnz)

    def to_scipy(self) -> scipy.sparse.csr_matrix:
        return self._data.copy()

    def to_csr(self) -> scipy.sparse.csr_matrix:
        return self.to_scipy().tocsr()

    def without(self, fmt: Format) -> "ResidualMatrix":
        if isinstance(fmt, VBR):
            reduced = self._data.tolil(copy=True)
            for r0, r1, c0, c1 in fmt.blocks:
                reduced[r0:r1, c0:c1] = 0
            csr = reduced.tocsr()
            csr.eliminate_zeros()
            return ResidualMatrix(csr, name=self.name)
        if isinstance(fmt, CSR):
            return self.empty_residual()
        if isinstance(fmt, VDIA):
            reduced = self._data.tolil(copy=True)
            for region_idx, diag_offset in enumerate(fmt.diag_offsets):
                seg_start = fmt.region_ptr[region_idx]
                seg_end = fmt.region_ptr[region_idx + 1]
                for segment_idx in range(seg_start, seg_end):
                    lower = fmt.lower_bw[segment_idx]
                    upper = fmt.upper_bw[segment_idx]
                    for row in range(fmt.row_start[segment_idx], fmt.row_end[segment_idx]):
                        col_start = max(0, row + diag_offset - lower)
                        col_end = min(self.ncols, row + diag_offset + upper + 1)
                        if col_start < col_end:
                            reduced[row, col_start:col_end] = 0
            csr = reduced.tocsr()
            csr.eliminate_zeros()
            return ResidualMatrix(csr, name=self.name)
        raise NotImplementedError(f"Residual removal is not implemented for {type(fmt).__name__}")

    def empty_residual(self) -> "ResidualMatrix":
        empty = scipy.sparse.csr_matrix(self._data.shape, dtype=self._data.dtype)
        return ResidualMatrix(empty, name=self.name)


class Matrix(ResidualMatrix):
    def __init__(self, source, name: str | None = None):
        data = _as_csr(source)
        if name is None and isinstance(source, (str, pathlib.Path)):
            name = pathlib.Path(source).stem
        super().__init__(data, name=name or "matrix")
