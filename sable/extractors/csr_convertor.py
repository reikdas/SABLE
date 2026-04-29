from __future__ import annotations

from sable.formats import CSR, Rep
from sable.matrix import ResidualMatrix


class CSRConvertor:
    produces = CSR

    def extract(self, A: ResidualMatrix) -> tuple[CSR, ResidualMatrix]:
        csr = A.to_csr()
        fmt = CSR(
            nrows=A.nrows,
            ncols=A.ncols,
            nnz=int(csr.nnz),
            indptr=Rep(csr.indptr.astype(int).tolist(), label="csr_indptr"),
            indices=Rep(csr.indices.astype(int).tolist(), label="csr_indices"),
            values=Rep(csr.data.astype(float).tolist(), label="csr_val"),
        )
        return fmt, A.empty_residual()
