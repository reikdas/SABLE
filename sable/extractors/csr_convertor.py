from __future__ import annotations

import time
from typing import Any

from sable.formats import CSR, Rep
from sable.matrix import ResidualMatrix


class CSRConvertor:
    produces = CSR

    def __init__(self):
        # Phase timings of the most recent extract() call, for inspection
        # benchmarking. Keys: convert_seconds, nnz.
        self.last_timing: dict[str, Any] | None = None

    def extract(self, A: ResidualMatrix) -> tuple[CSR, ResidualMatrix]:
        t0 = time.perf_counter()
        csr = A.to_csr()
        fmt = CSR(
            nrows=A.nrows,
            ncols=A.ncols,
            nnz=int(csr.nnz),
            indptr=Rep(csr.indptr.astype(int).tolist(), label="csr_indptr"),
            indices=Rep(csr.indices.astype(int).tolist(), label="csr_indices"),
            values=Rep(csr.data.astype(float).tolist(), label="csr_val"),
        )
        t1 = time.perf_counter()
        self.last_timing = {"convert_seconds": t1 - t0, "nnz": int(csr.nnz)}
        return fmt, A.empty_residual()
