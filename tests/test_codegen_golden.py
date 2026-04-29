"""Golden tests for frontend C code generation.

Each test regenerates C source from a small fixture matrix and compares it
against a stored reference file in ``tests/golden/fixture``.
"""

import os
import pathlib
import re

import numpy
import pytest
import scipy.sparse

from sable import Matrix, Plan
from sable.extractors import BlockDetectorSkip, CSRConvertor
from sable.kernels import (
    MKLCSRSpmm,
    MKLCSRSpmv,
    MKLVBRSpmm,
    MKLVBRSpmv,
    MixedVBRSpmm,
    MixedVBRSpmv,
    NaiveCSRSpmm,
    NaiveCSRSpmv,
    NaiveVBRSpmm,
    NaiveVBRSpmv,
    SPRegCSRSpmm,
    SPV8CSRSpmv,
)
from sable.tensor import DenseInput, DenseLayout

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden" / "fixture"
_PATH_RE = re.compile(r'fopen\("([^"]+)"')


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


@pytest.fixture
def matrix_fixture():
    dense = numpy.array(
        [
            [1, 2, 3, 0, 0, 0],
            [4, 5, 6, 0, 0, 0],
            [7, 8, 9, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 3],
        ],
        dtype=float,
    )
    return scipy.sparse.csr_matrix(dense)


MIXED_DENSE_BLOCKS = [(0, 3, 0, 3), (3, 11, 3, 11)]


def _mixed_dense_dispatch_matrix(include_sparse_remainder: bool) -> scipy.sparse.csr_matrix:
    dense = numpy.zeros((11, 11), dtype=float)
    dense[0:3, 0:3] = numpy.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=float,
    )
    dense[3:11, 3:11] = numpy.arange(1, 65, dtype=float).reshape(8, 8)
    if include_sparse_remainder:
        dense[0, 10] = 5.0
        dense[1, 9] = 4.0
        dense[2, 8] = 3.0
        dense[8, 2] = 6.0
        dense[9, 1] = 1.0
        dense[10, 0] = 2.0
    return scipy.sparse.csr_matrix(dense)


def _normalize_c_source(source: str) -> str:
    def _replace(match: re.Match) -> str:
        path = match.group(1)
        return f'fopen("<PATH>/{os.path.basename(path)}"'

    return _PATH_RE.sub(_replace, source)


def _write_vector(path: pathlib.Path, size: int) -> None:
    path.write_text(",".join(["1.0"] * size) + "\n")


def _write_matrix(path: pathlib.Path, rows: int, cols: int) -> None:
    path.write_text(",".join(["1.0"] * (rows * cols)) + "\n")


CODEGEN_VARIANTS = {
    "spmv_naive_naive": ("split", 1, NaiveVBRSpmv, NaiveCSRSpmv),
    "spmv_naive_spv8": ("split", 1, NaiveVBRSpmv, SPV8CSRSpmv),
    "spmv_mixed_mkl": ("split", 1, MixedVBRSpmv, MKLCSRSpmv),
    "spmv_mixed_naive": ("split", 1, MixedVBRSpmv, NaiveCSRSpmv),
    "spmv_mkl_naive": ("split", 1, MKLVBRSpmv, NaiveCSRSpmv),
    "spmv_naive_fully_dense": ("fully_dense", 1, NaiveVBRSpmv, None),
    "spmv_mkl_fully_dense": ("fully_dense", 1, MKLVBRSpmv, None),
    "spmv_mixed_fully_dense": ("fully_dense", 1, MixedVBRSpmv, None),
    "spmv_naive_fully_sparse": ("fully_sparse", 1, None, NaiveCSRSpmv),
    "spmv_mkl_fully_sparse": ("fully_sparse", 1, None, MKLCSRSpmv),
    "spmv_spv8_fully_sparse": ("fully_sparse", 1, None, SPV8CSRSpmv),
    "spmm_naive_naive": ("split", 2, NaiveVBRSpmm, NaiveCSRSpmm),
    "spmm_naive_spreg": ("split", 2, NaiveVBRSpmm, SPRegCSRSpmm),
    "spmm_mixed_naive": ("split", 2, MixedVBRSpmm, NaiveCSRSpmm),
    "spmm_mixed_spreg": ("split", 2, MixedVBRSpmm, SPRegCSRSpmm),
    "spmm_mkl_naive": ("split", 2, MKLVBRSpmm, NaiveCSRSpmm),
    "spmm_mkl_spreg": ("split", 2, MKLVBRSpmm, SPRegCSRSpmm),
    "spmm_naive_fully_dense": ("fully_dense", 2, NaiveVBRSpmm, None),
    "spmm_mkl_fully_dense": ("fully_dense", 2, MKLVBRSpmm, None),
    "spmm_mixed_fully_dense": ("fully_dense", 2, MixedVBRSpmm, None),
    "spmm_naive_fully_sparse": ("fully_sparse", 2, None, NaiveCSRSpmm),
    "spmm_mkl_fully_sparse": ("fully_sparse", 2, None, MKLCSRSpmm),
    "spmm_spreg_fully_sparse": ("fully_sparse", 2, None, SPRegCSRSpmm),
}


def _generate_c(variant: str, A: scipy.sparse.csr_matrix, tmpdir: pathlib.Path) -> str:
    fixture_kind, rhs_rank, dense_kernel_type, sparse_kernel_type = CODEGEN_VARIANTS[variant]
    mixed_dense_dispatch = dense_kernel_type in {MixedVBRSpmv, MixedVBRSpmm}
    if mixed_dense_dispatch:
        A = _mixed_dense_dispatch_matrix(include_sparse_remainder=fixture_kind == "split")
    elif fixture_kind == "fully_dense":
        A = scipy.sparse.csr_matrix(
            numpy.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
                dtype=float,
            )
        )
    plan = Plan(Matrix(A, name="fixture"), artifact_dir=str(tmpdir / "codegen"))

    if rhs_rank == 1:
        rhs_path = tmpdir / "x.vector"
        _write_vector(rhs_path, A.shape[1])
        plan.rhs(DenseInput.vector(str(rhs_path), A.shape[1]))
    else:
        rhs_path = tmpdir / "x.matrix"
        _write_matrix(rhs_path, A.shape[1], 512)
        plan.rhs(
            DenseInput.matrix(
                str(rhs_path),
                shape=(A.shape[1], 512),
                layout=DenseLayout.ROW_MAJOR,
            )
        )

    if fixture_kind in {"split", "fully_dense"}:
        if mixed_dense_dispatch:
            blocks = MIXED_DENSE_BLOCKS
        else:
            blocks = [(0, 3, 0, 3)] if fixture_kind == "split" else [(0, A.shape[0], 0, A.shape[1])]
        vbr = plan.extract(BlockDetectorSkip(blocks=blocks))
        plan.dispatch(vbr, dense_kernel_type())
    if fixture_kind in {"split", "fully_sparse"}:
        csr = plan.extract(CSRConvertor())
        plan.dispatch(csr, sparse_kernel_type())

    executor = plan.compile(filename="fixture", bench=1)
    return _normalize_c_source(pathlib.Path(executor.c_path).read_text())


@pytest.mark.parametrize("variant", list(CODEGEN_VARIANTS))
def test_codegen_golden(variant, matrix_fixture, update_golden, tmp_path):
    actual = _generate_c(variant, matrix_fixture, tmp_path)
    golden_path = GOLDEN_DIR / f"{variant}.c"

    if update_golden:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual)
        pytest.skip(f"Golden file updated: {golden_path}")

    if not golden_path.exists():
        pytest.fail(f"Golden file {golden_path} does not exist. Run with --update-golden to create it.")

    expected = golden_path.read_text()
    if actual != expected:
        import difflib

        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"golden/{variant}.c",
            tofile=f"actual/{variant}.c",
        )
        pytest.fail(
            f"Codegen output for {variant!r} differs from golden file.\n"
            f"Run with --update-golden to accept the new output.\n\n"
            f"{''.join(diff)}"
        )
