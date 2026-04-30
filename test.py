import os
import pathlib
import subprocess

import numpy
import pytest
import scipy
import scipy.sparse

from sable.build_config import MKL_AVAILABLE as MKL_AVAILABLE
from sable import Matrix, Plan
from sable.extractors import BlockDetector, BlockDetectorSkip, CSRConvertor
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
    UZPCSRSpmv,
)
from sable.tensor import DenseInput, DenseLayout

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)
from utils.fileio import write_dense_matrix, write_dense_vector


def _generated_vector_path(size: int) -> str:
    return os.path.abspath(os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{size}.vector"))


def _generated_matrix_path(rows: int, cols: int) -> str:
    return os.path.abspath(os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{rows}x{cols}.matrix"))


def _numeric_result_lines(output):
    result_lines = []
    skip_timing = True
    for line in output:
        line = line.strip()
        if not line:
            skip_timing = False
            continue
        if skip_timing:
            continue
        if line.startswith("Sparse:") or line.startswith("Dense:") or line.startswith("Dense Block"):
            continue
        try:
            float(line)
            result_lines.append(line)
        except ValueError:
            pass
    return result_lines


MIXED_DENSE_BLOCKS = [(0, 3, 0, 3), (3, 11, 3, 11)]
MIXED_DENSE_NNZ = 9 + 64
MIXED_SPARSE_REMAINDER_NNZ = 6


def _mixed_dense_dispatch_csr(include_sparse_remainder: bool = False):
    dense = numpy.zeros((11, 11), dtype=float)
    dense[0:3, 0:3] = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    dense[3:11, 3:11] = numpy.arange(1.0, 65.0).reshape(8, 8)
    if include_sparse_remainder:
        dense[0, 10] = 5.0
        dense[1, 9] = 4.0
        dense[2, 8] = 3.0
        dense[8, 2] = 6.0
        dense[9, 1] = 1.0
        dense[10, 0] = 2.0
    return scipy.sparse.csr_matrix(dense)


def _extract_mixed_dense_vbr(plan: Plan):
    vbr = plan.extract(BlockDetectorSkip(blocks=MIXED_DENSE_BLOCKS))
    assert len(vbr.val.values) == MIXED_DENSE_NNZ
    return vbr


def _check_avx512_support():
    """Check if AVX512 is supported on this CPU."""
    result = subprocess.run(
        ["grep", "-c", "avx512", "/proc/cpuinfo"],
        capture_output=True, text=True
    )
    return result.returncode == 0 and int(result.stdout.strip()) > 0


def _require_uzp_toolchain():
    if not os.path.isfile(os.path.join(BASE_PATH, "uzp_prepare.sh")):
        pytest.skip("UZP prepare script not available")
    if not os.path.isdir(os.path.join(BASE_PATH, "uzp-artifact")):
        pytest.skip("UZP artifact not available")


# ---------------------------------------------------------------------------
# SpMV — frontend single-threaded dispatches
# ---------------------------------------------------------------------------

def test_spmv_single_threaded_naive_fully_dense():
    """Test SpMV with handwritten VBR dense dispatch and no sparse remainder."""
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmv_single_threaded_naive_fully_dense"
    matrix = Matrix(scipy.sparse.csr_matrix(dense), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) == dense.size
    assert plan.residual.nnz == 0
    plan.dispatch(vbr, NaiveVBRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = dense.dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_naive_fully_sparse():
    """Test SpMV with handwritten CSR sparse dispatch and no dense blocks."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_naive_fully_sparse"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    csr = plan.extract(CSRConvertor())
    assert plan.residual.nnz == 0
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_naive_naive():
    """Test SpMV with handwritten VBR dense dispatch + handwritten CSR sparse dispatch."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_naive_naive"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, NaiveVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_naive_mkl():
    """Test SpMV with handwritten VBR dense dispatch + MKL CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_naive_mkl"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, NaiveVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, MKLCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_naive_spv8():
    """Test SpMV with handwritten VBR dense dispatch + SPV8 CSR sparse dispatch."""
    if not os.path.isfile(os.path.join(BASE_PATH, "spv8-public", "bin", "spmv_spv8.o")):
        pytest.skip("SPV8 object not built")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_naive_spv8"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, NaiveVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, SPV8CSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mkl_fully_dense():
    """Test SpMV with MKL VBR dense dispatch and no sparse remainder."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmv_single_threaded_mkl_fully_dense"
    matrix = Matrix(scipy.sparse.csr_matrix(dense), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) == dense.size
    assert plan.residual.nnz == 0
    plan.dispatch(vbr, MKLVBRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = dense.dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mkl_fully_sparse():
    """Test SpMV with MKL CSR sparse dispatch and no dense blocks."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_mkl_fully_sparse"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    csr = plan.extract(CSRConvertor())
    assert plan.residual.nnz == 0
    plan.dispatch(csr, MKLCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_spv8_fully_sparse():
    """Test SpMV with SPV8 CSR sparse dispatch and no dense blocks."""
    if not os.path.isfile(os.path.join(BASE_PATH, "spv8-public", "bin", "spmv_spv8.o")):
        pytest.skip("SPV8 object not built")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_spv8_fully_sparse"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    csr = plan.extract(CSRConvertor())
    assert plan.residual.nnz == 0
    plan.dispatch(csr, SPV8CSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_uzp_fully_sparse():
    """Test SpMV with UZP CSR sparse dispatch and no dense blocks."""
    _require_uzp_toolchain()
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_uzp_fully_sparse"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    csr = plan.extract(CSRConvertor())
    assert plan.residual.nnz == 0
    plan.dispatch(csr, UZPCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mkl_naive():
    """Test SpMV with MKL VBR dense dispatch + handwritten CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_mkl_naive"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, MKLVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mkl_mkl():
    """Test SpMV with MKL VBR dense dispatch + MKL CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_mkl_mkl"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, MKLVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, MKLCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mkl_spv8():
    """Test SpMV with MKL VBR dense dispatch + SPV8 CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    if not os.path.isfile(os.path.join(BASE_PATH, "spv8-public", "bin", "spmv_spv8.o")):
        pytest.skip("SPV8 object not built")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_mkl_spv8"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, MKLVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, SPV8CSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_naive_uzp():
    """Test SpMV with handwritten VBR dense dispatch + UZP CSR sparse dispatch."""
    _require_uzp_toolchain()
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_naive_uzp"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, NaiveVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, UZPCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mixed_fully_dense():
    """Test SpMV with mixed VBR dense dispatch and no sparse remainder."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_mixed_fully_dense"
    matrix = Matrix(_mixed_dense_dispatch_csr(), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_dense_vbr(plan)
    assert plan.residual.nnz == 0
    plan.dispatch(vbr, MixedVBRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mixed_naive():
    """Test SpMV with mixed VBR dense dispatch + handwritten CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_mixed_naive"
    matrix = Matrix(_mixed_dense_dispatch_csr(include_sparse_remainder=True), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_dense_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmv())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_SPARSE_REMAINDER_NNZ
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mixed_mkl():
    """Test SpMV with mixed VBR dense dispatch + MKL CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_mixed_mkl"
    matrix = Matrix(_mixed_dense_dispatch_csr(include_sparse_remainder=True), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_dense_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmv())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_SPARSE_REMAINDER_NNZ
    plan.dispatch(csr, MKLCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mixed_spv8():
    """Test SpMV with mixed VBR dense dispatch + SPV8 CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    if not os.path.isfile(os.path.join(BASE_PATH, "spv8-public", "bin", "spmv_spv8.o")):
        pytest.skip("SPV8 object not built")
    filename = "spmv_single_threaded_mixed_spv8"
    matrix = Matrix(_mixed_dense_dispatch_csr(include_sparse_remainder=True), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_dense_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmv())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_SPARSE_REMAINDER_NNZ
    plan.dispatch(csr, SPV8CSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_mixed_uzp():
    """Test SpMV with mixed VBR dense dispatch + UZP CSR sparse dispatch."""
    _require_uzp_toolchain()
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_mixed_uzp"
    matrix = Matrix(_mixed_dense_dispatch_csr(include_sparse_remainder=True), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_dense_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmv())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_SPARSE_REMAINDER_NNZ
    plan.dispatch(csr, UZPCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# SpMM — frontend single-threaded dispatches
# ---------------------------------------------------------------------------

def test_spmm_single_threaded_naive_fully_dense():
    """Test SpMM with handwritten VBR dense dispatch and no sparse remainder."""
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmm_single_threaded_naive_fully_dense"
    matrix = Matrix(scipy.sparse.csr_matrix(dense), name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) == dense.size
    assert plan.residual.nnz == 0
    plan.dispatch(vbr, NaiveVBRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = dense.dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_naive_fully_sparse():
    """Test SpMM with handwritten CSR sparse dispatch and no dense blocks."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_naive_fully_sparse"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    csr = plan.extract(CSRConvertor())
    assert plan.residual.nnz == 0
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_naive_naive():
    """Test SpMM with handwritten VBR dense dispatch + handwritten CSR sparse dispatch."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_naive_naive"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, NaiveVBRSpmm())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_naive_mkl():
    """Test SpMM with handwritten VBR dense dispatch + MKL CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_naive_mkl"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, NaiveVBRSpmm())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, MKLCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not _check_avx512_support(),
    reason="AVX512 not supported on this CPU"
)
def test_spmm_single_threaded_naive_spreg():
    """Test SpMM with handwritten VBR dense dispatch + SPReg CSR sparse dispatch."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_naive_spreg"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, NaiveVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz > 0, "Expected sparse residual for SPReg dispatch"
    plan.dispatch(csr, SPRegCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_mkl_fully_dense():
    """Test SpMM with MKL VBR dense dispatch and no sparse remainder."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmm_single_threaded_mkl_fully_dense"
    matrix = Matrix(scipy.sparse.csr_matrix(dense), name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) == dense.size
    assert plan.residual.nnz == 0
    plan.dispatch(vbr, MKLVBRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = dense.dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_mkl_fully_sparse():
    """Test SpMM with MKL CSR sparse dispatch and no dense blocks."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_mkl_fully_sparse"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    csr = plan.extract(CSRConvertor())
    assert plan.residual.nnz == 0
    plan.dispatch(csr, MKLCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not _check_avx512_support(),
    reason="AVX512 not supported on this CPU"
)
def test_spmm_single_threaded_spreg_fully_sparse():
    """Test SpMM with SPReg CSR sparse dispatch and no dense blocks."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_spreg_fully_sparse"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    csr = plan.extract(CSRConvertor())
    assert plan.residual.nnz == 0
    plan.dispatch(csr, SPRegCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_mkl_naive():
    """Test SpMM with MKL VBR dense dispatch + handwritten CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_mkl_naive"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, MKLVBRSpmm())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_mkl_mkl():
    """Test SpMM with MKL VBR dense dispatch + MKL CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_mkl_mkl"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, MKLVBRSpmm())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, MKLCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not _check_avx512_support(),
    reason="AVX512 not supported on this CPU"
)
def test_spmm_single_threaded_mkl_spreg():
    """Test SpMM with MKL VBR dense dispatch + SPReg CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_mkl_spreg"
    matrix = Matrix(mtx_path, name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a dense block"
    plan.dispatch(vbr, MKLVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz > 0, "Expected sparse residual for SPReg dispatch"
    plan.dispatch(csr, SPRegCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_mixed_fully_dense():
    """Test SpMM with mixed VBR dense dispatch and no sparse remainder."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_mixed_fully_dense"
    matrix = Matrix(_mixed_dense_dispatch_csr(), name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = _extract_mixed_dense_vbr(plan)
    assert plan.residual.nnz == 0
    plan.dispatch(vbr, MixedVBRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_mixed_naive():
    """Test SpMM with mixed VBR dense dispatch + handwritten CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_mixed_naive"
    matrix = Matrix(_mixed_dense_dispatch_csr(include_sparse_remainder=True), name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = _extract_mixed_dense_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_SPARSE_REMAINDER_NNZ
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_mixed_mkl():
    """Test SpMM with mixed VBR dense dispatch + MKL CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_mixed_mkl"
    matrix = Matrix(_mixed_dense_dispatch_csr(include_sparse_remainder=True), name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = _extract_mixed_dense_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_SPARSE_REMAINDER_NNZ
    plan.dispatch(csr, MKLCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not _check_avx512_support(),
    reason="AVX512 not supported on this CPU"
)
def test_spmm_single_threaded_mixed_spreg():
    """Test SpMM with mixed VBR dense dispatch + SPReg CSR sparse dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_mixed_spreg"
    matrix = Matrix(_mixed_dense_dispatch_csr(include_sparse_remainder=True), name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(
        DenseInput.matrix(
            _generated_matrix_path(cols, 512),
            shape=(cols, 512),
            layout=DenseLayout.ROW_MAJOR,
        )
    )
    vbr = _extract_mixed_dense_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_SPARSE_REMAINDER_NNZ, "Expected sparse residual for SPReg dispatch"
    plan.dispatch(csr, SPRegCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)
