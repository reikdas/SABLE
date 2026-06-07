import os
import pathlib

import numpy
import pytest
import scipy
import scipy.sparse

from sable.build_config import MKL_AVAILABLE as MKL_AVAILABLE
from sable import Matrix, Plan
from sable.extractors import BandExtractorSkip, BlockDetector, BlockDetectorSkip, CSRConvertor
from sable.kernels import (
    MKLCSRSpmm,
    MKLCSRSpmv,
    MKLDIASpmm,
    MKLDIASpmv,
    MKLVBRSpmm,
    MKLVBRSpmv,
    MixedVBRSpmm,
    MixedVBRSpmv,
    NaiveCSRSpmm,
    NaiveCSRSpmv,
    NaiveVDIASpmm,
    NaiveVDIASpmv,
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
        if line.startswith("Dispatch"):
            continue
        try:
            float(line)
            result_lines.append(line)
        except ValueError:
            pass
    return result_lines


MIXED_VBR_BLOCKS = [(0, 3, 0, 3), (3, 11, 3, 11)]
MIXED_VBR_NNZ = 9 + 64
MIXED_CSR_RESIDUAL_NNZ = 6


def _fully_dense_band(rows: int, cols: int):
    return [
        {
            "diag_offset": 0,
            "segments": [
                {
                    "rows": [0, rows],
                    "bandwidth": [max(rows - 1, 0), max(cols - 1, 0)],
                }
            ],
        }
    ]


def _tridiag_band(n: int):
    return [
        {
            "diag_offset": 0,
            "segments": [
                {
                    "rows": [0, n],
                    "bandwidth": [1, 1],
                }
            ],
        }
    ]


def _partial_band_matrix():
    """11x11 matrix: tridiagonal band in rows [0,6), random elsewhere."""
    values = numpy.zeros((11, 11), dtype=float)
    for i in range(6):
        for delta in [-1, 0, 1]:
            col = i + delta
            if 0 <= col < 11:
                values[i, col] = (i + 1) * 1.0 + delta * 0.5
    values[6, 0] = 3.0
    values[7, 10] = 2.0
    values[8, 3] = 4.0
    values[9, 5] = 1.0
    values[10, 7] = 5.0
    return scipy.sparse.csr_matrix(values)


def _partial_band():
    return [
        {
            "diag_offset": 0,
            "segments": [
                {
                    "rows": [0, 6],
                    "bandwidth": [1, 1],
                }
            ],
        }
    ]


def _mixed_vbr_dispatch_csr(include_csr_residual: bool = False):
    values = numpy.zeros((11, 11), dtype=float)
    values[0:3, 0:3] = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    values[3:11, 3:11] = numpy.arange(1.0, 65.0).reshape(8, 8)
    if include_csr_residual:
        values[0, 10] = 5.0
        values[1, 9] = 4.0
        values[2, 8] = 3.0
        values[8, 2] = 6.0
        values[9, 1] = 1.0
        values[10, 0] = 2.0
    return scipy.sparse.csr_matrix(values)


def _extract_mixed_vbr(plan: Plan):
    vbr = plan.extract(BlockDetectorSkip(blocks=MIXED_VBR_BLOCKS))
    assert len(vbr.val.values) == MIXED_VBR_NNZ
    return vbr


def _cpu_flags() -> set[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("flags"):
                    _, flags = line.split(":", 1)
                    return set(flags.split())
    except OSError:
        return set()
    return set()


def _has_cpu_flags(*flags: str) -> bool:
    cpu_flags = _cpu_flags()
    return all(flag in cpu_flags for flag in flags)


def _check_avx512_support():
    """Check if AVX512 is supported on this CPU."""
    return _has_cpu_flags("avx512f")


def _require_spv8_toolchain():
    if not os.path.isfile(os.path.join(BASE_PATH, "spv8-public", "bin", "spmv_spv8.o")):
        pytest.skip("SPV8 object not built")
    required_flags = ("avx512f", "avx512vl", "fma")
    cpu_flags = _cpu_flags()
    missing_flags = [flag for flag in required_flags if flag not in cpu_flags]
    if missing_flags:
        pytest.skip(f"SPV8 requires CPU flags: {', '.join(missing_flags)}")


def _require_uzp_toolchain():
    if not os.path.isfile(os.path.join(BASE_PATH, "uzp_prepare.sh")):
        pytest.skip("UZP prepare script not available")
    if not os.path.isdir(os.path.join(BASE_PATH, "uzp-artifact")):
        pytest.skip("UZP artifact not available")


# ---------------------------------------------------------------------------
# SpMV — frontend single-threaded dispatches
# ---------------------------------------------------------------------------

def test_spmv_single_threaded_blocknaive_fullydense():
    """Test SpMV with handwritten VBR VBR dispatch and no residual."""
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmv_single_threaded_blocknaive_fullydense"
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


def test_spmv_single_threaded_bandnaive_fullydense():
    """Test SpMV with naive VDIA band dispatch and no residual."""
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmv_single_threaded_bandnaive_fullydense"
    matrix = Matrix(scipy.sparse.csr_matrix(dense), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vdia = plan.extract(BandExtractorSkip(bands=_fully_dense_band(matrix.nrows, matrix.ncols)))
    assert vdia.nsegments == 1
    assert plan.residual.nnz == 0
    plan.dispatch(vdia, NaiveVDIASpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = dense.dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_bandmkl_fullydense():
    """Test SpMV with MKL DIA band dispatch and no residual."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmv_single_threaded_bandmkl_fullydense"
    matrix = Matrix(scipy.sparse.csr_matrix(dense), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vdia = plan.extract(BandExtractorSkip(bands=_fully_dense_band(matrix.nrows, matrix.ncols)))
    assert vdia.nsegments == 1
    assert plan.residual.nnz == 0
    plan.dispatch(vdia, MKLDIASpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = dense.dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_bandnaive_naive():
    """Test SpMV with naive VDIA band dispatch + naive CSR dispatch."""
    filename = "spmv_single_threaded_bandnaive_naive"
    matrix = Matrix(_partial_band_matrix(), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vdia = plan.extract(BandExtractorSkip(bands=_partial_band()))
    assert vdia.nsegments == 1
    assert plan.residual.nnz > 0
    plan.dispatch(vdia, NaiveVDIASpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_bandmkl_naive():
    """Test SpMV with MKL DIA band dispatch + naive CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_bandmkl_naive"
    matrix = Matrix(_partial_band_matrix(), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vdia = plan.extract(BandExtractorSkip(bands=_partial_band()))
    assert vdia.nsegments == 1
    assert plan.residual.nnz > 0
    plan.dispatch(vdia, MKLDIASpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_naive_csr_only():
    """Test SpMV with handwritten CSR dispatch and no VBR/VDIA regions."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_naive_csr_only"
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


def test_spmv_single_threaded_blocknaive_naive():
    """Test SpMV with handwritten VBR VBR dispatch + handwritten CSR dispatch."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_blocknaive_naive"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, NaiveVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blocknaive_mkl():
    """Test SpMV with handwritten VBR VBR dispatch + MKL CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_blocknaive_mkl"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, NaiveVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, MKLCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blocknaive_spv8():
    """Test SpMV with handwritten VBR VBR dispatch + SPV8 CSR dispatch."""
    _require_spv8_toolchain()
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_blocknaive_spv8"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, NaiveVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, SPV8CSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blockmkl_fullydense():
    """Test SpMV with MKL VBR VBR dispatch and no residual."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmv_single_threaded_blockmkl_fullydense"
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


def test_spmv_single_threaded_mkl_csr_only():
    """Test SpMV with MKL CSR dispatch and no VBR/VDIA regions."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_mkl_csr_only"
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


def test_spmv_single_threaded_spv8_csr_only():
    """Test SpMV with SPV8 CSR dispatch and no VBR/VDIA regions."""
    _require_spv8_toolchain()
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_spv8_csr_only"
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


def test_spmv_single_threaded_uzp_csr_only():
    """Test SpMV with UZP CSR dispatch and no VBR/VDIA regions."""
    _require_uzp_toolchain()
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_uzp_csr_only"
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


def test_spmv_single_threaded_blockmkl_naive():
    """Test SpMV with MKL VBR VBR dispatch + handwritten CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_blockmkl_naive"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, MKLVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blockmkl_mkl():
    """Test SpMV with MKL VBR VBR dispatch + MKL CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_blockmkl_mkl"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, MKLVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, MKLCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blockmkl_spv8():
    """Test SpMV with MKL VBR VBR dispatch + SPV8 CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    _require_spv8_toolchain()
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_blockmkl_spv8"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, MKLVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, SPV8CSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blocknaive_uzp():
    """Test SpMV with handwritten VBR VBR dispatch + UZP CSR dispatch."""
    _require_uzp_toolchain()
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmv_single_threaded_blocknaive_uzp"
    matrix = Matrix(mtx_path, name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = plan.extract(BlockDetector(min_density=0.5, min_area=9, threads=1, timeout_seconds=30))
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, NaiveVBRSpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, UZPCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blockmixed_fullydense():
    """Test SpMV with mixed VBR dispatch and no residual."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_blockmixed_fullydense"
    matrix = Matrix(_mixed_vbr_dispatch_csr(), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_vbr(plan)
    assert plan.residual.nnz == 0
    plan.dispatch(vbr, MixedVBRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blockmixed_naive():
    """Test SpMV with mixed VBR dispatch + handwritten CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_blockmixed_naive"
    matrix = Matrix(_mixed_vbr_dispatch_csr(include_csr_residual=True), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmv())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_CSR_RESIDUAL_NNZ
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blockmixed_mkl():
    """Test SpMV with mixed VBR dispatch + MKL CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_blockmixed_mkl"
    matrix = Matrix(_mixed_vbr_dispatch_csr(include_csr_residual=True), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmv())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_CSR_RESIDUAL_NNZ
    plan.dispatch(csr, MKLCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blockmixed_spv8():
    """Test SpMV with mixed VBR dispatch + SPV8 CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    _require_spv8_toolchain()
    filename = "spmv_single_threaded_blockmixed_spv8"
    matrix = Matrix(_mixed_vbr_dispatch_csr(include_csr_residual=True), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmv())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_CSR_RESIDUAL_NNZ
    plan.dispatch(csr, SPV8CSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_blockmixed_uzp():
    """Test SpMV with mixed VBR dispatch + UZP CSR dispatch."""
    _require_uzp_toolchain()
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_blockmixed_uzp"
    matrix = Matrix(_mixed_vbr_dispatch_csr(include_csr_residual=True), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vbr = _extract_mixed_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmv())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_CSR_RESIDUAL_NNZ
    plan.dispatch(csr, UZPCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# SpMM — frontend single-threaded dispatches
# ---------------------------------------------------------------------------

def test_spmm_single_threaded_blocknaive_fullydense():
    """Test SpMM with handwritten VBR VBR dispatch and no residual."""
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmm_single_threaded_blocknaive_fullydense"
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


def test_spmm_single_threaded_bandnaive_fullydense():
    """Test SpMM with naive VDIA band dispatch and no residual."""
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmm_single_threaded_bandnaive_fullydense"
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
    vdia = plan.extract(BandExtractorSkip(bands=_fully_dense_band(matrix.nrows, matrix.ncols)))
    assert vdia.nsegments == 1
    assert plan.residual.nnz == 0
    plan.dispatch(vdia, NaiveVDIASpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = dense.dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)



def test_spmm_single_threaded_bandnaive_naive():
    """Test SpMM with naive VDIA band dispatch + naive CSR dispatch."""
    filename = "spmm_single_threaded_bandnaive_naive"
    matrix = Matrix(_partial_band_matrix(), name=filename)
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
    vdia = plan.extract(BandExtractorSkip(bands=_partial_band()))
    assert vdia.nsegments == 1
    assert plan.residual.nnz > 0
    plan.dispatch(vdia, NaiveVDIASpmm())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)



def test_spmm_single_threaded_naive_csr_only():
    """Test SpMM with handwritten CSR dispatch and no VBR/VDIA regions."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_naive_csr_only"
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


def test_spmm_single_threaded_blocknaive_naive():
    """Test SpMM with handwritten VBR VBR dispatch + handwritten CSR dispatch."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_blocknaive_naive"
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
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, NaiveVBRSpmm())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_blocknaive_mkl():
    """Test SpMM with handwritten VBR VBR dispatch + MKL CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_blocknaive_mkl"
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
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
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
def test_spmm_single_threaded_blocknaive_spreg():
    """Test SpMM with handwritten VBR VBR dispatch + SPReg CSR dispatch."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_blocknaive_spreg"
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
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, NaiveVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz > 0, "Expected CSR residual for SPReg dispatch"
    plan.dispatch(csr, SPRegCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_blockmkl_fullydense():
    """Test SpMM with MKL VBR VBR dispatch and no residual."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    dense = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    filename = "spmm_single_threaded_blockmkl_fullydense"
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


def test_spmm_single_threaded_mkl_csr_only():
    """Test SpMM with MKL CSR dispatch and no VBR/VDIA regions."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_mkl_csr_only"
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
def test_spmm_single_threaded_spreg_csr_only():
    """Test SpMM with SPReg CSR dispatch and no VBR/VDIA regions."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_spreg_csr_only"
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


def test_spmm_single_threaded_blockmkl_naive():
    """Test SpMM with MKL VBR VBR dispatch + handwritten CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_blockmkl_naive"
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
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, MKLVBRSpmm())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_blockmkl_mkl():
    """Test SpMM with MKL VBR VBR dispatch + MKL CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_blockmkl_mkl"
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
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
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
def test_spmm_single_threaded_blockmkl_spreg():
    """Test SpMM with MKL VBR VBR dispatch + SPReg CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    filename = "spmm_single_threaded_blockmkl_spreg"
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
    assert len(vbr.val.values) > 0, "Expected split matrix with a VBR block"
    plan.dispatch(vbr, MKLVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz > 0, "Expected CSR residual for SPReg dispatch"
    plan.dispatch(csr, SPRegCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_blockmixed_fullydense():
    """Test SpMM with mixed VBR dispatch and no residual."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_blockmixed_fullydense"
    matrix = Matrix(_mixed_vbr_dispatch_csr(), name=filename)
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
    vbr = _extract_mixed_vbr(plan)
    assert plan.residual.nnz == 0
    plan.dispatch(vbr, MixedVBRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_blockmixed_naive():
    """Test SpMM with mixed VBR dispatch + handwritten CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_blockmixed_naive"
    matrix = Matrix(_mixed_vbr_dispatch_csr(include_csr_residual=True), name=filename)
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
    vbr = _extract_mixed_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_CSR_RESIDUAL_NNZ
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_blockmixed_mkl():
    """Test SpMM with mixed VBR dispatch + MKL CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_blockmixed_mkl"
    matrix = Matrix(_mixed_vbr_dispatch_csr(include_csr_residual=True), name=filename)
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
    vbr = _extract_mixed_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_CSR_RESIDUAL_NNZ
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
def test_spmm_single_threaded_blockmixed_spreg():
    """Test SpMM with mixed VBR dispatch + SPReg CSR dispatch."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_blockmixed_spreg"
    matrix = Matrix(_mixed_vbr_dispatch_csr(include_csr_residual=True), name=filename)
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
    vbr = _extract_mixed_vbr(plan)
    plan.dispatch(vbr, MixedVBRSpmm())
    csr = plan.extract(CSRConvertor())
    assert csr.nnz == MIXED_CSR_RESIDUAL_NNZ, "Expected CSR residual for SPReg dispatch"
    plan.dispatch(csr, SPRegCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# MKL DIA value tests: multi-segment bands (row0 != 0) and VDIA+VBR+CSR.
#
# These exercise paths the single-segment band tests above miss: mkl_ddiamv /
# mkl_ddiamm must shift idiag by each segment's row0, and mkl_ddiamm's dense
# operands are column-major (its SpMM lowering transposes the RHS once in setup
# and folds a column-major accumulator back in teardown).
# ---------------------------------------------------------------------------


def _two_segment_band_matrix():
    """6x6 with two row-disjoint band segments, so the second has row0 = 3."""
    values = numpy.zeros((6, 6), dtype=float)
    for i, j, x in [(0, 0, 1), (0, 1, 2), (1, 1, 3), (1, 2, 4), (2, 2, 5), (2, 3, 6),
                    (3, 2, 7), (3, 3, 8), (4, 3, 9), (4, 4, 10), (5, 4, 11), (5, 5, 12)]:
        values[i, j] = x
    return scipy.sparse.csr_matrix(values)


def _two_segment_band():
    return [{
        "diag_offset": 0,
        "segments": [
            {"rows": [0, 3], "bandwidth": [0, 1]},
            {"rows": [3, 6], "bandwidth": [1, 0]},
        ],
    }]


def _vdia_vbr_csr_matrix():
    """14x14: VDIA band (rows 0..3) + two VBR blocks + scattered CSR residual."""
    values = numpy.zeros((14, 14), dtype=float)
    values[0:3, 0:3] = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    values[3:6, 3:6] = numpy.arange(101, 110, dtype=float).reshape(3, 3)
    values[6:14, 6:14] = numpy.arange(201, 265, dtype=float).reshape(8, 8)
    for r, c, v in [(0, 13, 5.0), (1, 12, 4.0), (2, 11, 3.0), (11, 2, 6.0), (12, 1, 1.0), (13, 0, 2.0)]:
        values[r, c] = v
    return scipy.sparse.csr_matrix(values)


_VDIA_VBR_CSR_BAND = [{"diag_offset": 0, "segments": [{"rows": [0, 3], "bandwidth": [2, 2]}]}]
_VDIA_VBR_CSR_BLOCKS = [(3, 6, 3, 6), (6, 14, 6, 14)]


def test_spmv_single_threaded_bandmkl_two_segments():
    """SpMV with MKL DIA over a multi-segment band (row0 != 0) + naive CSR."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_bandmkl_two_segments"
    matrix = Matrix(_two_segment_band_matrix(), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    vdia = plan.extract(BandExtractorSkip(bands=_two_segment_band()))
    assert vdia.nsegments == 2
    plan.dispatch(vdia, MKLDIASpmv())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_bandmkl_two_segments():
    """SpMM with MKL DIA over a multi-segment band (row0 != 0) + naive CSR."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_bandmkl_two_segments"
    matrix = Matrix(_two_segment_band_matrix(), name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.matrix(_generated_matrix_path(cols, 512), shape=(cols, 512), layout=DenseLayout.ROW_MAJOR))
    vdia = plan.extract(BandExtractorSkip(bands=_two_segment_band()))
    assert vdia.nsegments == 2
    plan.dispatch(vdia, MKLDIASpmm())
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_single_threaded_bandmkl_blockmixed_csrnaive():
    """SpMV composing MKL DIA + mixed VBR + naive CSR."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmv_single_threaded_bandmkl_blockmixed_csrnaive"
    matrix = Matrix(_vdia_vbr_csr_matrix(), name=filename)
    cols = matrix.ncols

    write_dense_vector(1.0, cols)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.vector(_generated_vector_path(cols), cols))
    plan.dispatch(plan.extract(BandExtractorSkip(bands=_VDIA_VBR_CSR_BAND)), MKLDIASpmv())
    plan.dispatch(plan.extract(BlockDetectorSkip(_VDIA_VBR_CSR_BLOCKS)), MixedVBRSpmv())
    plan.dispatch(plan.extract(CSRConvertor()), NaiveCSRSpmv())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = matrix.to_scipy().dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_single_threaded_bandmkl_blockmixed_csrnaive():
    """SpMM composing MKL DIA + mixed VBR + naive CSR."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    filename = "spmm_single_threaded_bandmkl_blockmixed_csrnaive"
    matrix = Matrix(_vdia_vbr_csr_matrix(), name=filename)
    rows, cols = matrix.nrows, matrix.ncols

    write_dense_matrix(1.0, cols, 512)
    plan = Plan(matrix, artifact_dir="tests")
    plan.rhs(DenseInput.matrix(_generated_matrix_path(cols, 512), shape=(cols, 512), layout=DenseLayout.ROW_MAJOR))
    plan.dispatch(plan.extract(BandExtractorSkip(bands=_VDIA_VBR_CSR_BAND)), MKLDIASpmm())
    plan.dispatch(plan.extract(BlockDetectorSkip(_VDIA_VBR_CSR_BLOCKS)), MixedVBRSpmm())
    plan.dispatch(plan.extract(CSRConvertor()), NaiveCSRSpmm())
    output = plan.compile(filename=filename, bench=1).build().run().split("\n")

    result_lines = _numeric_result_lines(output)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = matrix.to_scipy().dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)
