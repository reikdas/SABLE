import pathlib

import numpy
import pytest
import scipy.sparse

import sable.compiler
from sable import Matrix, Operation, Plan, out_of_line
from sable.extractors import BandExtractorSkip, BlockDetectorSkip, CSRConvertor
from sable.formats import CSR, VDIA
from sable.kernels import (
    MixedVBRSpmm,
    NaiveVDIASpmv,
    NaiveVBRSpmv,
    SPRegCSRSpmm,
    SPV8CSRSpmv,
    UZPCSRSpmv,
)
from sable.tensor import DenseInput, DenseLayout


class CustomCSRSpmv:
    accepts = CSR
    operation = Operation.SPMV

    def emit_helpers(self):
        return "/* custom helper marker */\n"

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        return f"""\
/* custom csr kernel */
for (int row = 0; row < {fmt.nrows}; row++) {{
    double acc = 0.0;
    for (int p = {fmt.indptr}[row]; p < {fmt.indptr}[row + 1]; p++) {{
        acc += {fmt.values}[p] * {x}[{fmt.indices}[p]];
    }}
    {y}[row] += acc;
}}
"""


class CustomCSRSpmm:
    accepts = CSR
    operation = Operation.SPMM

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        return ""


class CustomOutOfLineCSRSpmv:
    accepts = CSR
    operation = Operation.SPMV

    def emit_call(self, fmt: CSR, y: str, x: str, rhs):
        return out_of_line(
            f"""\
for (int row = 0; row < {fmt.nrows}; row++) {{
    double acc = 0.0;
    for (int p = {fmt.indptr}[row]; p < {fmt.indptr}[row + 1]; p++) {{
        acc += {fmt.values}[p] * {x}[{fmt.indices}[p]];
    }}
    {y}[row] += acc;
}}
""",
            name="custom_csr_spmv_body",
        )


def _numeric_result_lines(output: str) -> list[float]:
    values = []
    skip_timing = True
    for line in output.splitlines():
        line = line.strip()
        if not line:
            skip_timing = False
            continue
        if skip_timing:
            continue
        values.append(float(line))
    return values


def test_user_defined_kernel_compiles_and_runs_without_compiler_changes(tmp_path):
    rhs_path = tmp_path / "x.vector"
    x = numpy.array([1.0, 2.0, 3.0])
    rhs_path.write_text(",".join(map(str, x)) + "\n")

    A = scipy.sparse.csr_matrix(
        numpy.array(
            [
                [2.0, 0.0, 4.0],
                [0.0, 5.0, 0.0],
                [7.0, 0.0, 8.0],
            ]
        )
    )
    matrix = Matrix(A, name="custom_frontend")
    plan = Plan(matrix, artifact_dir=str(tmp_path))
    plan.rhs(DenseInput.vector(str(rhs_path), size=A.shape[1]))
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, CustomCSRSpmv())

    executor = plan.compile(filename="custom_frontend", bench=1)
    source = pathlib.Path(executor.c_path).read_text()
    assert "custom csr kernel" in source
    assert ".sabledata" in source
    assert ".vbrc" not in source
    assert "gen_single_threaded" not in source

    output = executor.build().run()
    y_generated = numpy.array(_numeric_result_lines(output))
    numpy.testing.assert_allclose(y_generated, A.dot(x), rtol=1e-10, atol=1e-10)
    assert "-lmkl_rt" not in (executor.compile_command or [])


def test_frontend_compiler_does_not_use_legacy_cross_product_lowering():
    source = pathlib.Path(sable.compiler.__file__).read_text()
    assert "gen_single_threaded" not in source
    assert "_select_codegen" not in source


def test_out_of_line_helper_hoists_marked_kernel_body(tmp_path):
    rhs_path = tmp_path / "x.vector"
    x = numpy.array([1.0, 2.0])
    rhs_path.write_text(",".join(map(str, x)) + "\n")
    A = scipy.sparse.csr_matrix(numpy.array([[1.0, 2.0], [0.0, 3.0]]))

    plan = Plan(Matrix(A, name="custom_out_of_line"), artifact_dir=str(tmp_path))
    plan.rhs(DenseInput.vector(str(rhs_path), size=2))
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, CustomOutOfLineCSRSpmv())

    executor = plan.compile(filename="custom_out_of_line", bench=1)
    source = pathlib.Path(executor.c_path).read_text()
    assert (
        "static void custom_csr_spmv_body(double *y, const double *x, const int *csr_indptr, "
        "const int *csr_indices, const double *csr_val)"
    ) in source
    assert "custom_csr_spmv_body(y, x, csr_indptr, csr_indices, csr_val);" in source

    output = executor.build().run()
    y_generated = numpy.array(_numeric_result_lines(output))
    numpy.testing.assert_allclose(y_generated, A.dot(x), rtol=1e-10, atol=1e-10)


def test_vbr_kernel_uses_parameterized_out_of_line_helper(tmp_path):
    rhs_path = tmp_path / "x.vector"
    rhs_path.write_text("1.0,1.0,1.0,1.0\n")
    A = scipy.sparse.csr_matrix(
        numpy.array(
            [
                [1.0, 2.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 6.0],
                [0.0, 0.0, 7.0, 8.0],
            ]
        )
    )
    blocks = [(0, 2, 0, 2), (2, 4, 2, 4)]

    call_plan = Plan(Matrix(A, name="vbr_call"), artifact_dir=str(tmp_path / "call"))
    call_plan.rhs(DenseInput.vector(str(rhs_path), size=4))
    call_vbr = call_plan.extract(BlockDetectorSkip(blocks=blocks))
    call_plan.dispatch(call_vbr, NaiveVBRSpmv())
    call_source = pathlib.Path(call_plan.compile(filename="vbr_call", bench=1).c_path).read_text()

    assert call_source.count("static void vbr_val_spmv_naive_block(") == 1
    assert "vbr_val_spmv_naive_block(y, x, vbr_val, 0, 2, 0, 2, 0);" in call_source
    assert "vbr_val_spmv_naive_block(y, x, vbr_val, 2, 4, 2, 4, 4);" in call_source


def test_vbr_kernel_marks_loop_blocks_out_of_line_and_leaves_blas_inline(tmp_path):
    rhs_path = tmp_path / "x.matrix"
    rhs_path.write_text(",".join(["1.0"] * 20) + "\n")
    A = numpy.zeros((10, 10))
    A[:8, :8] = 1.0
    A[8:10, 8:10] = 1.0

    plan = Plan(
        Matrix(scipy.sparse.csr_matrix(A), name="vbr_blas_inline"),
        artifact_dir=str(tmp_path),
    )
    plan.rhs(DenseInput.matrix(str(rhs_path), shape=(10, 2), layout=DenseLayout.ROW_MAJOR))
    vbr = plan.extract(BlockDetectorSkip(blocks=[(0, 8, 0, 8), (8, 10, 8, 10)]))
    plan.dispatch(vbr, MixedVBRSpmm())
    source = pathlib.Path(plan.compile(filename="vbr_blas_inline", bench=1).c_path).read_text()

    assert "static void vbr_val_spmm_mkl_block_0" not in source
    assert "vbr_val_spmm_mkl_block_0(y, x, vbr_val);" not in source
    assert "cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans" in source
    assert source.count("static void vbr_val_spmm_naive_block(") == 1
    assert "vbr_val_spmm_naive_block(y, x, vbr_val, 8, 10, 8, 10, 64, 2);" in source


def test_vdia_kernel_uses_parameterized_out_of_line_helper(tmp_path):
    rhs_path = tmp_path / "x.vector"
    rhs_path.write_text("1.0,1.0,1.0,1.0\n")
    A = scipy.sparse.csr_matrix(numpy.diag([1.0, 2.0, 3.0, 4.0]))
    bands = [
        {
            "diag_offset": 0,
            "segments": [
                {
                    "rows": [0, 2],
                    "bandwidth": [0, 0],
                },
                {
                    "rows": [2, 4],
                    "bandwidth": [0, 0],
                }
            ],
        }
    ]

    call_plan = Plan(Matrix(A, name="vdia_call"), artifact_dir=str(tmp_path / "call"))
    call_plan.rhs(DenseInput.vector(str(rhs_path), size=4))
    call_vdia = call_plan.extract(BandExtractorSkip(bands=bands))
    call_plan.dispatch(call_vdia, NaiveVDIASpmv())
    call_source = pathlib.Path(call_plan.compile(filename="vdia_call", bench=1).c_path).read_text()

    assert call_source.count("static void vdia_data_spmv_naive_segment(") == 1
    assert "vdia_data_spmv_naive_segment(y, x, vdia_data, 4, 0, 0, 2, 0, 0, 0);" in call_source
    assert "vdia_data_spmv_naive_segment(y, x, vdia_data, 4, 0, 2, 4, 0, 0, 2);" in call_source


def test_plan_rejects_mixed_kernel_operations(tmp_path):
    matrix = Matrix(scipy.sparse.eye(2, format="csr"), name="mixed_operations")
    plan = Plan(matrix, artifact_dir=str(tmp_path))
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, CustomCSRSpmv())

    with pytest.raises(ValueError, match="agree on operation"):
        plan.dispatch(csr, CustomCSRSpmm())


def test_spreg_kernel_contributes_cxx_build_requirements(tmp_path):
    matrix = Matrix(scipy.sparse.eye(2, format="csr"), name="spreg_flags")
    plan = Plan(matrix, artifact_dir=str(tmp_path))
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, SPRegCSRSpmm())

    command = sable.compiler.build_compile_command_for_plan(plan, "input.c", "output")

    assert command[0] == "g++"
    assert command[1:3] == ["-x", "c++"]
    assert any(path.endswith("libspreg.a") for path in command) or any(
        path.endswith("spmm_spreg_wrapper.cpp") for path in command
    )
    if any(path.endswith("libspreg.a") for path in command):
        lib_index = next(i for i, path in enumerate(command) if path.endswith("libspreg.a"))
        reset_index = next(i for i in range(len(command) - 1) if command[i : i + 2] == ["-x", "none"])
        assert reset_index < lib_index
    assert "-DENABLE_AVX512" in command


def test_spv8_kernel_contributes_object_and_include_flags(tmp_path):
    matrix = Matrix(scipy.sparse.eye(2, format="csr"), name="spv8_flags")
    plan = Plan(matrix, artifact_dir=str(tmp_path))
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, SPV8CSRSpmv())

    command = sable.compiler.build_compile_command_for_plan(plan, "input.c", "output")

    assert command[0] == "gcc"
    assert any(path.endswith("spmv_spv8.o") for path in command)
    assert any(flag.endswith("spv8-public/src") for flag in command if flag.startswith("-I"))


def test_uzp_kernel_contributes_sources_and_flags(tmp_path):
    matrix = Matrix(scipy.sparse.eye(2, format="csr"), name="uzp_flags")
    plan = Plan(matrix, artifact_dir=str(tmp_path))
    kernel = UZPCSRSpmv()
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, kernel)

    command = sable.compiler.build_compile_command_for_plan(plan, "input.c", "output")

    assert command[0] == "gcc"
    assert any(path.endswith("uzp-genex/polybench.c") for path in command)
    assert any(path.endswith("uzp-genex/spf_structure.c") for path in command)
    assert any(path.endswith("uzp-genex/spf_executors.c") for path in command)
    assert any(path.endswith("uzp-genex/spf_executors_uninc.c") for path in command)
    assert any(flag.endswith("uzp-genex") for flag in command if flag.startswith("-I"))
    assert "-DGEN_EXECUTOR_SPMV_ORIGINAL" in command
    assert "-lm" in command
    assert kernel.runtime_env() == {"OMP_NUM_THREADS": "1"}


def test_block_detector_skip_packs_vbr_and_tracks_csr_remainder():
    A = scipy.sparse.csr_matrix(
        numpy.array(
            [
                [2.0, 0.0, 4.0],
                [0.0, 5.0, 0.0],
                [7.0, 0.0, 8.0],
            ]
        )
    )
    matrix = Matrix(A, name="precomputed_blocks")

    plan = Plan(matrix, artifact_dir="/tmp")
    vbr = plan.extract(BlockDetectorSkip(blocks=[(0, 2, 0, 2)]))

    assert vbr.blocks == [(0, 2, 0, 2)]
    assert vbr.rpntr == [0, 2, 3]
    assert vbr.cpntr == [0, 2, 3]
    assert vbr.val.values == [2.0, 0.0, 0.0, 5.0]
    numpy.testing.assert_array_equal(
        plan.residual.to_scipy().toarray(),
        numpy.array(
            [
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0],
                [7.0, 0.0, 8.0],
            ]
        ),
    )

    csr = plan.extract(CSRConvertor())

    assert csr.nnz == 3
    assert csr.indptr.values == [0, 1, 1, 3]
    assert csr.indices.values == [2, 0, 2]
    assert csr.values.values == [4.0, 7.0, 8.0]
    assert plan.residual.nnz == 0


def test_block_detector_skip_can_claim_a_fully_dense_matrix():
    A = scipy.sparse.csr_matrix(numpy.array([[1.0, 2.0], [3.0, 4.0]]))
    plan = Plan(Matrix(A, name="fully_claimed_blocks"), artifact_dir="/tmp")

    vbr = plan.extract(BlockDetectorSkip(blocks=[(0, 2, 0, 2)]))

    assert vbr.blocks == [(0, 2, 0, 2)]
    assert vbr.val.values == [1.0, 3.0, 2.0, 4.0]
    assert plan.residual.nnz == 0


def test_band_extractor_skip_packs_vdia_and_tracks_remainder():
    A = scipy.sparse.csr_matrix(
        numpy.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
    )
    bands = [
        {
            "diag_offset": 0,
            "segments": [
                {
                    "rows": [0, 3],
                    "bandwidth": [2, 2],
                }
            ],
        }
    ]
    plan = Plan(Matrix(A, name="fully_dense_band"), artifact_dir="/tmp")

    vdia = plan.extract(BandExtractorSkip(bands=bands))

    assert isinstance(vdia, VDIA)
    assert vdia.nregions == 1
    assert vdia.nsegments == 1
    assert vdia.region_ptr == [0, 1]
    assert vdia.diag_offsets == [0]
    assert vdia.row_start == [0]
    assert vdia.row_end == [3]
    assert vdia.lower_bw == [2]
    assert vdia.upper_bw == [2]
    assert vdia.data_ptr == [0, 15]
    assert vdia.data.values == [
        0.0,
        0.0,
        7.0,
        0.0,
        4.0,
        8.0,
        1.0,
        5.0,
        9.0,
        2.0,
        6.0,
        0.0,
        3.0,
        0.0,
        0.0,
    ]
    assert plan.residual.nnz == 0


def test_csr_convertor_claims_all_entries_without_prior_regions():
    A = scipy.sparse.csr_matrix(
        numpy.array(
            [
                [0.0, 2.0, 0.0],
                [3.0, 0.0, 4.0],
            ]
        )
    )
    plan = Plan(Matrix(A, name="csr_only"), artifact_dir="/tmp")

    csr = plan.extract(CSRConvertor())

    assert csr.nnz == 3
    assert csr.indptr.values == [0, 1, 3]
    assert csr.indices.values == [1, 0, 2]
    assert csr.values.values == [2.0, 3.0, 4.0]
    assert plan.residual.nnz == 0


def test_plan_compile_rejects_unclaimed_residual(tmp_path):
    rhs_path = tmp_path / "x.vector"
    rhs_path.write_text("1.0,1.0\n")

    matrix = Matrix(scipy.sparse.eye(2, format="csr"), name="incomplete")
    plan = Plan(matrix, artifact_dir=str(tmp_path))
    plan.rhs(DenseInput.vector(str(rhs_path), size=2))

    with pytest.raises(ValueError, match="unclaimed entries"):
        plan.compile(filename="incomplete", bench=1)


def test_plan_compile_rejects_extracted_format_without_dispatch(tmp_path):
    rhs_path = tmp_path / "x.vector"
    rhs_path.write_text("1.0,1.0\n")

    matrix = Matrix(scipy.sparse.eye(2, format="csr"), name="undispatched")
    plan = Plan(matrix, artifact_dir=str(tmp_path))
    plan.rhs(DenseInput.vector(str(rhs_path), size=2))
    plan.extract(CSRConvertor())

    with pytest.raises(ValueError, match="no dispatched kernel"):
        plan.compile(filename="undispatched", bench=1)
