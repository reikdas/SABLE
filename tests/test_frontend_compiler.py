import pathlib

import numpy
import pytest
import scipy.sparse

import sable.compiler
from sable import Matrix, Operation, Plan
from sable.extractors import BlockDetectorSkip, CSRConvertor
from sable.formats import CSR
from sable.kernels import SPRegCSRSpmm, SPV8CSRSpmv, UZPCSRSpmv
from sable.tensor import DenseInput


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
    csr = plan.extract(CSRConvertor())
    plan.dispatch(csr, UZPCSRSpmv())

    command = sable.compiler.build_compile_command_for_plan(plan, "input.c", "output")

    assert command[0] == "gcc"
    assert any(path.endswith("uzp-genex/polybench.c") for path in command)
    assert any(path.endswith("uzp-genex/spf_structure.c") for path in command)
    assert any(path.endswith("uzp-genex/spf_executors.c") for path in command)
    assert any(path.endswith("uzp-genex/spf_executors_uninc.c") for path in command)
    assert any(flag.endswith("uzp-genex") for flag in command if flag.startswith("-I"))
    assert "-DGEN_EXECUTOR_SPMV_ORIGINAL" in command
    assert "-lm" in command


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
    assert vbr.rpntr.values == [0, 2, 3]
    assert vbr.cpntr.values == [0, 2, 3]
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
    plan = Plan(Matrix(A, name="fully_dense_blocks"), artifact_dir="/tmp")

    vbr = plan.extract(BlockDetectorSkip(blocks=[(0, 2, 0, 2)]))

    assert vbr.blocks == [(0, 2, 0, 2)]
    assert vbr.val.values == [1.0, 3.0, 2.0, 4.0]
    assert plan.residual.nnz == 0


def test_csr_convertor_claims_all_entries_without_dense_blocks():
    A = scipy.sparse.csr_matrix(
        numpy.array(
            [
                [0.0, 2.0, 0.0],
                [3.0, 0.0, 4.0],
            ]
        )
    )
    plan = Plan(Matrix(A, name="fully_sparse"), artifact_dir="/tmp")

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
