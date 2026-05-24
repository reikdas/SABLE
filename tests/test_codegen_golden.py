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
from sable.extractors import BandExtractorSkip, BlockDetectorSkip, CSRConvertor
from sable.kernels import (
    MKLCSRSpmm,
    MKLCSRSpmv,
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

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden" / "fixture"
_PATH_RE = re.compile(r'fopen\("([^"]+)"')


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


FIXTURE_SIZE = 14
BAND_REGIONS = [
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
BLOCK_REGIONS = [(3, 6, 3, 6), (6, 14, 6, 14)]
CSR_RESIDUAL_ENTRIES = [
    (0, 13, 5.0),
    (1, 12, 4.0),
    (2, 11, 3.0),
    (11, 2, 6.0),
    (12, 1, 1.0),
    (13, 0, 2.0),
]


def _fixture_matrix(include_band: bool, include_block: bool, include_csr: bool) -> scipy.sparse.csr_matrix:
    values = numpy.zeros((FIXTURE_SIZE, FIXTURE_SIZE), dtype=float)

    if include_band:
        values[0:3, 0:3] = numpy.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=float,
        )

    if include_block:
        values[3:6, 3:6] = numpy.arange(101, 110, dtype=float).reshape(3, 3)
        values[6:14, 6:14] = numpy.arange(201, 265, dtype=float).reshape(8, 8)

    if include_csr:
        for row, col, value in CSR_RESIDUAL_ENTRIES:
            values[row, col] = value

    return scipy.sparse.csr_matrix(values)


def _normalize_c_source(source: str) -> str:
    source = source.replace(str(pathlib.Path(__file__).resolve().parents[1]), "<REPO>")

    def _replace(match: re.Match) -> str:
        path = match.group(1)
        return f'fopen("<PATH>/{os.path.basename(path)}"'

    return _PATH_RE.sub(_replace, source)


def _write_vector(path: pathlib.Path, size: int) -> None:
    path.write_text(",".join(["1.0"] * size) + "\n")


def _write_matrix(path: pathlib.Path, rows: int, cols: int) -> None:
    path.write_text(",".join(["1.0"] * (rows * cols)) + "\n")


SPMV_KERNELS = {
    "band": {
        "bandnaive": NaiveVDIASpmv,
        "bandmkl": MKLDIASpmv,
    },
    "block": {
        "blocknaive": NaiveVBRSpmv,
        "blockmkl": MKLVBRSpmv,
        "blockmixed": MixedVBRSpmv,
    },
    "csr": {
        "csrnaive": NaiveCSRSpmv,
        "csrmkl": MKLCSRSpmv,
        "csrspv8": SPV8CSRSpmv,
        "csruzp": UZPCSRSpmv,
    },
}

SPMM_KERNELS = {
    "band": {
        "bandnaive": NaiveVDIASpmm,
    },
    "block": {
        "blocknaive": NaiveVBRSpmm,
        "blockmkl": MKLVBRSpmm,
        "blockmixed": MixedVBRSpmm,
    },
    "csr": {
        "csrnaive": NaiveCSRSpmm,
        "csrmkl": MKLCSRSpmm,
        "csrspreg": SPRegCSRSpmm,
    },
}


def _build_codegen_variants() -> dict[str, tuple[int, list[tuple[str, str, type]]]]:
    variants: dict[str, tuple[int, list[tuple[str, str, type]]]] = {}
    for op_name, rhs_rank, kernels_by_format in (
        ("spmv", 1, SPMV_KERNELS),
        ("spmm", 2, SPMM_KERNELS),
    ):
        for include_band in (False, True):
            for include_block in (False, True):
                for include_csr in (False, True):
                    if not (include_band or include_block or include_csr):
                        continue

                    selected_formats = []
                    if include_band:
                        selected_formats.append("band")
                    if include_block:
                        selected_formats.append("block")
                    if include_csr:
                        selected_formats.append("csr")

                    def add_variants(
                        index: int,
                        dispatches: list[tuple[str, str, type]],
                    ) -> None:
                        if index == len(selected_formats):
                            variant = "_".join([op_name] + [token for _, token, _ in dispatches])
                            variants[variant] = (rhs_rank, dispatches)
                            return
                        format_kind = selected_formats[index]
                        for token, kernel_type in kernels_by_format[format_kind].items():
                            add_variants(index + 1, dispatches + [(format_kind, token, kernel_type)])

                    add_variants(0, [])
    return variants


CODEGEN_VARIANTS = _build_codegen_variants()


def _generate_c(variant: str, tmpdir: pathlib.Path) -> str:
    rhs_rank, dispatches = CODEGEN_VARIANTS[variant]
    selected_formats = {format_kind for format_kind, _, _ in dispatches}
    A = _fixture_matrix(
        include_band="band" in selected_formats,
        include_block="block" in selected_formats,
        include_csr="csr" in selected_formats,
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

    for format_kind, _, kernel_type in dispatches:
        if format_kind == "band":
            vdia = plan.extract(BandExtractorSkip(bands=BAND_REGIONS))
            plan.dispatch(vdia, kernel_type())
        elif format_kind == "block":
            vbr = plan.extract(BlockDetectorSkip(blocks=BLOCK_REGIONS))
            plan.dispatch(vbr, kernel_type())
        elif format_kind == "csr":
            csr = plan.extract(CSRConvertor())
            plan.dispatch(csr, kernel_type())
        else:
            raise AssertionError(f"Unhandled format kind: {format_kind}")

    executor = plan.compile(filename="fixture", bench=1)
    return _normalize_c_source(pathlib.Path(executor.c_path).read_text())


def _two_block_vbr_matrix() -> scipy.sparse.csr_matrix:
    values = numpy.zeros((6, 6), dtype=float)
    values[0:2, 0:2] = numpy.array([[1, 2], [3, 4]], dtype=float)
    values[3:6, 3:6] = numpy.arange(11, 20, dtype=float).reshape(3, 3)
    return scipy.sparse.csr_matrix(values)


def _two_segment_vdia_matrix() -> scipy.sparse.csr_matrix:
    values = numpy.zeros((6, 6), dtype=float)
    values[0, 0] = 1
    values[0, 1] = 2
    values[1, 1] = 3
    values[1, 2] = 4
    values[2, 2] = 5
    values[2, 3] = 6
    values[3, 2] = 7
    values[3, 3] = 8
    values[4, 3] = 9
    values[4, 4] = 10
    values[5, 4] = 11
    values[5, 5] = 12
    return scipy.sparse.csr_matrix(values)


TWO_VBR_BLOCKS = [(0, 2, 0, 2), (3, 6, 3, 6)]
TWO_VDIA_SEGMENTS = [
    {
        "diag_offset": 0,
        "segments": [
            {
                "rows": [0, 3],
                "bandwidth": [0, 1],
            },
            {
                "rows": [3, 6],
                "bandwidth": [1, 0],
            },
        ],
    }
]


def _generate_two_vbr_c(op_name: str, tmpdir: pathlib.Path) -> str:
    A = _two_block_vbr_matrix()
    plan = Plan(Matrix(A, name="fixture"), artifact_dir=str(tmpdir / "codegen"))
    if op_name == "spmv":
        rhs_path = tmpdir / "x.vector"
        _write_vector(rhs_path, A.shape[1])
        plan.rhs(DenseInput.vector(str(rhs_path), A.shape[1]))
        kernel = NaiveVBRSpmv()
    else:
        rhs_path = tmpdir / "x.matrix"
        _write_matrix(rhs_path, A.shape[1], 4)
        plan.rhs(DenseInput.matrix(str(rhs_path), shape=(A.shape[1], 4), layout=DenseLayout.ROW_MAJOR))
        kernel = NaiveVBRSpmm()

    vbr = plan.extract(BlockDetectorSkip(blocks=TWO_VBR_BLOCKS))
    plan.dispatch(vbr, kernel)
    executor = plan.compile(filename="fixture", bench=1)
    return _normalize_c_source(pathlib.Path(executor.c_path).read_text())


def _generate_two_vdia_c(op_name: str, tmpdir: pathlib.Path) -> str:
    A = _two_segment_vdia_matrix()
    plan = Plan(Matrix(A, name="fixture"), artifact_dir=str(tmpdir / "codegen"))
    if op_name == "spmv":
        rhs_path = tmpdir / "x.vector"
        _write_vector(rhs_path, A.shape[1])
        plan.rhs(DenseInput.vector(str(rhs_path), A.shape[1]))
        kernel = NaiveVDIASpmv()
    else:
        rhs_path = tmpdir / "x.matrix"
        _write_matrix(rhs_path, A.shape[1], 4)
        plan.rhs(DenseInput.matrix(str(rhs_path), shape=(A.shape[1], 4), layout=DenseLayout.ROW_MAJOR))
        kernel = NaiveVDIASpmm()

    vdia = plan.extract(BandExtractorSkip(bands=TWO_VDIA_SEGMENTS))
    plan.dispatch(vdia, kernel)
    executor = plan.compile(filename="fixture", bench=1)
    return _normalize_c_source(pathlib.Path(executor.c_path).read_text())


def _generate_two_vdia_mkl_c(tmpdir: pathlib.Path) -> str:
    A = _two_segment_vdia_matrix()
    plan = Plan(Matrix(A, name="fixture"), artifact_dir=str(tmpdir / "codegen"))
    rhs_path = tmpdir / "x.vector"
    _write_vector(rhs_path, A.shape[1])
    plan.rhs(DenseInput.vector(str(rhs_path), A.shape[1]))

    vdia = plan.extract(BandExtractorSkip(bands=TWO_VDIA_SEGMENTS))
    plan.dispatch(vdia, MKLDIASpmv())
    executor = plan.compile(filename="fixture", bench=1)
    return _normalize_c_source(pathlib.Path(executor.c_path).read_text())


PARAMETERIZED_OUT_OF_LINE_VARIANTS = {
    "spmv_blocknaive_two_blocks": lambda tmpdir: _generate_two_vbr_c("spmv", tmpdir),
    "spmm_blocknaive_two_blocks": lambda tmpdir: _generate_two_vbr_c("spmm", tmpdir),
    "spmv_bandnaive_two_segments": lambda tmpdir: _generate_two_vdia_c("spmv", tmpdir),
    "spmm_bandnaive_two_segments": lambda tmpdir: _generate_two_vdia_c("spmm", tmpdir),
    "spmv_bandmkl_two_segments": lambda tmpdir: _generate_two_vdia_mkl_c(tmpdir),
}


def _generate_interp_c(variant: str, tmpdir: pathlib.Path) -> str:
    rhs_rank, dispatches = CODEGEN_VARIANTS[variant]
    selected_formats = {format_kind for format_kind, _, _ in dispatches}
    A = _fixture_matrix(
        include_band="band" in selected_formats,
        include_block="block" in selected_formats,
        include_csr="csr" in selected_formats,
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

    for format_kind, _, kernel_type in dispatches:
        if format_kind == "band":
            vdia = plan.extract(BandExtractorSkip(bands=BAND_REGIONS))
            plan.dispatch(vdia, kernel_type())
        elif format_kind == "block":
            vbr = plan.extract(BlockDetectorSkip(blocks=BLOCK_REGIONS))
            plan.dispatch(vbr, kernel_type())
        elif format_kind == "csr":
            csr = plan.extract(CSRConvertor())
            plan.dispatch(csr, kernel_type())
        else:
            raise AssertionError(f"Unhandled format kind: {format_kind}")

    executor = plan.interpret(filename="fixture", bench=1)
    return _normalize_c_source(pathlib.Path(executor.c_path).read_text())


INTERP_CODEGEN_VARIANTS = {k: v for k, v in CODEGEN_VARIANTS.items() if "bandnaive" in k}


def _generate_two_vdia_interp_c(op_name: str, tmpdir: pathlib.Path) -> str:
    A = _two_segment_vdia_matrix()
    plan = Plan(Matrix(A, name="fixture"), artifact_dir=str(tmpdir / "codegen"))
    if op_name == "spmv":
        rhs_path = tmpdir / "x.vector"
        _write_vector(rhs_path, A.shape[1])
        plan.rhs(DenseInput.vector(str(rhs_path), A.shape[1]))
        kernel = NaiveVDIASpmv()
    else:
        rhs_path = tmpdir / "x.matrix"
        _write_matrix(rhs_path, A.shape[1], 4)
        plan.rhs(DenseInput.matrix(str(rhs_path), shape=(A.shape[1], 4), layout=DenseLayout.ROW_MAJOR))
        kernel = NaiveVDIASpmm()

    vdia = plan.extract(BandExtractorSkip(bands=TWO_VDIA_SEGMENTS))
    plan.dispatch(vdia, kernel)
    executor = plan.interpret(filename="fixture", bench=1)
    return _normalize_c_source(pathlib.Path(executor.c_path).read_text())


INTERP_PARAMETERIZED_VARIANTS = {
    "spmv_bandnaive_two_segments_interp": lambda tmpdir: _generate_two_vdia_interp_c("spmv", tmpdir),
    "spmm_bandnaive_two_segments_interp": lambda tmpdir: _generate_two_vdia_interp_c("spmm", tmpdir),
}


def _assert_parameterized_out_of_line_shape(variant: str, source: str) -> None:
    if variant == "spmv_blocknaive_two_blocks":
        assert source.count("static void vbr_val_spmv_naive_block(") == 1
        assert "vbr_val_spmv_naive_block(y, x, vbr_val, 0, 2, 0, 2, 0);" in source
        assert "vbr_val_spmv_naive_block(y, x, vbr_val, 3, 6, 3, 6, 4);" in source
        assert "vbr_val_spmv_naive_block_0" not in source
    elif variant == "spmm_blocknaive_two_blocks":
        assert source.count("static void vbr_val_spmm_naive_block(") == 1
        assert "vbr_val_spmm_naive_block(y, x, vbr_val, 0, 2, 0, 2, 0, 4);" in source
        assert "vbr_val_spmm_naive_block(y, x, vbr_val, 3, 6, 3, 6, 4, 4);" in source
        assert "vbr_val_spmm_naive_block_0" not in source
    elif variant == "spmv_bandnaive_two_segments":
        assert source.count("static void vdia_val_spmv_naive_segment(") == 1
        assert "vdia_val_spmv_naive_segment(y, x, vdia_val, vdia_idiag, 0, 3, 2, 0, 0);" in source
        assert "vdia_val_spmv_naive_segment(y, x, vdia_val, vdia_idiag, 3, 3, 2, 2, 6);" in source
        assert "vdia_val_spmv_naive_segment_0" not in source
    elif variant == "spmm_bandnaive_two_segments":
        assert source.count("static void vdia_val_spmm_naive_segment(") == 1
        assert "vdia_val_spmm_naive_segment(y, x, vdia_val, vdia_idiag, 0, 3, 2, 0, 0, 4);" in source
        assert "vdia_val_spmm_naive_segment(y, x, vdia_val, vdia_idiag, 3, 3, 2, 2, 6, 4);" in source
        assert "vdia_val_spmm_naive_segment_0" not in source
    elif variant == "spmv_bandmkl_two_segments":
        assert source.count("static void vdia_val_spmv_mkl_dia_segment(") == 1
        assert "vdia_val_spmv_mkl_dia_segment(y, x, vdia_val, vdia_idiag, 0, 3, 2, 0, 0);" in source
        assert "vdia_val_spmv_mkl_dia_segment(y, x, vdia_val, vdia_idiag, 3, 3, 2, 2, 6);" in source
        assert "vdia_val_spmv_mkl_dia_segment_0" not in source
    else:
        raise AssertionError(f"Unhandled parameterized out_of_line variant: {variant}")


def _assert_golden(variant: str, actual: str, golden_path: pathlib.Path, update_golden: bool) -> None:
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


@pytest.mark.parametrize("variant", list(CODEGEN_VARIANTS))
def test_codegen_golden(variant, update_golden, tmp_path):
    actual = _generate_c(variant, tmp_path)
    _assert_golden(variant, actual, GOLDEN_DIR / f"{variant}.c", update_golden)


@pytest.mark.parametrize("variant", list(PARAMETERIZED_OUT_OF_LINE_VARIANTS))
def test_parameterized_out_of_line_codegen_golden(variant, update_golden, tmp_path):
    actual = PARAMETERIZED_OUT_OF_LINE_VARIANTS[variant](tmp_path)
    _assert_parameterized_out_of_line_shape(variant, actual)
    _assert_golden(variant, actual, GOLDEN_DIR / f"{variant}.c", update_golden)


def _assert_interp_parameterized_shape(variant: str, source: str) -> None:
    if "spmv_bandnaive_two_segments" in variant:
        assert source.count("static void vdia_val_spmv_naive_segment(") == 1
        assert "for (int _interp_s = 0; _interp_s < 2; _interp_s++)" in source
        assert "_interp_vdia_val_spmv_naive_segment_row0[_interp_s]" in source
        assert "_interp_vdia_val_spmv_naive_segment_val_off[_interp_s]" in source
    elif "spmm_bandnaive_two_segments" in variant:
        assert source.count("static void vdia_val_spmm_naive_segment(") == 1
        assert "for (int _interp_s = 0; _interp_s < 2; _interp_s++)" in source
        assert "_interp_vdia_val_spmm_naive_segment_row0[_interp_s]" in source
        assert "_interp_vdia_val_spmm_naive_segment_val_off[_interp_s]" in source
    else:
        raise AssertionError(f"Unhandled interp parameterized variant: {variant}")


@pytest.mark.parametrize("variant", list(INTERP_CODEGEN_VARIANTS))
def test_interp_codegen_golden(variant, update_golden, tmp_path):
    actual = _generate_interp_c(variant, tmp_path)
    _assert_golden(f"{variant}_interp", actual, GOLDEN_DIR / f"{variant}_interp.c", update_golden)


@pytest.mark.parametrize("variant", list(INTERP_PARAMETERIZED_VARIANTS))
def test_interp_parameterized_codegen_golden(variant, update_golden, tmp_path):
    actual = INTERP_PARAMETERIZED_VARIANTS[variant](tmp_path)
    _assert_interp_parameterized_shape(variant, actual)
    _assert_golden(variant, actual, GOLDEN_DIR / f"{variant}.c", update_golden)
