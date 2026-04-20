"""Golden tests for SpMV C code generation.

Each test regenerates C source from a small fixture matrix and compares it
against a stored reference file in ``tests/golden/``.  This catches
unexpected changes to codegen output during refactorings.

Regenerate golden files after *intentional* codegen changes::

    NIXPKGS_ALLOW_UNFREE=1 nix-shell --run \\
        "python3 -m pytest tests/test_codegen_golden.py --update-golden -v"

Run normally (comparison mode)::

    NIXPKGS_ALLOW_UNFREE=1 nix-shell --run \\
        "python3 -m pytest tests/test_codegen_golden.py -v"
"""

import os
import pathlib
import re

import numpy as np
import pytest
import scipy.io
import scipy.sparse

from src.codegen import (
    gen_single_threaded_spmv_naive_naive,
    gen_single_threaded_spmv_naive_spv8,
    gen_single_threaded_spmv_blas_mkl,
    gen_single_threaded_spmv_blas_naive,
)
from utils.convert_real_to_vbr import convert_sparse_to_vbrc_with_blocks, _write_vbrc_file
from utils.fileio import write_dense_vector

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden" / "fixture"


# ---------------------------------------------------------------------------
# pytest fixture: --update-golden flag
# ---------------------------------------------------------------------------


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


# ---------------------------------------------------------------------------
# Path normalisation
# ---------------------------------------------------------------------------

# Matches absolute paths embedded via fopen() in generated C, e.g.:
#   fopen("/tmp/.../test_matrix.vbrc", "r")
#   fopen("/home/.../generated_vector_6.vector", "r")
_PATH_RE = re.compile(r'fopen\("([^"]+)"')


def _normalize_c_source(source: str) -> str:
    """Replace machine-specific absolute paths with stable placeholders."""
    def _replace(m: re.Match) -> str:
        path = m.group(1)
        basename = os.path.basename(path)
        return f'fopen("<PATH>/{basename}"'
    return _PATH_RE.sub(_replace, source)


# ---------------------------------------------------------------------------
# Fixture: small matrix with one dense block + sparse remainder
# ---------------------------------------------------------------------------

@pytest.fixture
def vbrc_fixture():
    """Return VBRC arrays for a 6×6 matrix with a 3×3 dense block."""
    dense = np.array([
        [1, 2, 3, 0, 0, 0],
        [4, 5, 6, 0, 0, 0],
        [7, 8, 9, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 3],
    ], dtype=float)
    A = scipy.sparse.csc_matrix(dense)
    dense_blocks = [(0, 3, 0, 3)]
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(A, dense_blocks)
    return {
        "val": val, "indx": indx, "bindx": bindx,
        "rpntr": rpntr, "cpntr": cpntr,
        "bpntrb": bpntrb, "bpntre": bpntre,
        "ublocks": ublocks,
        "indptr": indptr, "indices": indices, "csr_val": csr_val,
        "rows": 6, "cols": 6,
    }


# ---------------------------------------------------------------------------
# Codegen variants to test
# ---------------------------------------------------------------------------

CODEGEN_VARIANTS = {
    "naive_naive": gen_single_threaded_spmv_naive_naive,
    "naive_spv8": gen_single_threaded_spmv_naive_spv8,
    "blas_mkl": gen_single_threaded_spmv_blas_mkl,
    "blas_naive": gen_single_threaded_spmv_blas_naive,
}


def _generate_c(gen_func, vbrc_data, tmpdir: str) -> str:
    """Call a codegen function and return the normalised C source."""
    vbr_dir = tmpdir
    codegen_dir = os.path.join(tmpdir, "codegen")
    os.makedirs(codegen_dir, exist_ok=True)

    _write_vbrc_file(
        "fixture", vbr_dir,
        vbrc_data["val"], vbrc_data["indx"], vbrc_data["bindx"],
        vbrc_data["rpntr"], vbrc_data["cpntr"],
        vbrc_data["bpntrb"], vbrc_data["bpntre"],
        vbrc_data["ublocks"],
        vbrc_data["indptr"], vbrc_data["indices"], vbrc_data["csr_val"],
    )
    write_dense_vector(1.0, vbrc_data["cols"])

    gen_func(
        vbrc_data["val"], vbrc_data["indx"], vbrc_data["bindx"],
        vbrc_data["rpntr"], vbrc_data["cpntr"],
        vbrc_data["bpntrb"], vbrc_data["bpntre"],
        vbrc_data["ublocks"],
        vbrc_data["indptr"], vbrc_data["indices"], vbrc_data["csr_val"],
        codegen_dir, "fixture", vbr_dir, bench=1,
    )

    c_path = os.path.join(codegen_dir, "fixture.c")
    with open(c_path) as f:
        return _normalize_c_source(f.read())


# ---------------------------------------------------------------------------
# Parametrised golden test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", list(CODEGEN_VARIANTS))
def test_codegen_golden(variant, vbrc_fixture, update_golden, tmp_path):
    gen_func = CODEGEN_VARIANTS[variant]
    actual = _generate_c(gen_func, vbrc_fixture, str(tmp_path))

    golden_path = GOLDEN_DIR / f"{variant}.c"

    if update_golden:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual)
        pytest.skip(f"Golden file updated: {golden_path}")
    else:
        if not golden_path.exists():
            pytest.fail(
                f"Golden file {golden_path} does not exist.  "
                f"Run with --update-golden to create it."
            )
        expected = golden_path.read_text()
        if actual != expected:
            # Show a useful diff
            import difflib
            diff = difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"golden/{variant}.c",
                tofile=f"actual/{variant}.c",
            )
            diff_text = "".join(diff)
            pytest.fail(
                f"Codegen output for {variant!r} differs from golden file.\n"
                f"Run with --update-golden to accept the new output.\n\n"
                f"{diff_text}"
            )
