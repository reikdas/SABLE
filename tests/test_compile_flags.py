"""Tests for the temporary legacy UZP compile-command path.

Most backend compile requirements are now tested through frontend kernels in
``tests/test_frontend_compiler.py``.  This file only covers
``src.consts.build_compile_command`` while UZP still uses legacy codegen.
"""

import pytest

from sable.build_config import CFLAGS, DenseKernel, MKL_FLAGS, SparseKernel
from src.consts import UZP_FLAGS, UZP_SOURCES, build_compile_command


def _cmd(dense_kernel: DenseKernel) -> list[str]:
    return build_compile_command("input.c", "output", SparseKernel.UZP, dense_kernel)


def _flags(cmd: list[str]) -> list[str]:
    idx = cmd.index("-o")
    return cmd[idx + 2 :]


def _sources(cmd: list[str]) -> list[str]:
    return cmd[cmd.index("input.c") + 1 : cmd.index("-o")]


@pytest.mark.parametrize("dense_kernel", list(DenseKernel))
def test_uzp_command_has_expected_shape(dense_kernel):
    cmd = _cmd(dense_kernel)

    assert cmd[0] == "gcc"
    assert cmd[1] == "input.c"
    assert "-o" in cmd
    assert cmd[cmd.index("-o") + 1] == "output"


@pytest.mark.parametrize("dense_kernel", list(DenseKernel))
def test_uzp_command_includes_uzp_sources(dense_kernel):
    assert _sources(_cmd(dense_kernel)) == UZP_SOURCES


@pytest.mark.parametrize("dense_kernel", list(DenseKernel))
def test_uzp_command_includes_base_and_uzp_flags(dense_kernel):
    flags = _flags(_cmd(dense_kernel))

    for flag in CFLAGS:
        assert flag in flags
    for flag in UZP_FLAGS:
        assert flag in flags


@pytest.mark.parametrize("dense_kernel", [DenseKernel.MKL, DenseKernel.MIXED])
def test_uzp_with_blas_dense_dispatch_adds_mkl_flags(dense_kernel):
    flags = _flags(_cmd(dense_kernel))

    for flag in MKL_FLAGS:
        assert flag in flags


def test_uzp_with_naive_dense_dispatch_does_not_add_mkl_flags():
    flags = _flags(_cmd(DenseKernel.NAIVE))

    for flag in MKL_FLAGS:
        assert flag not in flags
