"""Unit tests for build_compile_command() — the pure-function that assembles
the gcc command line for each (SparseKernel, DenseKernel) combination.

Run with:  pytest tests/test_compile_flags.py -v
"""

import pytest

from src.consts import (
    SparseKernel,
    DenseKernel,
    CFLAGS,
    MKL_FLAGS,
    SPV8_FLAGS,
    UZP_FLAGS,
    UZP_SOURCES,
    build_compile_command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cmd(sparse_kernel: SparseKernel, dense_kernel: DenseKernel) -> list[str]:
    """Shortcut that calls build_compile_command with dummy paths."""
    return build_compile_command("input.c", "output", sparse_kernel, dense_kernel)


def _flags(cmd: list[str]) -> list[str]:
    """Return everything in *cmd* after the "-o <output>" pair."""
    idx = cmd.index("-o")
    return cmd[idx + 2 :]  # skip "-o" and the output path


def _sources(cmd: list[str]) -> list[str]:
    """Return extra source files (items between input.c and -o)."""
    start = cmd.index("input.c") + 1
    end = cmd.index("-o")
    return cmd[start:end]


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

class TestStructure:
    """Every command must have the basic gcc structure."""

    @pytest.mark.parametrize("sparse_kernel", list(SparseKernel))
    @pytest.mark.parametrize("dense_kernel", list(DenseKernel))
    def test_starts_with_gcc_and_source(self, sparse_kernel, dense_kernel):
        cmd = _cmd(sparse_kernel, dense_kernel)
        assert cmd[0] == "gcc"
        assert cmd[1] == "input.c"

    @pytest.mark.parametrize("sparse_kernel", list(SparseKernel))
    @pytest.mark.parametrize("dense_kernel", list(DenseKernel))
    def test_has_output_flag(self, sparse_kernel, dense_kernel):
        cmd = _cmd(sparse_kernel, dense_kernel)
        assert "-o" in cmd
        idx = cmd.index("-o")
        assert cmd[idx + 1] == "output"

    @pytest.mark.parametrize("sparse_kernel", list(SparseKernel))
    @pytest.mark.parametrize("dense_kernel", list(DenseKernel))
    def test_includes_cflags(self, sparse_kernel, dense_kernel):
        cmd = _cmd(sparse_kernel, dense_kernel)
        for flag in CFLAGS:
            assert flag in cmd, f"Missing CFLAG {flag!r}"


# ---------------------------------------------------------------------------
# SparseKernel-specific flags
# ---------------------------------------------------------------------------

class TestMKLSparseKernel:
    def test_mkl_naive_includes_mkl_flags(self):
        flags = _flags(_cmd(SparseKernel.MKL, DenseKernel.NAIVE))
        for f in MKL_FLAGS:
            assert f in flags

    def test_mkl_mkl_includes_mkl_flags_once(self):
        """MKL sparse_kernel + MKL dense should NOT duplicate MKL flags."""
        cmd = _cmd(SparseKernel.MKL, DenseKernel.MKL)
        assert cmd.count("-lmkl_rt") == 1

    def test_no_extra_sources(self):
        assert _sources(_cmd(SparseKernel.MKL, DenseKernel.NAIVE)) == []


class TestSPV8SparseKernel:
    def test_spv8_naive_includes_spv8_flags(self):
        flags = _flags(_cmd(SparseKernel.SPV8, DenseKernel.NAIVE))
        for f in SPV8_FLAGS:
            assert f in flags

    def test_spv8_mkl_adds_mkl_flags(self):
        flags = _flags(_cmd(SparseKernel.SPV8, DenseKernel.MKL))
        for f in MKL_FLAGS:
            assert f in flags, f"MKL dense kernel should add MKL flag {f!r}"
        for f in SPV8_FLAGS:
            assert f in flags, f"SPV8 flag {f!r} should still be present"

    def test_no_extra_sources(self):
        assert _sources(_cmd(SparseKernel.SPV8, DenseKernel.NAIVE)) == []


class TestUZPSparseKernel:
    def test_uzp_naive_includes_uzp_flags(self):
        flags = _flags(_cmd(SparseKernel.UZP, DenseKernel.NAIVE))
        for f in UZP_FLAGS:
            assert f in flags

    def test_uzp_naive_includes_uzp_sources(self):
        sources = _sources(_cmd(SparseKernel.UZP, DenseKernel.NAIVE))
        assert sources == UZP_SOURCES

    def test_uzp_mkl_adds_mkl_flags(self):
        flags = _flags(_cmd(SparseKernel.UZP, DenseKernel.MKL))
        for f in MKL_FLAGS:
            assert f in flags
        for f in UZP_FLAGS:
            assert f in flags

    def test_uzp_mkl_keeps_uzp_sources(self):
        sources = _sources(_cmd(SparseKernel.UZP, DenseKernel.MKL))
        assert sources == UZP_SOURCES


class TestNaiveSparseKernel:
    def test_naive_naive_is_just_cflags(self):
        flags = _flags(_cmd(SparseKernel.NAIVE, DenseKernel.NAIVE))
        assert flags == list(CFLAGS)

    def test_naive_mkl_adds_mkl_flags(self):
        flags = _flags(_cmd(SparseKernel.NAIVE, DenseKernel.MKL))
        assert flags == list(CFLAGS) + list(MKL_FLAGS)

    def test_no_extra_sources(self):
        assert _sources(_cmd(SparseKernel.NAIVE, DenseKernel.NAIVE)) == []


# ---------------------------------------------------------------------------
# Cross-cutting: MKL dense kernel always brings in MKL
# ---------------------------------------------------------------------------

class TestMKLDenseAlwaysAddsMKL:
    @pytest.mark.parametrize("sparse_kernel", list(SparseKernel))
    def test_mkl_always_has_mkl_rt(self, sparse_kernel):
        cmd = _cmd(sparse_kernel, DenseKernel.MKL)
        assert "-lmkl_rt" in cmd

    @pytest.mark.parametrize("sparse_kernel", list(SparseKernel))
    def test_naive_dense_non_mkl_sparse_kernel_no_mkl_rt(self, sparse_kernel):
        if sparse_kernel == SparseKernel.MKL:
            pytest.skip("MKL sparse_kernel always has MKL flags")
        cmd = _cmd(sparse_kernel, DenseKernel.NAIVE)
        assert "-lmkl_rt" not in cmd
