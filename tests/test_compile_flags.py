"""Unit tests for build_compile_command() — the pure-function that assembles
the gcc command line for each (Backend, DenseKernel) combination.

Run with:  pytest tests/test_compile_flags.py -v
"""

import pytest

from src.consts import (
    Backend,
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

def _cmd(backend: Backend, dense_kernel: DenseKernel) -> list[str]:
    """Shortcut that calls build_compile_command with dummy paths."""
    return build_compile_command("input.c", "output", backend, dense_kernel)


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

    @pytest.mark.parametrize("backend", list(Backend))
    @pytest.mark.parametrize("dense_kernel", list(DenseKernel))
    def test_starts_with_gcc_and_source(self, backend, dense_kernel):
        cmd = _cmd(backend, dense_kernel)
        assert cmd[0] == "gcc"
        assert cmd[1] == "input.c"

    @pytest.mark.parametrize("backend", list(Backend))
    @pytest.mark.parametrize("dense_kernel", list(DenseKernel))
    def test_has_output_flag(self, backend, dense_kernel):
        cmd = _cmd(backend, dense_kernel)
        assert "-o" in cmd
        idx = cmd.index("-o")
        assert cmd[idx + 1] == "output"

    @pytest.mark.parametrize("backend", list(Backend))
    @pytest.mark.parametrize("dense_kernel", list(DenseKernel))
    def test_includes_cflags(self, backend, dense_kernel):
        cmd = _cmd(backend, dense_kernel)
        for flag in CFLAGS:
            assert flag in cmd, f"Missing CFLAG {flag!r}"


# ---------------------------------------------------------------------------
# Backend-specific flags
# ---------------------------------------------------------------------------

class TestMKLBackend:
    def test_mkl_naive_includes_mkl_flags(self):
        flags = _flags(_cmd(Backend.MKL, DenseKernel.NAIVE))
        for f in MKL_FLAGS:
            assert f in flags

    def test_mkl_blas_includes_mkl_flags_once(self):
        """MKL backend + BLAS dense should NOT duplicate MKL flags."""
        cmd = _cmd(Backend.MKL, DenseKernel.BLAS)
        assert cmd.count("-lmkl_rt") == 1

    def test_no_extra_sources(self):
        assert _sources(_cmd(Backend.MKL, DenseKernel.NAIVE)) == []


class TestSPV8Backend:
    def test_spv8_naive_includes_spv8_flags(self):
        flags = _flags(_cmd(Backend.SPV8, DenseKernel.NAIVE))
        for f in SPV8_FLAGS:
            assert f in flags

    def test_spv8_blas_adds_mkl_flags(self):
        flags = _flags(_cmd(Backend.SPV8, DenseKernel.BLAS))
        for f in MKL_FLAGS:
            assert f in flags, f"BLAS dense kernel should add MKL flag {f!r}"
        for f in SPV8_FLAGS:
            assert f in flags, f"SPV8 flag {f!r} should still be present"

    def test_no_extra_sources(self):
        assert _sources(_cmd(Backend.SPV8, DenseKernel.NAIVE)) == []


class TestUZPBackend:
    def test_uzp_naive_includes_uzp_flags(self):
        flags = _flags(_cmd(Backend.UZP, DenseKernel.NAIVE))
        for f in UZP_FLAGS:
            assert f in flags

    def test_uzp_naive_includes_uzp_sources(self):
        sources = _sources(_cmd(Backend.UZP, DenseKernel.NAIVE))
        assert sources == UZP_SOURCES

    def test_uzp_blas_adds_mkl_flags(self):
        flags = _flags(_cmd(Backend.UZP, DenseKernel.BLAS))
        for f in MKL_FLAGS:
            assert f in flags
        for f in UZP_FLAGS:
            assert f in flags

    def test_uzp_blas_keeps_uzp_sources(self):
        sources = _sources(_cmd(Backend.UZP, DenseKernel.BLAS))
        assert sources == UZP_SOURCES


class TestNaiveBackend:
    def test_naive_naive_is_just_cflags(self):
        flags = _flags(_cmd(Backend.NAIVE, DenseKernel.NAIVE))
        assert flags == list(CFLAGS)

    def test_naive_blas_adds_mkl_flags(self):
        flags = _flags(_cmd(Backend.NAIVE, DenseKernel.BLAS))
        assert flags == list(CFLAGS) + list(MKL_FLAGS)

    def test_no_extra_sources(self):
        assert _sources(_cmd(Backend.NAIVE, DenseKernel.NAIVE)) == []


# ---------------------------------------------------------------------------
# Cross-cutting: BLAS dense kernel always brings in MKL
# ---------------------------------------------------------------------------

class TestBlasDenseAlwaysAddsMKL:
    @pytest.mark.parametrize("backend", list(Backend))
    def test_blas_always_has_mkl_rt(self, backend):
        cmd = _cmd(backend, DenseKernel.BLAS)
        assert "-lmkl_rt" in cmd

    @pytest.mark.parametrize("backend", list(Backend))
    def test_naive_dense_non_mkl_backend_no_mkl_rt(self, backend):
        if backend == Backend.MKL:
            pytest.skip("MKL backend always has MKL flags")
        cmd = _cmd(backend, DenseKernel.NAIVE)
        assert "-lmkl_rt" not in cmd
