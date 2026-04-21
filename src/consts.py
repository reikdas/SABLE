import os
from enum import Enum

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")

class SparseKernel(str, Enum):
    MKL = "mkl"
    SPV8 = "spv8"
    UZP = "uzp"
    NAIVE = "naive"

class DenseKernel(str, Enum):
    NAIVE = "naive"
    BLAS = "blas"

def _detect_mkl_config() -> tuple[list[str], bool]:
    """Detect MKL compiler flags and availability.

    Resolution order:
    1. ``MKLROOT`` environment variable (standard Intel oneAPI setup).
    2. nix-shell: gcc wrapper injects include/lib paths via
       ``NIX_CFLAGS_COMPILE`` / ``NIX_LDFLAGS``, so only ``-lmkl_rt`` is needed.
    3. System-installed MKL (e.g. ``apt install intel-mkl``): headers reachable
       from standard include paths or ``/usr/include/mkl/``.
    4. Hardcoded fallback (Purdue cluster path).
    """
    # 1. MKLROOT (Intel oneAPI / system install)
    mklroot = os.environ.get("MKLROOT")
    if mklroot and os.path.isdir(os.path.join(mklroot, "include")):
        return [f"-I{mklroot}/include", f"-L{mklroot}/lib/intel64", "-lmkl_rt"], True

    # 2. nix-shell: the gcc wrapper injects include/lib paths via
    #    NIX_CFLAGS_COMPILE / NIX_LDFLAGS, so only -lmkl_rt is needed.
    #    (Only relevant when running inside a Nix shell.)
    nix_cflags = os.environ.get("NIX_CFLAGS_COMPILE", "")
    nix_ldflags = os.environ.get("NIX_LDFLAGS", "")
    if any(
        "mkl" in token.lower()
        for token in (nix_cflags.split() + nix_ldflags.split())
    ):
        return ["-lmkl_rt"], True

    # 3. System-installed MKL (apt install intel-mkl / distro packages)
    _system_mkl_include_dirs = [
        "/usr/include",
        "/usr/include/mkl",
        "/usr/include/x86_64-linux-gnu",
    ]
    for inc_dir in _system_mkl_include_dirs:
        if os.path.isfile(os.path.join(inc_dir, "mkl.h")):
            flags = ["-lmkl_rt"]
            if inc_dir != "/usr/include":
                flags = [f"-I{inc_dir}"] + flags
            return flags, True

    # 4. Hardcoded fallback
    _mkl_path = os.path.join(
        "/home", "min", "a", "das160", "intel", "oneapi", "mkl", "latest"
    )
    flags = [f"-I{_mkl_path}/include", f"-L{_mkl_path}/lib/intel64", "-lmkl_rt"]
    available = os.path.isdir(os.path.join(_mkl_path, "include"))
    return flags, available


MKL_FLAGS, MKL_AVAILABLE = _detect_mkl_config()
SPV8_FLAGS = [
    f"{BASE_PATH}/spv8-public/bin/spmv_spv8.o",
    f"-I{BASE_PATH}/spv8-public/src"
]
_UZP_GENEX_DIR = os.path.join(BASE_PATH, "uzp-artifact", "spmv-executors", "uzp-genex")
UZP_SOURCES = [
    os.path.join(_UZP_GENEX_DIR, "polybench.c"),
    os.path.join(_UZP_GENEX_DIR, "spf_structure.c"),
    os.path.join(_UZP_GENEX_DIR, "spf_executors.c"),
    os.path.join(_UZP_GENEX_DIR, "spf_executors_uninc.c"),
]
UZP_FLAGS = [
    f"-I{_UZP_GENEX_DIR}",
    "-DGEN_EXECUTOR_SPMV_ORIGINAL",
    "-lm",
]
CFLAGS = [
    "-O3",
    "-march=native",
    "-funroll-all-loops",
    "-mprefer-vector-width=512",
    "-mavx",
    "-ffast-math",
    "-fopenmp",
    "-lpthread",
]

# Extra source files and compiler flags to add per sparse kernel.
BACKEND_EXTRA_SOURCES: dict[SparseKernel, list[str]] = {
    SparseKernel.MKL: [],
    SparseKernel.SPV8: [],
    SparseKernel.UZP: UZP_SOURCES,
    SparseKernel.NAIVE: [],
}
BACKEND_FLAGS: dict[SparseKernel, list[str]] = {
    SparseKernel.MKL: MKL_FLAGS,
    SparseKernel.SPV8: SPV8_FLAGS,
    SparseKernel.UZP: UZP_FLAGS,
    SparseKernel.NAIVE: [],
}


def build_compile_command(
    c_file_path: str,
    output_path: str,
    sparse_kernel: SparseKernel = SparseKernel.NAIVE,
    dense_kernel: DenseKernel = DenseKernel.NAIVE,
) -> list[str]:
    """Build the gcc command for compiling generated C code (SpMV or SpMM).

    This is a pure function (no side-effects) so it can be unit-tested in
    isolation.  The caller is responsible for actually running the command.

    The *sparse_kernel* selects sparse-kernel-specific sources and flags from
    :data:`BACKEND_EXTRA_SOURCES` / :data:`BACKEND_FLAGS`.  MKL flags are
    added whenever *dense_kernel* is :attr:`DenseKernel.BLAS` (``cblas_dgemv``
    lives in MKL) or *sparse_kernel* is :attr:`SparseKernel.MKL`.
    """
    extra_sources = list(BACKEND_EXTRA_SOURCES[sparse_kernel])
    extra_flags = list(BACKEND_FLAGS[sparse_kernel])

    # For dense kernel BLAS, add MKL flags if needed, i.e. if not already added by the sparse kernel selection.
    if dense_kernel == DenseKernel.BLAS and sparse_kernel != SparseKernel.MKL:
        extra_flags += MKL_FLAGS

    return (
        ["gcc", c_file_path]
        + extra_sources
        + ["-o", output_path]
        + CFLAGS
        + extra_flags
    )
