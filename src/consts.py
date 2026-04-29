import os

from sable.build_config import CFLAGS, DenseKernel, MKL_FLAGS, REPO_ROOT, SparseKernel


BASE_PATH = REPO_ROOT

SPV8_FLAGS = [
    f"{BASE_PATH}/spv8-public/bin/spmv_spv8.o",
    f"-I{BASE_PATH}/spv8-public/src",
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

# Legacy compile-command support for the remaining UZP codegen path.
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
    extra_sources = list(BACKEND_EXTRA_SOURCES[sparse_kernel])
    extra_flags = list(BACKEND_FLAGS[sparse_kernel])

    if dense_kernel in (DenseKernel.MKL, DenseKernel.MIXED) and sparse_kernel != SparseKernel.MKL:
        extra_flags += MKL_FLAGS

    return (
        ["gcc", c_file_path]
        + extra_sources
        + ["-o", output_path]
        + list(CFLAGS)
        + extra_flags
    )
