import os
from enum import Enum

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")

class Backend(str, Enum):
    MKL = "mkl"
    SPV8 = "spv8"
    UZP = "uzp"
    NAIVE = "naive"

class DenseKernel(str, Enum):
    NAIVE = "naive"
    BLAS = "blas"

MKL_PATH = os.path.join("/home", "min", "a", "das160", "intel", "oneapi", "mkl", "latest")
    # TODO: ^^ this isn't portable. We should do better. E.g. read this from an environment variable or config file.
MKL_FLAGS = [
    f"-I{MKL_PATH}/include", f"-L{MKL_PATH}/lib/intel64",
    "-lmkl_rt"]
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

# Extra source files and compiler flags to add per backend.
BACKEND_EXTRA_SOURCES: dict[Backend, list[str]] = {
    Backend.MKL: [],
    Backend.SPV8: [],
    Backend.UZP: UZP_SOURCES,
    Backend.NAIVE: [],
}
BACKEND_FLAGS: dict[Backend, list[str]] = {
    Backend.MKL: MKL_FLAGS,
    Backend.SPV8: SPV8_FLAGS,
    Backend.UZP: UZP_FLAGS,
    Backend.NAIVE: [],
}
