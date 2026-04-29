from __future__ import annotations

import glob
import os

from sable.formats import CSR
from sable.build_config import MKL_FLAGS
from sable.kernels.base import SpmmKernel, SpmvKernel


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SPREG_BASE = os.path.join(_REPO_ROOT, "sparse-register-tiling", "spmm_nano_kernels")
_SPV8_BASE = os.path.join(_REPO_ROOT, "spv8-public")


def _mkl_compile_flags() -> list[str]:
    return [flag for flag in MKL_FLAGS if flag.startswith("-I")]


def _mkl_link_flags() -> list[str]:
    return [flag for flag in MKL_FLAGS if not flag.startswith("-I")]


def _empty_list() -> list[str]:
    return []


def _empty_dict() -> dict[str, str]:
    return {}


def _spreg_source_files() -> list[str]:
    generated_dir = os.path.join(_SPREG_BASE, "generated")
    return [
        os.path.join(_SPREG_BASE, "src", "cake_block_dims.cpp"),
        os.path.join(_SPREG_BASE, "src", "ExecutorFactory.cpp"),
        os.path.join(_SPREG_BASE, "src", "kernel_mapping.cpp"),
        os.path.join(_SPREG_BASE, "src", "mapping_io.cpp"),
        os.path.join(_SPREG_BASE, "src", "packing.cpp"),
        os.path.join(_SPREG_BASE, "src", "utils", "misc.cpp"),
        os.path.join(_SPREG_BASE, "src", "spmm_spreg_wrapper.cpp"),
        *glob.glob(os.path.join(generated_dir, "*.cpp")),
        *glob.glob(os.path.join(generated_dir, "AVX512", "factories", "*", "*.cpp")),
    ]


def _spreg_include_flags() -> list[str]:
    include_dirs = [
        os.path.join(_SPREG_BASE, "include"),
        os.path.join(_SPREG_BASE, "third_party", "version2", "vectorclass"),
        os.path.join(_SPREG_BASE, "third_party", "rte"),
        os.path.join(_SPREG_BASE, "generated"),
        os.path.join(_SPREG_BASE, "generated", "AVX512", "include"),
    ]
    return [f"-I{include_dir}" for include_dir in include_dirs]


def _spreg_name(fmt: CSR, suffix: str) -> str:
    base = str(fmt.indptr) or "csr"
    return f"{base}_{suffix}"


def _spv8_name(fmt: CSR, suffix: str) -> str:
    base = str(fmt.indptr) or "csr"
    return f"{base}_{suffix}"


class NaiveCSRSpmv(SpmvKernel):
    accepts = CSR

    def emit_includes(self) -> list[str]:
        return []

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: CSR, rhs) -> str:
        return ""

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        return f"""\
for (int i = 0; i < {fmt.nrows}; i++) {{
    for (int p = {fmt.indptr}[i]; p < {fmt.indptr}[i + 1]; p++) {{
        {y}[i] += {fmt.values}[p] * {x}[{fmt.indices}[p]];
    }}
}}
"""

    def emit_teardown(self, fmt: CSR, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


class MKLCSRSpmv(SpmvKernel):
    accepts = CSR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_spblas.h>"]

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: CSR, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        return f"""\
sparse_matrix_t csr_handle;
struct matrix_descr csr_descr;
csr_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
csr_descr.mode = SPARSE_FILL_MODE_FULL;
csr_descr.diag = SPARSE_DIAG_NON_UNIT;
mkl_sparse_d_create_csr(&csr_handle, SPARSE_INDEX_BASE_ZERO,
    {fmt.nrows}, {fmt.ncols},
    {fmt.indptr}, {fmt.indptr} + 1,
    {fmt.indices}, {fmt.values});
"""

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        return f"""\
mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_handle, csr_descr, {x}, 1.0, {y});
"""

    def emit_teardown(self, fmt: CSR, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        return "mkl_sparse_destroy(csr_handle);\n"

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class SPV8CSRSpmv(SpmvKernel):
    accepts = CSR

    def source_files(self) -> list[str]:
        return [os.path.join(_SPV8_BASE, "bin", "spmv_spv8.o")]

    def compile_flags(self) -> list[str]:
        return [f"-I{os.path.join(_SPV8_BASE, 'src')}"]

    def link_flags(self) -> list[str]:
        return []

    def runtime_env(self) -> dict[str, str]:
        return {}

    def emit_includes(self) -> list[str]:
        return ['#include "utility.h"']

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: CSR, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        mat = _spv8_name(fmt, "spv8_mat")
        tr = _spv8_name(fmt, "spv8_tr")
        scratch = _spv8_name(fmt, "spv8_y")
        return f"""\
struct csr_matrix {mat} = input_matrix({fmt.nnz}, {fmt.nrows}, {fmt.ncols},
    {fmt.values}, {fmt.indices}, {fmt.indptr});
struct tr_matrix {tr} = process(&{mat});
double *{scratch} = (double *)calloc({fmt.nrows}, sizeof(double));
assert({scratch} != NULL);
"""

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        tr = _spv8_name(fmt, "spv8_tr")
        scratch = _spv8_name(fmt, "spv8_y")
        return f"""\
memset({scratch}, 0, {fmt.nrows} * sizeof(double));
spmv_tr_spvv8_kernel(&{tr}, {x}, {scratch});
for (int i = 0; i < {fmt.nrows}; i++) {{
    {y}[i] += {scratch}[i];
}}
"""

    def emit_teardown(self, fmt: CSR, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        mat = _spv8_name(fmt, "spv8_mat")
        tr = _spv8_name(fmt, "spv8_tr")
        scratch = _spv8_name(fmt, "spv8_y")
        return f"""\
free({scratch});
for (int spv8_t = 0; spv8_t < {tr}.task_count; spv8_t++) {{
    free({tr}.tasks[spv8_t]);
}}
free({tr}.tasks);
free({tr}.task_sizes);
free({tr}.spvv8_len);
destroy_matrix(&{tr}.mat);
destroy_matrix(&{mat});
"""


class NaiveCSRSpmm(SpmmKernel):
    accepts = CSR

    def emit_includes(self) -> list[str]:
        return []

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: CSR, rhs) -> str:
        return ""

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        nrhs = rhs.shape[1]
        return f"""\
for (int i = 0; i < {fmt.nrows}; i++) {{
    for (int p = {fmt.indptr}[i]; p < {fmt.indptr}[i + 1]; p++) {{
        int col = {fmt.indices}[p];
        double a = {fmt.values}[p];
        for (int j = 0; j < {nrhs}; j++) {{
            {y}[i * {nrhs} + j] += a * {x}[col * {nrhs} + j];
        }}
    }}
}}
"""

    def emit_teardown(self, fmt: CSR, rhs) -> str:
        return ""

    compile_flags = staticmethod(_empty_list)
    link_flags = staticmethod(_empty_list)
    runtime_env = staticmethod(_empty_dict)


class MKLCSRSpmm(SpmmKernel):
    accepts = CSR

    def emit_includes(self) -> list[str]:
        return ["#include <mkl.h>", "#include <mkl_spblas.h>"]

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: CSR, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        return f"""\
sparse_matrix_t csr_handle;
struct matrix_descr csr_descr;
csr_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
csr_descr.mode = SPARSE_FILL_MODE_FULL;
csr_descr.diag = SPARSE_DIAG_NON_UNIT;
mkl_sparse_d_create_csr(&csr_handle, SPARSE_INDEX_BASE_ZERO,
    {fmt.nrows}, {fmt.ncols},
    {fmt.indptr}, {fmt.indptr} + 1,
    {fmt.indices}, {fmt.values});
"""

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        nrhs = rhs.shape[1]
        return f"""\
mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_handle, csr_descr,
    SPARSE_LAYOUT_ROW_MAJOR, {x}, {nrhs}, {nrhs}, 1.0, {y}, {nrhs});
"""

    def emit_teardown(self, fmt: CSR, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        return "mkl_sparse_destroy(csr_handle);\n"

    def compile_flags(self) -> list[str]:
        return _mkl_compile_flags()

    def link_flags(self) -> list[str]:
        return _mkl_link_flags()

    def runtime_env(self) -> dict[str, str]:
        return {"MKL_THREADING_LAYER": "GNU"}


class SPRegCSRSpmm(SpmmKernel):
    accepts = CSR

    def __init__(self, threads: int = 1):
        self.threads = threads

    def compiler(self) -> str:
        return "g++"

    def pre_source_flags(self) -> list[str]:
        return ["-x", "c++"]

    def source_files(self) -> list[str]:
        return _spreg_source_files()

    def compile_flags(self) -> list[str]:
        return [
            "-std=c++17",
            "-DRTE_CACHE_LINE_SIZE=64",
            "-DENABLE_AVX512",
            "-mavx512f",
            *_spreg_include_flags(),
        ]

    def link_flags(self) -> list[str]:
        return ["-lstdc++fs"]

    def runtime_env(self) -> dict[str, str]:
        return {}

    def emit_includes(self) -> list[str]:
        if self.threads != 1:
            raise ValueError("SPRegCSRSpmm is currently wired for single-threaded frontend tests")
        return [
            "#ifdef __cplusplus",
            "#ifndef restrict",
            "#define restrict __restrict__",
            "#endif /* restrict */",
            "#endif /* __cplusplus */",
            '#include "spmm_spreg_wrapper.h"',
        ]

    def emit_helpers(self) -> str:
        return ""

    def emit_setup(self, fmt: CSR, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        nrhs = rhs.shape[1]
        handle = _spreg_name(fmt, "spreg_handle")
        scratch = _spreg_name(fmt, "spreg_y")
        return f"""\
void *{handle} = spmm_spreg_init({fmt.values}, {fmt.indices}, {fmt.indptr},
    {fmt.nrows}, {fmt.ncols}, {nrhs}, {self.threads});
if ({handle} == NULL) {{
    fprintf(stderr, "Failed to initialize sparse-register-tiling executor\\n");
    return 1;
}}
double *{scratch} = (double *)calloc({fmt.nrows * nrhs}, sizeof(double));
assert({scratch} != NULL);
"""

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        nrhs = rhs.shape[1]
        handle = _spreg_name(fmt, "spreg_handle")
        scratch = _spreg_name(fmt, "spreg_y")
        return f"""\
spmm_spreg_execute({handle}, {scratch}, {x});
for (int i = 0; i < {fmt.nrows * nrhs}; i++) {{
    {y}[i] += {scratch}[i];
}}
"""

    def emit_teardown(self, fmt: CSR, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        handle = _spreg_name(fmt, "spreg_handle")
        scratch = _spreg_name(fmt, "spreg_y")
        return f"""\
spmm_spreg_cleanup({handle});
free({scratch});
"""
