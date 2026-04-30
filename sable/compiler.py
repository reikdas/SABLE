from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from sable.build_config import CFLAGS

from .formats import CSR, Rep, VBR
from .plan import Plan


@dataclass
class RepBinding:
    rep: Rep
    label: str
    c_name: str
    c_type: str

    @property
    def size(self) -> int:
        return len(self.rep.values)


@dataclass
class CompiledExecutor:
    c_path: str
    data_path: str
    artifact_dir: str
    filename: str
    codegen_time_ms: int
    plan: Plan
    runtime_env: dict[str, str]
    runtime_cwd: str | None = None
    binary_path: str | None = None
    compile_command: list[str] | None = None

    def build(self, output_path: str | None = None) -> "CompiledExecutor":
        output_path = output_path or os.path.join(self.artifact_dir, self.filename)
        output_path = os.path.abspath(output_path)
        command = build_compile_command_for_plan(self.plan, self.c_path, output_path)
        subprocess.check_call(command)
        self.binary_path = output_path
        self.compile_command = command
        return self

    def run(self) -> str:
        if self.binary_path is None:
            self.build()
        output = subprocess.check_output(
            [self.binary_path],
            cwd=self.runtime_cwd or self.artifact_dir,
            env=_runtime_env(self.compile_command or [], self.runtime_env),
        )
        return output.decode("utf-8")

    def execute(self) -> str:
        return self.run()


def _sanitize_identifier(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not sanitized:
        sanitized = "rep"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _kernel_list(kernel: object, method_name: str) -> list[str]:
    method = getattr(kernel, method_name, None)
    if method is None:
        return []
    values = method()
    return list(values or [])


def _kernel_dict(kernel: object, method_name: str) -> dict[str, str]:
    method = getattr(kernel, method_name, None)
    if method is None:
        return {}
    values = method()
    return dict(values or {})


def _kernel_value(kernel: object, method_name: str) -> Any:
    method = getattr(kernel, method_name, None)
    if method is None:
        return None
    if callable(method):
        return method()
    return method


def _kernel_text(kernel: object, method_name: str, *args) -> str:
    method = getattr(kernel, method_name, None)
    if method is None:
        return ""
    return method(*args) or ""


def _runtime_env(compile_command: list[str], kernel_env: dict[str, str]) -> dict[str, str] | None:
    lib_dirs = [flag[2:] for flag in compile_command if flag.startswith("-L")]
    if not lib_dirs and not kernel_env:
        return None

    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH")
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    env.update(kernel_env)
    return env


def _rep_c_type(field_name: str, rep: Rep) -> str:
    if field_name in {"val", "values", "data"}:
        return "double"
    if any(isinstance(value, float) for value in rep.values):
        return "double"
    return "int"


def _bind_reps(plan: Plan) -> list[RepBinding]:
    used_names: set[str] = set()
    bindings: list[RepBinding] = []

    for dispatch in plan.dispatches:
        fmt = dispatch.fmt
        if not is_dataclass(fmt):
            continue
        for field in fields(fmt):
            value = getattr(fmt, field.name)
            if not isinstance(value, Rep):
                continue

            base = _sanitize_identifier(value.label or f"{type(fmt).__name__.lower()}_{field.name}")
            c_name = base
            suffix = 1
            while c_name in used_names:
                suffix += 1
                c_name = f"{base}_{suffix}"
            used_names.add(c_name)
            value.c_name = c_name
            bindings.append(
                RepBinding(
                    rep=value,
                    label=c_name,
                    c_name=c_name,
                    c_type=_rep_c_type(field.name, value),
                )
            )

    return bindings


def _format_array(values: list[Any]) -> str:
    return ",".join(map(str, values))


def _write_sabledata(data_path: str, bindings: list[RepBinding]) -> None:
    with open(data_path, "w") as f:
        for binding in bindings:
            f.write(f"{binding.label}=[{_format_array(binding.rep.values)}]\n")


def _c_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _reader_helpers() -> str:
    return r"""
static void skip_to_array(FILE *file) {
    int c;
    while ((c = fgetc(file)) != EOF && c != '[') {}
    if (c == EOF) {
        fprintf(stderr, "Unexpected end of data file\n");
        exit(1);
    }
}

static void finish_array_line(FILE *file) {
    int c;
    while ((c = fgetc(file)) != EOF && c != '\n') {}
}

static void read_double_array(FILE *file, double *out, int size) {
    skip_to_array(file);
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &out[i]) != 1) {
            fprintf(stderr, "Failed to read double array\n");
            exit(1);
        }
        if (i + 1 < size) {
            int comma = fgetc(file);
            if (comma != ',') {
                fprintf(stderr, "Malformed double array\n");
                exit(1);
            }
        }
    }
    finish_array_line(file);
}

static void read_int_array(FILE *file, int *out, int size) {
    skip_to_array(file);
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%d", &out[i]) != 1) {
            fprintf(stderr, "Failed to read int array\n");
            exit(1);
        }
        if (i + 1 < size) {
            int comma = fgetc(file);
            if (comma != ',') {
                fprintf(stderr, "Malformed int array\n");
                exit(1);
            }
        }
    }
    finish_array_line(file);
}

static void read_dense_input(FILE *file, double *out, int size) {
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &out[i]) != 1) {
            fprintf(stderr, "Failed to read dense input\n");
            exit(1);
        }
        if (i + 1 < size) {
            int comma = fgetc(file);
            if (comma != ',') {
                fprintf(stderr, "Malformed dense input\n");
                exit(1);
            }
        }
    }
}
"""


def _alloc_size(size: int) -> int:
    return max(size, 1)


def _output_size(plan: Plan) -> int:
    rhs = plan.rhs_input
    if len(rhs.shape) == 1:
        return plan.matrix.nrows
    return plan.matrix.nrows * rhs.shape[1]


def _input_size(plan: Plan) -> int:
    rhs = plan.rhs_input
    if len(rhs.shape) == 1:
        return rhs.shape[0]
    return rhs.shape[0] * rhs.shape[1]


def build_compile_command_for_plan(plan: Plan, c_path: str, output_path: str) -> list[str]:
    compiler = "gcc"
    pre_source_flags: list[str] = []
    source_files: list[str] = []
    compile_flags: list[str] = []
    link_flags: list[str] = []
    for dispatch in plan.dispatches:
        requested_compiler = _kernel_value(dispatch.kernel, "compiler")
        if requested_compiler is not None:
            if compiler != "gcc" and compiler != requested_compiler:
                raise ValueError(f"Conflicting compiler requirements: {compiler} and {requested_compiler}")
            compiler = str(requested_compiler)
        pre_source_flags.extend(_kernel_list(dispatch.kernel, "pre_source_flags"))
        source_files.extend(_kernel_list(dispatch.kernel, "source_files"))
        compile_flags.extend(_kernel_list(dispatch.kernel, "compile_flags"))
        link_flags.extend(_kernel_list(dispatch.kernel, "link_flags"))

    reset_language = ["-x", "none"] if "-x" in pre_source_flags else []

    return (
        [compiler]
        + _dedupe(pre_source_flags)
        + [os.path.abspath(c_path)]
        + _dedupe(source_files)
        + reset_language
        + ["-o", os.path.abspath(output_path)]
        + list(CFLAGS)
        + _dedupe(compile_flags)
        + _dedupe(link_flags)
    )


def _collect_runtime_env(plan: Plan) -> dict[str, str]:
    env: dict[str, str] = {}
    for dispatch in plan.dispatches:
        env.update(_kernel_dict(dispatch.kernel, "runtime_env"))
    return env


def _collect_runtime_cwd(plan: Plan) -> str | None:
    runtime_cwd: str | None = None
    for dispatch in plan.dispatches:
        cwd = _kernel_value(dispatch.kernel, "runtime_cwd")
        if cwd is None:
            continue
        cwd = os.path.abspath(str(cwd))
        if runtime_cwd is not None and runtime_cwd != cwd:
            raise ValueError(f"Conflicting runtime working directories: {runtime_cwd} and {cwd}")
        runtime_cwd = cwd
    return runtime_cwd


def _emit_source(plan: Plan, data_path: str, bindings: list[RepBinding], bench: int) -> str:
    rhs = plan.rhs_input
    y_size = _output_size(plan)
    x_size = _input_size(plan)

    includes = [
        "#include <assert.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#include <time.h>",
    ]
    helpers = [_reader_helpers()]
    setup: list[str] = []
    calls: list[tuple[str, str, int | None]] = []
    teardown: list[str] = []

    for dispatch in plan.dispatches:
        includes.extend(_kernel_list(dispatch.kernel, "emit_includes"))
        helpers.append(_kernel_text(dispatch.kernel, "emit_helpers"))
        setup.append(_kernel_text(dispatch.kernel, "emit_setup", dispatch.fmt, rhs))
        call = getattr(dispatch.kernel, "emit_call", None)
        if call is None:
            raise TypeError(f"{type(dispatch.kernel).__name__} must implement emit_call(...)")
        timed_calls = getattr(dispatch.kernel, "emit_timed_calls", None)
        if isinstance(dispatch.fmt, VBR) and callable(timed_calls):
            dense_block_index = sum(1 for category, _, _ in calls if category == "dense")
            for snippet in timed_calls(dispatch.fmt, "y", "x", rhs) or []:
                calls.append(("dense", snippet or "", dense_block_index))
                dense_block_index += 1
        else:
            category = "dense" if isinstance(dispatch.fmt, VBR) else "sparse"
            if not isinstance(dispatch.fmt, (CSR, VBR)):
                category = "sparse"
            calls.append((category, call(dispatch.fmt, "y", "x", rhs) or "", None))
        teardown.append(_kernel_text(dispatch.kernel, "emit_teardown", dispatch.fmt, rhs))

    lines: list[str] = []
    lines.extend(f"{include}\n" for include in _dedupe(includes))
    lines.append("\n")
    lines.extend(f"{helper}\n" for helper in helpers if helper)
    lines.append("\nint main(void) {\n")
    lines.append(f"    double *y = (double *)calloc({_alloc_size(y_size)}, sizeof(double));\n")
    lines.append(f"    double *x = (double *)malloc({_alloc_size(x_size)} * sizeof(double));\n")
    lines.append("    assert(y != NULL);\n")
    lines.append("    assert(x != NULL);\n")
    for binding in bindings:
        lines.append(
            f"    {binding.c_type} *{binding.c_name} = ({binding.c_type} *)"
            f"malloc({_alloc_size(binding.size)} * sizeof({binding.c_type}));\n"
        )
        lines.append(f"    assert({binding.c_name} != NULL);\n")

    lines.append(f'    FILE *matrix_file = fopen("{_c_string(os.path.abspath(data_path))}", "r");\n')
    lines.append("    assert(matrix_file != NULL);\n")
    for binding in bindings:
        reader = "read_double_array" if binding.c_type == "double" else "read_int_array"
        lines.append(f"    {reader}(matrix_file, {binding.c_name}, {binding.size});\n")
    lines.append("    fclose(matrix_file);\n")

    lines.append(f'    FILE *rhs_file = fopen("{_c_string(os.path.abspath(rhs.path))}", "r");\n')
    lines.append("    assert(rhs_file != NULL);\n")
    lines.append(f"    read_dense_input(rhs_file, x, {x_size});\n")
    lines.append("    fclose(rhs_file);\n\n")

    for snippet in setup:
        if snippet:
            lines.append(snippet)
            if not snippet.endswith("\n"):
                lines.append("\n")

    lines.append("    struct timespec t1, t2;\n")
    lines.append(f"    double *sparse_times = (double *)calloc({bench}, sizeof(double));\n")
    lines.append(f"    double *dense_times = (double *)calloc({bench}, sizeof(double));\n")
    dense_block_count = sum(1 for category, _, _ in calls if category == "dense")
    lines.append(
        f"    double (*dense_block_times)[{bench}] = "
        f"(double (*)[{bench}])calloc({_alloc_size(dense_block_count)}, {bench} * sizeof(double));\n"
    )
    lines.append("    assert(sparse_times != NULL);\n")
    lines.append("    assert(dense_times != NULL);\n")
    lines.append("    assert(dense_block_times != NULL);\n")
    lines.append(f"    for (int iter = 0; iter < {bench}; iter++) {{\n")
    lines.append(f"        memset(y, 0, {y_size} * sizeof(double));\n")
    lines.append("        double iter_sparse_ns = 0.0;\n")
    lines.append("        double iter_dense_ns = 0.0;\n")
    for category, snippet, block_index in calls:
        if snippet:
            lines.append("        clock_gettime(CLOCK_MONOTONIC, &t1);\n")
            lines.append(snippet)
            if not snippet.endswith("\n"):
                lines.append("\n")
            lines.append("        clock_gettime(CLOCK_MONOTONIC, &t2);\n")
            target = "iter_dense_ns" if category == "dense" else "iter_sparse_ns"
            lines.append(
                f"        {target} += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + "
                "(t2.tv_nsec - t1.tv_nsec);\n"
            )
            if block_index is not None:
                lines.append(
                    f"        dense_block_times[{block_index}][iter] = "
                    "(t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);\n"
                )
    lines.append("        sparse_times[iter] = iter_sparse_ns;\n")
    lines.append("        dense_times[iter] = iter_dense_ns;\n")
    lines.append("    }\n\n")

    for snippet in teardown:
        if snippet:
            lines.append(snippet)
            if not snippet.endswith("\n"):
                lines.append("\n")

    lines.append('    printf("Sparse: ");\n')
    lines.append(f"    for (int i = 0; i < {bench}; i++) {{\n")
    lines.append('        printf("%.0f,", sparse_times[i]);\n')
    lines.append("    }\n")
    lines.append('    printf("\\n");\n')
    lines.append('    printf("Dense: ");\n')
    lines.append(f"    for (int i = 0; i < {bench}; i++) {{\n")
    lines.append('        printf("%.0f,", dense_times[i]);\n')
    lines.append("    }\n")
    lines.append('    printf("\\n");\n')
    for block_index in range(dense_block_count):
        lines.append(f'    printf("Dense Block {block_index + 1}: ");\n')
        lines.append(f"    for (int i = 0; i < {bench}; i++) {{\n")
        lines.append(f'        printf("%.0f,", dense_block_times[{block_index}][i]);\n')
        lines.append("    }\n")
        lines.append('    printf("\\n");\n')
    lines.append('    printf("\\n");\n')
    lines.append(f"    for (int i = 0; i < {y_size}; i++) {{\n")
    lines.append('        printf("%.17g\\n", y[i]);\n')
    lines.append("    }\n")

    for binding in bindings:
        lines.append(f"    free({binding.c_name});\n")
    lines.append("    free(dense_block_times);\n")
    lines.append("    free(dense_times);\n")
    lines.append("    free(sparse_times);\n")
    lines.append("    free(x);\n")
    lines.append("    free(y);\n")
    lines.append("    return 0;\n")
    lines.append("}\n")
    return "".join(lines)


def compile(plan: Plan, filename: str | None = None, bench: int = 5, threads: int = 1) -> CompiledExecutor:
    if threads != 1:
        raise ValueError("The frontend compiler is single-threaded for now")
    if bench <= 0:
        raise ValueError("bench must be positive")

    _ = plan.rhs_input
    plan.ensure_complete()

    os.makedirs(plan.artifact_dir, exist_ok=True)
    filename = filename or plan.matrix.name
    data_path = os.path.join(plan.artifact_dir, f"{filename}.sabledata")
    c_path = os.path.join(plan.artifact_dir, f"{filename}.c")

    start = time.time_ns() // 1_000_000
    bindings = _bind_reps(plan)
    _write_sabledata(data_path, bindings)
    source = _emit_source(plan, data_path, bindings, bench)
    with open(c_path, "w") as f:
        f.write(source)
    end = time.time_ns() // 1_000_000

    return CompiledExecutor(
        c_path=c_path,
        data_path=data_path,
        artifact_dir=plan.artifact_dir,
        filename=filename,
        codegen_time_ms=end - start,
        plan=plan,
        runtime_env=_collect_runtime_env(plan),
        runtime_cwd=_collect_runtime_cwd(plan),
    )
