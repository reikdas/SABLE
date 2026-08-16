from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from inspect import Parameter, signature
from typing import Any

from sable.build_config import CFLAGS

from .codegen import OutOfLineCode
from .formats import Format, Rep
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


@dataclass
class TimedCall:
    dispatch_index: int
    snippet: str
    part_index: int
    part_ordinal: int


KernelSnippet = str | OutOfLineCode


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


def _normalize_kernel_snippets(value: object) -> list[KernelSnippet]:
    if value is None:
        return []
    if isinstance(value, (str, OutOfLineCode)):
        return [value]
    if isinstance(value, Iterable):
        snippets: list[KernelSnippet] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (str, OutOfLineCode)):
                snippets.append(item)
                continue
            raise TypeError(f"Kernel emitted unsupported snippet type: {type(item).__name__}")
        return snippets
    raise TypeError(f"Kernel emitted unsupported snippet type: {type(value).__name__}")


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


def _kernel_context_text(kernel: object, method_name: str, *args) -> str:
    method = getattr(kernel, method_name, None)
    if method is None:
        return ""
    try:
        params = list(signature(method).parameters.values())
    except (TypeError, ValueError):
        return method(*args) or ""
    if any(param.kind == Parameter.VAR_POSITIONAL for param in params):
        return method(*args) or ""
    positional_params = [
        param
        for param in params
        if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return method(*args[: len(positional_params)]) or ""


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

    def bind(rep: Rep, fallback_label: str, field_name: str) -> None:
        base = _sanitize_identifier(rep.label or fallback_label)
        c_name = base
        suffix = 1
        while c_name in used_names:
            suffix += 1
            c_name = f"{base}_{suffix}"
        used_names.add(c_name)
        rep.c_name = c_name
        bindings.append(
            RepBinding(
                rep=rep,
                label=c_name,
                c_name=c_name,
                c_type=_rep_c_type(field_name, rep),
            )
        )

    for dispatch in plan.dispatches:
        fmt = dispatch.fmt
        if is_dataclass(fmt):
            for field in fields(fmt):
                value = getattr(fmt, field.name)
                if isinstance(value, Rep):
                    bind(value, f"{type(fmt).__name__.lower()}_{field.name}", field.name)
        # Kernels may stage dense operands of their own, e.g. the additive
        # input of an update kernel dispatched without a format.
        for attr_name, value in vars(dispatch.kernel).items():
            if isinstance(value, Rep):
                bind(value, f"{type(dispatch.kernel).__name__.lower()}_{attr_name}", attr_name)

    return bindings


def _write_array_values(file, values: list[Any], chunk_size: int = 65536) -> None:
    for start in range(0, len(values), chunk_size):
        if start:
            file.write(",")
        file.write(",".join(map(str, values[start:start + chunk_size])))


def _write_sabledata(data_path: str, bindings: list[RepBinding]) -> None:
    with open(data_path, "w") as f:
        for binding in bindings:
            f.write(f"{binding.label}=[")
            _write_array_values(f, binding.rep.values)
            f.write("]\n")


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


def _collect_loop_hooks(plan: Plan) -> tuple[str, str, list[str]]:
    """Gather the iteration-loop hooks emitted by the dispatched kernels.

    Returns (state, init, conditions): C declarations emitted once before the
    benchmark loop, statements run before each benchmark repetition, and the
    C expressions (one per emitting kernel) that keep the loop running.
    """
    state_parts: list[str] = []
    init_parts: list[str] = []
    conditions: list[str] = []
    for dispatch in plan.dispatches:
        state = _kernel_text(dispatch.kernel, "emit_loop_state")
        if state:
            state_parts.append(state)
        init = _kernel_text(dispatch.kernel, "emit_loop_init")
        if init:
            init_parts.append(init)
        condition = _kernel_text(dispatch.kernel, "emit_loop_condition")
        if condition.strip():
            conditions.append(condition.strip())
    return "".join(state_parts), "".join(init_parts), conditions


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


def _dispatch_reps(fmt: Format | None, kernel: object) -> list[Rep]:
    reps: list[Rep] = []
    if is_dataclass(fmt):
        for field in fields(fmt):
            value = getattr(fmt, field.name)
            if isinstance(value, Rep):
                reps.append(value)
    for value in vars(kernel).values():
        if isinstance(value, Rep):
            reps.append(value)
    return reps


def _bindings_used_by_body(
    body: str, fmt: Format | None, kernel: object, bindings: list[RepBinding]
) -> list[RepBinding]:
    reps = _dispatch_reps(fmt, kernel)
    return [
        binding
        for binding in bindings
        if any(binding.rep is rep for rep in reps) and binding.c_name in body
    ]


def _out_of_line_helper_name(
    snippet: OutOfLineCode,
    dispatch_index: int,
    part_ordinal: int | None,
) -> str:
    if snippet.name:
        return _sanitize_identifier(snippet.name)
    base = f"sable_out_of_line_dispatch_{dispatch_index + 1}"
    if part_ordinal is not None:
        return f"{base}_part_{part_ordinal}"
    return base


def _emit_out_of_line_helper(
    name: str,
    body: str,
    used_bindings: list[RepBinding],
    parameters: tuple[str, ...],
) -> str:
    params = ["double *y", "const double *x"]
    params.extend(f"const {binding.c_type} *{binding.c_name}" for binding in used_bindings)
    params.extend(parameters)
    return f"""\
static void {name}({", ".join(params)}) {{
{body}}}
"""


def _emit_out_of_line_call(
    name: str,
    used_bindings: list[RepBinding],
    arguments: tuple[str, ...],
) -> str:
    args = ["y", "x"]
    args.extend(binding.c_name for binding in used_bindings)
    args.extend(arguments)
    return f"{name}({', '.join(args)});\n"


def _lower_kernel_snippet(
    snippet: KernelSnippet,
    dispatch_index: int,
    part_ordinal: int | None,
    fmt: Format | None,
    kernel: object,
    bindings: list[RepBinding],
    helpers: list[str],
    helper_definitions: dict[str, str],
) -> str:
    if isinstance(snippet, str):
        return snippet
    name = _out_of_line_helper_name(snippet, dispatch_index, part_ordinal)
    used_bindings = _bindings_used_by_body(snippet.body, fmt, kernel, bindings)
    helper = _emit_out_of_line_helper(name, snippet.body, used_bindings, snippet.parameters)
    existing = helper_definitions.get(name)
    if existing is None:
        helper_definitions[name] = helper
        helpers.append(helper)
    elif existing != helper:
        raise ValueError(f"Conflicting out_of_line helper definition for {name}")
    return _emit_out_of_line_call(name, used_bindings, snippet.arguments)


def _emit_source(
    plan: Plan,
    data_path: str,
    bindings: list[RepBinding],
    bench: int,
    loop_hooks: tuple[str, str, list[str]],
) -> str:
    rhs = plan.rhs_input
    y_size = _output_size(plan)
    x_size = _input_size(plan)
    iteration = plan.iteration

    includes = [
        "#include <assert.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#include <time.h>",
    ]
    helpers = [_reader_helpers()]
    setup: list[str] = []
    calls: list[TimedCall] = []
    teardown: list[str] = []
    part_count = 0
    out_of_line_helper_definitions: dict[str, str] = {}

    for dispatch_index, dispatch in enumerate(plan.dispatches):
        includes.extend(_kernel_list(dispatch.kernel, "emit_includes"))
        helpers.append(_kernel_context_text(dispatch.kernel, "emit_helpers", dispatch.fmt, rhs))
        setup.append(_kernel_text(dispatch.kernel, "emit_setup", dispatch.fmt, rhs))
        call = getattr(dispatch.kernel, "emit_call", None)
        if call is None:
            raise TypeError(f"{type(dispatch.kernel).__name__} must implement emit_call(...)")

        timed_calls = getattr(dispatch.kernel, "emit_timed_calls", None)
        if callable(timed_calls):
            snippets = _normalize_kernel_snippets(timed_calls(dispatch.fmt, "y", "x", rhs))
            for part_ordinal, snippet in enumerate(snippets, start=1):
                lowered = _lower_kernel_snippet(
                    snippet,
                    dispatch_index,
                    part_ordinal,
                    dispatch.fmt,
                    dispatch.kernel,
                    bindings,
                    helpers,
                    out_of_line_helper_definitions,
                )
                calls.append(
                    TimedCall(
                        dispatch_index=dispatch_index,
                        snippet=lowered or "",
                        part_index=part_count,
                        part_ordinal=part_ordinal,
                    )
                )
                part_count += 1
        else:
            snippets = _normalize_kernel_snippets(call(dispatch.fmt, "y", "x", rhs))
            if len(snippets) <= 1:
                snippet = snippets[0] if snippets else ""
                lowered = _lower_kernel_snippet(
                    snippet,
                    dispatch_index,
                    1,
                    dispatch.fmt,
                    dispatch.kernel,
                    bindings,
                    helpers,
                    out_of_line_helper_definitions,
                )
                calls.append(TimedCall(
                    dispatch_index=dispatch_index,
                    snippet=lowered or "",
                    part_index=part_count,
                    part_ordinal=1,
                ))
                part_count += 1
            else:
                for part_ordinal, snippet in enumerate(snippets, start=1):
                    lowered = _lower_kernel_snippet(
                        snippet,
                        dispatch_index,
                        part_ordinal,
                        dispatch.fmt,
                        dispatch.kernel,
                        bindings,
                        helpers,
                        out_of_line_helper_definitions,
                    )
                    calls.append(
                        TimedCall(
                            dispatch_index=dispatch_index,
                            snippet=lowered or "",
                            part_index=part_count,
                            part_ordinal=part_ordinal,
                        )
                    )
                    part_count += 1
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
    if iteration is not None:
        # Iterative plans may mutate x, so keep a pristine copy of the dense
        # input to restore before each benchmark repetition.
        lines.append(f"    double *sable_x0 = (double *)malloc({_alloc_size(x_size)} * sizeof(double));\n")
        lines.append("    assert(sable_x0 != NULL);\n")
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
    if iteration is not None:
        lines.append(f"    read_dense_input(rhs_file, sable_x0, {x_size});\n")
        lines.append(f"    memcpy(x, sable_x0, {x_size} * sizeof(double));\n")
    else:
        lines.append(f"    read_dense_input(rhs_file, x, {x_size});\n")
    lines.append("    fclose(rhs_file);\n\n")

    for snippet in setup:
        if snippet:
            lines.append(snippet)
            if not snippet.endswith("\n"):
                lines.append("\n")

    lines.append("    struct timespec t1, t2;\n")
    dispatch_count = len(plan.dispatches)
    lines.append(
        f"    double (*dispatch_part_times)[{bench}] = "
        f"(double (*)[{bench}])calloc({_alloc_size(part_count)}, {bench} * sizeof(double));\n"
    )
    lines.append("    assert(dispatch_part_times != NULL);\n")
    if iteration is not None:
        loop_state, loop_init, loop_conditions = loop_hooks
        lines.append(f"    int *sable_iteration_counts = (int *)calloc({bench}, sizeof(int));\n")
        lines.append("    assert(sable_iteration_counts != NULL);\n")
        if loop_state:
            lines.append(loop_state)
            if not loop_state.endswith("\n"):
                lines.append("\n")
    lines.append(f"    for (int iter = 0; iter < {bench}; iter++) {{\n")
    # Inside the iteration loop, the per-part times accumulate ("+=") across
    # the algorithm iterations of a benchmark repetition.
    time_assign = "="
    if iteration is not None:
        lines.append(f"        memcpy(x, sable_x0, {x_size} * sizeof(double));\n")
        if loop_init:
            lines.append(loop_init)
            if not loop_init.endswith("\n"):
                lines.append("\n")
        lines.append("        int sable_iteration = 0;\n")
        condition_parts = []
        if iteration.max_iterations is not None:
            condition_parts.append(f"sable_iteration < {iteration.max_iterations}")
        condition_parts.extend(f"({condition})" for condition in loop_conditions)
        lines.append(f"        while ({' && '.join(condition_parts)}) {{\n")
        time_assign = "+="
    lines.append(f"        memset(y, 0, {y_size} * sizeof(double));\n")
    for timed_call in calls:
        if timed_call.snippet:
            lines.append("        clock_gettime(CLOCK_MONOTONIC, &t1);\n")
            lines.append(timed_call.snippet)
            if not timed_call.snippet.endswith("\n"):
                lines.append("\n")
            lines.append("        clock_gettime(CLOCK_MONOTONIC, &t2);\n")
            lines.append(
                f"        dispatch_part_times[{timed_call.part_index}][iter] {time_assign} "
                "(t2.tv_sec - t1.tv_sec) * 1000000000.0 + "
                "(t2.tv_nsec - t1.tv_nsec);\n"
            )
    if iteration is not None:
        lines.append("        sable_iteration++;\n")
        lines.append("        }\n")
        lines.append("        sable_iteration_counts[iter] = sable_iteration;\n")
    lines.append("    }\n\n")

    for snippet in teardown:
        if snippet:
            lines.append(snippet)
            if not snippet.endswith("\n"):
                lines.append("\n")

    parts_by_dispatch: dict[int, list[int]] = {}
    for timed_call in calls:
        if timed_call.snippet:
            parts_by_dispatch.setdefault(timed_call.dispatch_index, []).append(timed_call.part_index)

    for dispatch_index in range(dispatch_count):
        lines.append(f'    printf("Dispatch {dispatch_index + 1}: ");\n')
        lines.append(f"    for (int i = 0; i < {bench}; i++) {{\n")
        part_indices = parts_by_dispatch.get(dispatch_index, [])
        if part_indices:
            sum_expr = " + ".join(f"dispatch_part_times[{pi}][i]" for pi in part_indices)
        else:
            # A dispatch with no timed parts (e.g. an empty VBR) does no work, so
            # it takes 0 ns.  Emit a double literal: "%.0f" against an int 0 is
            # undefined behaviour and prints stale garbage from an SSE register.
            sum_expr = "0.0"
        lines.append(f'        printf("%.0f,", {sum_expr});\n')
        lines.append("    }\n")
        lines.append('    printf("\\n");\n')
    for timed_call in calls:
        if len(parts_by_dispatch.get(timed_call.dispatch_index, [])) <= 1:
            continue
        lines.append(f'    printf("Dispatch {timed_call.dispatch_index + 1} Part {timed_call.part_ordinal}: ");\n')
        lines.append(f"    for (int i = 0; i < {bench}; i++) {{\n")
        lines.append(f'        printf("%.0f,", dispatch_part_times[{timed_call.part_index}][i]);\n')
        lines.append("    }\n")
        lines.append('    printf("\\n");\n')
    if iteration is not None:
        lines.append('    printf("Iterations: ");\n')
        lines.append(f"    for (int i = 0; i < {bench}; i++) {{\n")
        lines.append('        printf("%d,", sable_iteration_counts[i]);\n')
        lines.append("    }\n")
        lines.append('    printf("\\n");\n')
    lines.append('    printf("\\n");\n')
    lines.append(f"    for (int i = 0; i < {y_size}; i++) {{\n")
    lines.append('        printf("%.17g\\n", y[i]);\n')
    lines.append("    }\n")

    for binding in bindings:
        lines.append(f"    free({binding.c_name});\n")
    lines.append("    free(dispatch_part_times);\n")
    if iteration is not None:
        lines.append("    free(sable_iteration_counts);\n")
        lines.append("    free(sable_x0);\n")
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

    loop_hooks = _collect_loop_hooks(plan)
    iteration = plan.iteration
    if iteration is not None:
        _, _, loop_conditions = loop_hooks
        if not loop_conditions and iteration.max_iterations is None:
            raise ValueError(
                "Plan.iterate needs a dispatched kernel that emits a loop condition "
                "(emit_loop_condition) or an explicit max_iterations"
            )

    os.makedirs(plan.artifact_dir, exist_ok=True)
    filename = filename or plan.matrix.name
    data_path = os.path.join(plan.artifact_dir, f"{filename}.sabledata")
    c_path = os.path.join(plan.artifact_dir, f"{filename}.c")

    start = time.time_ns() // 1_000_000
    bindings = _bind_reps(plan)
    _write_sabledata(data_path, bindings)
    source = _emit_source(plan, data_path, bindings, bench, loop_hooks)
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


