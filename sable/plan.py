from __future__ import annotations

from dataclasses import dataclass

from .formats import Format
from .matrix import Matrix, ResidualMatrix
from .operation import kernel_operation, rhs_rank_for_operation
from .tensor import DenseInput


@dataclass
class Dispatch:
    fmt: Format
    kernel: object


class Plan:
    def __init__(self, matrix: Matrix, artifact_dir: str):
        self.matrix = matrix
        self.artifact_dir = artifact_dir
        self._rhs: DenseInput | None = None
        self.dispatches: list[Dispatch] = []
        self._residual: ResidualMatrix = matrix
        self._pending_formats: dict[int, str] = {}

    @property
    def rhs_input(self) -> DenseInput:
        if self._rhs is None:
            raise ValueError("Plan.rhs(...) must be called before compile(plan)")
        return self._rhs

    def rhs(self, dense_input: DenseInput) -> None:
        if len(dense_input.shape) == 1:
            if dense_input.shape[0] != self.matrix.ncols:
                raise ValueError(
                    f"Vector RHS has size {dense_input.shape[0]}, expected {self.matrix.ncols}"
                )
        elif len(dense_input.shape) == 2:
            if dense_input.shape[0] != self.matrix.ncols:
                raise ValueError(
                    f"Matrix RHS has first dimension {dense_input.shape[0]}, expected {self.matrix.ncols}"
                )
        else:
            raise ValueError(f"Unsupported RHS rank: {len(dense_input.shape)}")
        self._rhs = dense_input

    @property
    def residual(self) -> ResidualMatrix:
        return self._residual

    def extract(self, extractor: object) -> Format:
        extract = getattr(extractor, "extract", None)
        if extract is None or not callable(extract):
            raise TypeError(f"{type(extractor).__name__} must implement extract(...)")

        before_nnz = self._residual.nnz
        fmt, residual = extract(self._residual)
        if not isinstance(fmt, Format):
            raise TypeError(f"{type(extractor).__name__}.extract(...) must return a Format")
        if not isinstance(residual, ResidualMatrix):
            raise TypeError(f"{type(extractor).__name__}.extract(...) must return a ResidualMatrix")

        produces = getattr(extractor, "produces", None)
        if produces is not None and not isinstance(fmt, produces):
            raise TypeError(
                f"{type(extractor).__name__} declares {produces.__name__}, got {type(fmt).__name__}"
            )
        if residual.nnz > before_nnz:
            raise ValueError(
                f"{type(extractor).__name__} increased residual nnz from {before_nnz} to {residual.nnz}"
            )

        self._residual = residual
        if residual.nnz < before_nnz:
            self._pending_formats[id(fmt)] = type(fmt).__name__
        return fmt

    def dispatch(self, fmt: Format, kernel: object) -> None:
        accepts = getattr(kernel, "accepts", None)
        if accepts is not None and not isinstance(fmt, accepts):
            raise TypeError(f"{type(kernel).__name__} expects {accepts.__name__}, got {type(fmt).__name__}")

        operation = kernel_operation(kernel)
        if (
            self._rhs is not None
            and operation is not None
            and rhs_rank_for_operation(operation) != len(self._rhs.shape)
        ):
            raise ValueError(
                f"{type(kernel).__name__} expects {operation.value.upper()}, got RHS rank {len(self._rhs.shape)}"
            )

        existing_operations = set()
        for dispatch in self.dispatches:
            existing_operation = kernel_operation(dispatch.kernel)
            if existing_operation is not None:
                existing_operations.add(existing_operation)
        if operation is not None and existing_operations and operation not in existing_operations:
            raise ValueError("All kernels in a plan must agree on operation")

        self.dispatches.append(Dispatch(fmt=fmt, kernel=kernel))
        self._pending_formats.pop(id(fmt), None)

    def ensure_complete(self) -> None:
        if self._residual.nnz != 0:
            raise ValueError(f"Plan is incomplete: residual has {self._residual.nnz} unclaimed entries")
        if self._pending_formats:
            pending = ", ".join(self._pending_formats.values())
            raise ValueError(f"Plan is incomplete: extracted format(s) have no dispatched kernel: {pending}")

    def compile(self, filename: str | None = None, bench: int = 5, threads: int = 1):
        from .compiler import compile as compile_plan

        return compile_plan(self, filename=filename, bench=bench, threads=threads)

    def interpret(self, filename: str | None = None, bench: int = 5, threads: int = 1):
        from .compiler import interpret as interpret_plan

        return interpret_plan(self, filename=filename, bench=bench, threads=threads)
