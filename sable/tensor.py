from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DenseLayout(Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


@dataclass
class DenseInput:
    path: str
    shape: tuple[int, ...]
    layout: DenseLayout | None = None

    @classmethod
    def vector(cls, path: str, size: int) -> "DenseInput":
        return cls(path=path, shape=(size,))

    @classmethod
    def matrix(
        cls,
        path: str,
        shape: tuple[int, int],
        layout: DenseLayout,
    ) -> "DenseInput":
        return cls(path=path, shape=shape, layout=layout)
