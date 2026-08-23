from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import time
from typing import Any

import scipy.io
import yaml

from utils.fileio import parse_yaml_blocks

from sable.formats import Rep, VBR
from sable.matrix import ResidualMatrix

from .vbr_packing import convert_matrix_to_vbrc_with_blocks


Block = tuple[int, int, int, int]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PARTITIONER_ROOT = _REPO_ROOT / "find-submatrices"
_PARTITIONER_BUILD_DIR = _PARTITIONER_ROOT / "build"
_PARTITIONER_BIN = _PARTITIONER_BUILD_DIR / "partition_matrix"


def pack_blocks_as_vbr(A: ResidualMatrix, blocks: list[Block]) -> VBR:
    val, indx, bindx, rpntr, cpntr, bpntrb, _, _, _ = (
        convert_matrix_to_vbrc_with_blocks(A.to_scipy(), blocks)
    )
    return VBR(
        nrows=A.nrows,
        ncols=A.ncols,
        val=Rep(list(map(float, val)), label="vbr_val"),
        indx=list(map(int, indx)),
        bindx=list(map(int, bindx)),
        rpntr=list(map(int, rpntr)),
        cpntr=list(map(int, cpntr)),
        bpntrb=list(map(int, bpntrb)),
        blocks=list(blocks),
    )


def _ensure_partitioner(partitioner_path: str | os.PathLike[str] | None = None) -> pathlib.Path:
    if partitioner_path is not None:
        path = pathlib.Path(partitioner_path)
        if not path.exists():
            raise FileNotFoundError(f"Block partitioner not found: {path}")
        return path

    if _PARTITIONER_BIN.exists():
        return _PARTITIONER_BIN

    configure = ["cmake", "-S", str(_PARTITIONER_ROOT), "-B", str(_PARTITIONER_BUILD_DIR)]
    build = [
        "cmake",
        "--build",
        str(_PARTITIONER_BUILD_DIR),
        "--target",
        "partition_matrix",
        "-j",
        str(max(1, min(os.cpu_count() or 1, 8))),
    ]
    try:
        subprocess.run(configure, check=True, cwd=str(_REPO_ROOT), capture_output=True, text=True)
        subprocess.run(build, check=True, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Failed to build find-submatrices partition_matrix. "
            f"Run {' '.join(configure)} and {' '.join(build)} manually for details."
        ) from exc

    if not _PARTITIONER_BIN.exists():
        raise FileNotFoundError(f"Expected partitioner binary was not produced: {_PARTITIONER_BIN}")
    return _PARTITIONER_BIN


def _parse_partitioner_yaml(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    blocks: list[dict[str, Any]] = []
    for block in data.get("blocks") or []:
        rows = block.get("rows")
        cols = block.get("cols")
        if not rows or not cols:
            continue
        blocks.append(
            {
                "rows": [int(rows[0]), int(rows[1])],
                "cols": [int(cols[0]), int(cols[1])],
                "area": int(block.get("area", 0)),
                "density": float(block.get("density", 0.0)),
                "found_at_seconds": float(block.get("found_at_seconds", 0.0)),
            }
        )
    return {
        "timeout": bool(data.get("timeout", False)),
        "timeout_seconds_budget": float(data.get("timeout_seconds_budget", 0.0)),
        "partitioner_read_seconds": float(data.get("read_seconds", 0.0)),
        "partitioner_search_seconds": float(data.get("search_seconds", 0.0)),
        "blocks": blocks,
    }


def find_blocks_with_meta(
    A: ResidualMatrix,
    min_density: float,
    min_area: int,
    gamma: float,
    timeout_seconds: float,
    threads: int,
    partitioner_path: str | os.PathLike[str] | None = None,
) -> tuple[list[Block], dict[str, Any]]:
    partitioner = _ensure_partitioner(partitioner_path)
    with tempfile.TemporaryDirectory(prefix="sable-block-detector-") as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        matrix_path = tmp_path / f"{A.name}.mtx"
        output_path = tmp_path / f"{A.name}.yaml"
        t0 = time.perf_counter()
        scipy.io.mmwrite(str(matrix_path), A.to_scipy())
        t1 = time.perf_counter()

        command = [
            str(partitioner),
            str(matrix_path),
            "--min-density",
            str(min_density),
            "--min-area",
            str(min_area),
            "--gamma",
            str(gamma),
            "--threads",
            str(threads),
            "--timeout-seconds",
            str(timeout_seconds),
            "--output",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, cwd=str(_PARTITIONER_ROOT), capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Block partitioner failed:\n"
                f"command: {' '.join(command)}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc
        t2 = time.perf_counter()

        blocks = parse_yaml_blocks(str(output_path))
        meta = _parse_partitioner_yaml(str(output_path))
        meta["mmwrite_seconds"] = t1 - t0
        meta["partitioner_seconds"] = t2 - t1
        return blocks, meta


def find_blocks(
    A: ResidualMatrix,
    min_density: float,
    min_area: int,
    gamma: float,
    timeout_seconds: float,
    threads: int,
    partitioner_path: str | os.PathLike[str] | None = None,
) -> list[Block]:
    blocks, _ = find_blocks_with_meta(
        A, min_density, min_area, gamma, timeout_seconds, threads, partitioner_path
    )
    return blocks


class BlockDetector:
    produces = VBR

    def __init__(
        self,
        min_density: float = 0.5,
        min_area: int = 2500,
        gamma: float = 1.5,
        timeout_seconds: float = 4.0 * 60.0 * 60.0,
        threads: int = 20,
        partitioner_path: str | os.PathLike[str] | None = None,
    ):
        self.min_density = min_density
        self.min_area = min_area
        self.gamma = gamma
        self.timeout_seconds = timeout_seconds
        self.threads = threads
        self.partitioner_path = partitioner_path
        # Phase timings of the most recent extract() call, for inspection
        # benchmarking. Keys: mmwrite_seconds, partitioner_seconds,
        # partitioner_read_seconds, partitioner_search_seconds, pack_seconds,
        # residual_seconds, timeout, blocks (incl. per-block found_at_seconds).
        self.last_timing: dict[str, Any] | None = None

    def extract(self, A: ResidualMatrix) -> tuple[VBR, ResidualMatrix]:
        blocks, meta = find_blocks_with_meta(
            A,
            self.min_density,
            self.min_area,
            self.gamma,
            self.timeout_seconds,
            self.threads,
            self.partitioner_path,
        )
        t0 = time.perf_counter()
        fmt = pack_blocks_as_vbr(A, blocks)
        t1 = time.perf_counter()
        residual = A.without(fmt)
        t2 = time.perf_counter()
        meta["pack_seconds"] = t1 - t0
        meta["residual_seconds"] = t2 - t1
        self.last_timing = meta
        return fmt, residual


class BlockDetectorSkip:
    produces = VBR

    def __init__(self, blocks: list[Block]):
        self.blocks = list(blocks)

    def extract(self, A: ResidualMatrix) -> tuple[VBR, ResidualMatrix]:
        fmt = pack_blocks_as_vbr(A, self.blocks)
        return fmt, A.without(fmt)
