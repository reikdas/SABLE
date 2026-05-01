from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

import scipy.io

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
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, _, _, _ = (
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
        bpntre=list(map(int, bpntre)),
        ublocks=list(map(int, ublocks)),
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


def find_blocks(
    A: ResidualMatrix,
    min_density: float,
    min_area: int,
    gamma: float,
    timeout_seconds: float,
    threads: int,
    partitioner_path: str | os.PathLike[str] | None = None,
) -> list[Block]:
    partitioner = _ensure_partitioner(partitioner_path)
    with tempfile.TemporaryDirectory(prefix="sable-block-detector-") as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        matrix_path = tmp_path / f"{A.name}.mtx"
        output_path = tmp_path / f"{A.name}.yaml"
        scipy.io.mmwrite(str(matrix_path), A.to_scipy())

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
        return parse_yaml_blocks(str(output_path))


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

    def extract(self, A: ResidualMatrix) -> tuple[VBR, ResidualMatrix]:
        blocks = find_blocks(
            A,
            self.min_density,
            self.min_area,
            self.gamma,
            self.timeout_seconds,
            self.threads,
            self.partitioner_path,
        )
        fmt = pack_blocks_as_vbr(A, blocks)
        return fmt, A.without(fmt)


class BlockDetectorSkip:
    produces = VBR

    def __init__(self, blocks: list[Block]):
        self.blocks = list(blocks)

    def extract(self, A: ResidualMatrix) -> tuple[VBR, ResidualMatrix]:
        fmt = pack_blocks_as_vbr(A, self.blocks)
        return fmt, A.without(fmt)
