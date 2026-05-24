from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy

from utils.fileio import parse_yaml_bands

from sable.formats import Rep, VDIA
from sable.matrix import ResidualMatrix


Band = dict[str, Any]
Segment = tuple[int, int, int, int]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VDIA_FINDER_ROOT = _REPO_ROOT / "find-submatrices"


def _pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list/tuple")
    return int(value[0]), int(value[1])


def _normalise_band(band: Any) -> tuple[int, list[Segment]]:
    if hasattr(band, "to_dict"):
        band = band.to_dict()
    if not isinstance(band, dict):
        raise TypeError(f"Band must be a dict-like object, got {type(band).__name__}")

    diag_offset = int(band["diag_offset"])
    segments: list[Segment] = []
    for raw_segment in band.get("segments", []):
        if hasattr(raw_segment, "to_dict"):
            raw_segment = raw_segment.to_dict()
        if not isinstance(raw_segment, dict):
            raise TypeError(f"Band segment must be a dict-like object, got {type(raw_segment).__name__}")
        row_start, row_end = _pair(raw_segment["rows"], "segment rows")
        lower_bw, upper_bw = _pair(raw_segment["bandwidth"], "segment bandwidth")
        if row_end <= row_start:
            raise ValueError(f"Segment row range must be non-empty, got [{row_start}, {row_end})")
        if lower_bw < 0 or upper_bw < 0:
            raise ValueError(f"Segment bandwidth must be non-negative, got [{lower_bw}, {upper_bw}]")
        segments.append((row_start, row_end, lower_bw, upper_bw))
    return diag_offset, segments


def _csr_value(csr, row: int, col: int) -> float:
    start = csr.indptr[row]
    end = csr.indptr[row + 1]
    pos = start + int(numpy.searchsorted(csr.indices[start:end], col, side="left"))
    if pos < end and int(csr.indices[pos]) == col:
        return float(csr.data[pos])
    return 0.0


def pack_bands_as_vdia(A: ResidualMatrix, bands: list[Any]) -> VDIA:
    csr = A.to_csr()
    csr.sort_indices()

    seg_row_start: list[int] = []
    seg_nrows: list[int] = []
    seg_ndiags: list[int] = []
    seg_val_ptr: list[int] = []
    seg_idiag_ptr: list[int] = []
    val: list[float] = []
    idiag: list[int] = []

    for band in bands:
        diag_offset, segments = _normalise_band(band)
        if not segments:
            continue

        for r0, r1, lower, upper in segments:
            if r0 < 0 or r1 > A.nrows:
                raise ValueError(f"Segment rows [{r0}, {r1}) are outside matrix with {A.nrows} rows")

            rows = r1 - r0
            ndiags = lower + upper + 1

            seg_row_start.append(r0)
            seg_nrows.append(rows)
            seg_ndiags.append(ndiags)
            seg_val_ptr.append(len(val))
            seg_idiag_ptr.append(len(idiag))

            for delta in range(-lower, upper + 1):
                idiag.append(diag_offset + delta)

            for delta in range(-lower, upper + 1):
                actual_diag = diag_offset + delta
                for row in range(r0, r1):
                    col = row + actual_diag
                    if 0 <= col < A.ncols:
                        val.append(_csr_value(csr, row, col))
                    else:
                        val.append(0.0)

    return VDIA(
        nrows=A.nrows,
        ncols=A.ncols,
        nsegments=len(seg_row_start),
        seg_row_start=seg_row_start,
        seg_nrows=seg_nrows,
        seg_ndiags=seg_ndiags,
        seg_val_ptr=seg_val_ptr,
        seg_idiag_ptr=seg_idiag_ptr,
        val=Rep(val, label="vdia_val"),
        idiag=Rep(idiag, label="vdia_idiag"),
    )


class BandExtractor:
    produces = VDIA

    def __init__(
        self,
        min_density: float = 0.5,
        min_segment_len: int = 20,
        min_region_len: int = 50,
        min_region_area: int = 2500,
        min_savings: float = 0.10,
        max_diag: int | None = None,
        max_bw: int | None = None,
        verbose: bool = False,
    ):
        self.min_density = min_density
        self.min_segment_len = min_segment_len
        self.min_region_len = min_region_len
        self.min_region_area = min_region_area
        self.min_savings = min_savings
        self.max_diag = max_diag
        self.max_bw = max_bw
        self.verbose = verbose

    def extract(self, A: ResidualMatrix) -> tuple[VDIA, ResidualMatrix]:
        if str(_VDIA_FINDER_ROOT) not in sys.path:
            sys.path.insert(0, str(_VDIA_FINDER_ROOT))
        from find_vdia import find_vdia_regions

        regions = find_vdia_regions(
            A.to_csr(),
            min_density=self.min_density,
            min_segment_len=self.min_segment_len,
            min_region_len=self.min_region_len,
            min_region_area=self.min_region_area,
            min_savings=self.min_savings,
            max_diag=self.max_diag,
            max_bw=self.max_bw,
            verbose=self.verbose,
        )
        fmt = pack_bands_as_vdia(A, [region.to_dict() for region in regions])
        return fmt, A.without(fmt)


class BandExtractorSkip:
    produces = VDIA

    def __init__(self, bands: list[Any] | None = None, yaml_path: str | pathlib.Path | None = None):
        if bands is not None and yaml_path is not None:
            raise ValueError("Pass either bands or yaml_path, not both")
        self.bands = parse_yaml_bands(str(yaml_path)) if yaml_path is not None else list(bands or [])

    @classmethod
    def from_yaml(cls, yaml_path: str | pathlib.Path) -> "BandExtractorSkip":
        return cls(yaml_path=yaml_path)

    def extract(self, A: ResidualMatrix) -> tuple[VDIA, ResidualMatrix]:
        fmt = pack_bands_as_vdia(A, self.bands)
        return fmt, A.without(fmt)
