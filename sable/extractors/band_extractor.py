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

    region_ptr: list[int] = [0]
    diag_offsets: list[int] = []
    row_start: list[int] = []
    row_end: list[int] = []
    lower_bw: list[int] = []
    upper_bw: list[int] = []
    data_ptr: list[int] = [0]
    data: list[float] = []

    for band in bands:
        diag_offset, segments = _normalise_band(band)
        if not segments:
            continue

        diag_offsets.append(diag_offset)
        for r0, r1, lower, upper in segments:
            if r0 < 0 or r1 > A.nrows:
                raise ValueError(f"Segment rows [{r0}, {r1}) are outside matrix with {A.nrows} rows")

            row_start.append(r0)
            row_end.append(r1)
            lower_bw.append(lower)
            upper_bw.append(upper)

            rows = r1 - r0
            for delta in range(-lower, upper + 1):
                actual_diag = diag_offset + delta
                for row in range(r0, r1):
                    col = row + actual_diag
                    if 0 <= col < A.ncols:
                        data.append(_csr_value(csr, row, col))
                    else:
                        data.append(0.0)
            data_ptr.append(len(data))

        region_ptr.append(len(row_start))

    return VDIA(
        nrows=A.nrows,
        ncols=A.ncols,
        nregions=len(diag_offsets),
        nsegments=len(row_start),
        region_ptr=region_ptr,
        diag_offsets=diag_offsets,
        row_start=row_start,
        row_end=row_end,
        lower_bw=lower_bw,
        upper_bw=upper_bw,
        data_ptr=data_ptr,
        data=Rep(data, label="vdia_data"),
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
