from __future__ import annotations

import pathlib
import sys
import time
from typing import Any, Callable

import numpy

from sable.formats import Rep, VDIA
from sable.matrix import ResidualMatrix
from utils.fileio import parse_yaml_bands

# The band finder lives in find-submatrices/, which is not a package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "find-submatrices"))
from find_vdia import (  # noqa: E402
    BAND_TIMEOUT_SECONDS,
    SPMM_MIN_COVERAGE,
    SPMV_MIN_COVERAGE,
    select_bands,
    spmm_slot_cost,
    spmv_slot_cost,
)


Segment = tuple[int, int, int, int]


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
    """Bands for a VDIA kernel whose stored slot of a band of w diagonals costs
    slot_cost(w) CSR nonzeros (find_vdia.select_bands): row-disjoint regions
    in row order, or none when they would hold less than `min_coverage` of the
    nonzeros or the search runs longer than `timeout_seconds`. After extract(),
    `bands` holds the complete record of every band, which BandExtractorSkip
    can replay, `selection` the selector's summary, and `last_timing` the
    phase timings for inspection benchmarking."""

    produces = VDIA

    def __init__(self, slot_cost: Callable[[int], float], min_coverage: float,
                 timeout_seconds: float | None = BAND_TIMEOUT_SECONDS):
        self.slot_cost = slot_cost
        self.min_coverage = min_coverage
        self.timeout_seconds = timeout_seconds
        self.bands: list[dict[str, Any]] = []
        self.selection: dict[str, Any] = {}
        self.last_timing: dict[str, Any] | None = None

    def extract(self, A: ResidualMatrix) -> tuple[VDIA, ResidualMatrix]:
        t0 = time.perf_counter()
        csr = A.to_csr()
        csr.sort_indices()
        t1 = time.perf_counter()
        self.selection = {}
        regions = select_bands(csr, self.slot_cost, self.min_coverage,
                               timeout_seconds=self.timeout_seconds, info=self.selection)
        t2 = time.perf_counter()
        self.bands = [region.to_dict() for region in regions]
        fmt = pack_bands_as_vdia(A, self.bands)
        t3 = time.perf_counter()
        residual = A.without(fmt)
        t4 = time.perf_counter()
        self.last_timing = {
            "to_csr_seconds": t1 - t0,
            "search_seconds": t2 - t1,
            "pack_seconds": t3 - t2,
            "residual_seconds": t4 - t3,
            "bands": [
                {
                    "rows": band.get("rows"),
                    "diag_offset": band.get("diag_offset"),
                    "total_nnz": band.get("total_nnz"),
                    "vdia_area": band.get("vdia_area"),
                    "found_at_seconds": band.get("found_at_seconds"),
                }
                for band in self.bands
            ],
        }
        return fmt, residual


class SpMVBandExtractor(BandExtractor):
    """BandExtractor for NaiveVDIASpmv: a slot costs 0.90 nonzeros at every
    width (find_vdia.spmv_slot_cost)."""

    def __init__(self, min_coverage: float = SPMV_MIN_COVERAGE,
                 timeout_seconds: float | None = BAND_TIMEOUT_SECONDS):
        super().__init__(spmv_slot_cost, min_coverage, timeout_seconds)


class SpMMBandExtractor(BandExtractor):
    """BandExtractor for NaiveVDIASpmm: the slot cost measured for the tiled
    kernel at 512 columns (find_vdia.spmm_slot_cost)."""

    def __init__(self, min_coverage: float = SPMM_MIN_COVERAGE,
                 timeout_seconds: float | None = BAND_TIMEOUT_SECONDS):
        super().__init__(spmm_slot_cost, min_coverage, timeout_seconds)


class BandExtractorSkip:
    produces = VDIA

    def __init__(self, bands: list[Any] | None = None, yaml_path: str | pathlib.Path | None = None):
        if bands is not None and yaml_path is not None:
            raise ValueError("Pass either bands or yaml_path, not both")
        self.bands = parse_yaml_bands(str(yaml_path)) if yaml_path is not None else list(bands or [])

    def extract(self, A: ResidualMatrix) -> tuple[VDIA, ResidualMatrix]:
        fmt = pack_bands_as_vdia(A, self.bands)
        return fmt, A.without(fmt)
