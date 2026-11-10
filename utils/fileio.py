"""Files the benchmarks, tests, and Skip extractors read and write.

Dense right-hand-side inputs: the generated C opens the right-hand side by
path when it runs, so one file per shape is enough. The writers are
idempotent: a file that already exists with the expected length and value is
left alone and its path is returned, so calling them once per kernel variant
costs one stat rather than one rewrite.

Check inputs: the benchmarks use a non-uniform right-hand side so that the
output check (bench_suitesparse.py) catches a value added into the wrong
column, which an all-ones input hides. `check_values` regenerates the same
values in Python for the reference product.

Extraction recordings: the YAML written by find-submatrices (blocks) and
scan_bands.py (bands) is the input of BlockDetectorSkip and BandExtractorSkip,
and of the benchmarks that replay a stored extraction.
"""
import os
import pathlib
import re
from typing import Any, List, Tuple

import numpy
import yaml

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")


def dense_tensors_dir() -> str:
    return os.environ.get("SABLE_DENSE_TENSOR_DIR") or os.path.join(BASE_PATH, "Generated_dense_tensors")


def dense_vector_path(size: int) -> str:
    return os.path.abspath(os.path.join(dense_tensors_dir(), f"generated_vector_{size}.vector"))


def dense_matrix_path(rows: int, cols: int) -> str:
    return os.path.abspath(os.path.join(dense_tensors_dir(), f"generated_matrix_{rows}x{cols}.matrix"))


def _expected_length(value: str, count: int) -> int:
    # "v,v,...,v\n": count values, count - 1 separators, one newline.
    return len(value) * count + max(count - 1, 0) + 1


def _already_written(path: str, value: str, size: int) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) != _expected_length(value, size):
        return False
    with open(path) as f:
        head = f.read(len(value) + 1)
    return head == value + ("," if size > 1 else "\n")


def _write_repeated_values(path: str, val: float, size: int, chunk_size: int = 65536) -> str:
    value = str(val)
    if _already_written(path, value, size):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write to a temporary name and rename, so a partial file from an
    # interrupted run never passes the length check above.
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        written = 0
        first = True
        while written < size:
            chunk_count = min(chunk_size, size - written)
            chunk = ",".join([value] * chunk_count)
            if not first:
                f.write(",")
            f.write(chunk)
            first = False
            written += chunk_count
        f.write("\n")
    os.replace(tmp_path, path)
    return path


def write_dense_vector(val: float, size: int) -> str:
    """Write the all-`val` vector of length `size` if absent; return its path."""
    return _write_repeated_values(dense_vector_path(size), val, size)


def write_dense_matrix(val: float, m: int, n: int) -> str:
    """Write the all-`val` m x n row-major matrix if absent; return its path."""
    return _write_repeated_values(dense_matrix_path(m, n), val, n * m)


def write_dense_values(name: str, values) -> str:
    """Write a dense tensor holding `values`, comma-separated as
    read_dense_input expects, to `name` in the dense tensors directory;
    return its path. Unlike the constant writers, this always rewrites.
    """
    path = os.path.abspath(os.path.join(dense_tensors_dir(), name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    chunk_size = 65536
    with open(tmp_path, "w") as f:
        for start in range(0, len(values), chunk_size):
            if start:
                f.write(",")
            f.write(",".join(f"{float(v):.17g}" for v in values[start:start + chunk_size]))
        f.write("\n")
    os.replace(tmp_path, path)
    return path


CHECK_MULTIPLIER = 40503
CHECK_MODULUS = 65521
CHECK_LEVELS = 16
# 0.5 + m/16 has an exact binary value and a short decimal form, which keeps
# the check inputs of million-column SpMM operands near the size of the
# all-ones files.
_CHECK_STRINGS = [repr(0.5 + m / CHECK_LEVELS) for m in range(CHECK_LEVELS)]


def check_values(size: int) -> numpy.ndarray:
    """Values in [0.5, 1.5): position k holds 0.5 + m/16, m = (k * 40503 mod 65521) mod 16."""
    k = numpy.arange(size, dtype=numpy.int64)
    return 0.5 + ((k * CHECK_MULTIPLIER) % CHECK_MODULUS % CHECK_LEVELS) / CHECK_LEVELS


def check_vector_path(size: int) -> str:
    return os.path.abspath(os.path.join(dense_tensors_dir(), f"check16_vector_{size}.vector"))


def check_matrix_path(rows: int, cols: int) -> str:
    return os.path.abspath(os.path.join(dense_tensors_dir(), f"check16_matrix_{rows}x{cols}.matrix"))


def _write_check_values(path: str, size: int, chunk_size: int = 1 << 20) -> str:
    # Files are only ever renamed into place complete, so one that exists is whole.
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        for start in range(0, size, chunk_size):
            k = numpy.arange(start, min(start + chunk_size, size), dtype=numpy.int64)
            levels = ((k * CHECK_MULTIPLIER) % CHECK_MODULUS % CHECK_LEVELS).tolist()
            if start:
                f.write(",")
            f.write(",".join(_CHECK_STRINGS[m] for m in levels))
        f.write("\n")
    os.replace(tmp_path, path)
    return path


def write_check_vector(size: int) -> str:
    """Write the check vector of length `size` if absent; return its path."""
    return _write_check_values(check_vector_path(size), size)


def write_check_matrix(m: int, n: int) -> str:
    """Write the m x n row-major check matrix if absent; return its path."""
    return _write_check_values(check_matrix_path(m, n), m * n)


# ---------------------------------------------------------------------------
# Extraction recordings
# ---------------------------------------------------------------------------


def parse_yaml_blocks(yaml_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse YAML file to extract block coordinates.
    
    Args:
        yaml_path: Path to YAML file containing block information
    
    Returns:
        List of block coordinates as (row_start, row_end, col_start, col_end)
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    blocks = []
    for block in data.get('blocks', []):
        # rows/cols are [start, end) with the end exclusive, as the partitioner writes them
        rows_data = block['rows']
        cols_data = block['cols']
        
        # Handle string format "[30, 2107]" (quoted in YAML)
        if isinstance(rows_data, str):
            # Match both [start, end] and [start, end) for backward compatibility
            rows_match = re.match(r'\[(\d+),\s*(\d+)[\])]', rows_data)
            if rows_match:
                row_start = int(rows_match.group(1))
                row_end = int(rows_match.group(2))
            else:
                continue
        elif isinstance(rows_data, list) and len(rows_data) == 2:
            row_start = int(rows_data[0])
            row_end = int(rows_data[1])
        else:
            continue
        
        if isinstance(cols_data, str):
            # Match both [start, end] and [start, end) for backward compatibility
            cols_match = re.match(r'\[(\d+),\s*(\d+)[\])]', cols_data)
            if cols_match:
                col_start = int(cols_match.group(1))
                col_end = int(cols_match.group(2))
            else:
                continue
        elif isinstance(cols_data, list) and len(cols_data) == 2:
            col_start = int(cols_data[0])
            col_end = int(cols_data[1])
        else:
            continue
        
        blocks.append((row_start, row_end, col_start, col_end))
    
    return blocks


def parse_yaml_bands(yaml_path: str) -> List[dict[str, Any]]:
    """Parse a VDIA band result YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("bands") or [])
