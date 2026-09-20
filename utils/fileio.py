"""Files the benchmarks, tests, and Skip extractors read and write.

Dense right-hand-side inputs: the generated C opens the right-hand side by
path when it runs, so one file per shape is enough. The writers are
idempotent: a file that already exists with the expected length and value is
left alone and its path is returned, so calling them once per kernel variant
costs one stat rather than one rewrite.

Extraction recordings: the YAML written by find-submatrices (blocks) and
find_vdia.py (bands) is the input of BlockDetectorSkip and BandExtractorSkip,
and of the benchmarks that replay a stored extraction.
"""
import os
import pathlib
import re
from typing import Any, List, Tuple

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
