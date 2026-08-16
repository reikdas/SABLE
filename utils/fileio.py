import os
import pathlib
import re
from typing import Any, List, Tuple

import yaml

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

def _generated_dense_tensors_dir() -> str:
    dir_name = os.environ.get("SABLE_DENSE_TENSOR_DIR") or os.path.join(BASE_PATH, "Generated_dense_tensors")
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def write_dense_values(name: str, values) -> str:
    """Write a flat dense tensor file with the given per-element values,
    comma-separated as read_dense_input expects.  The file lands in the
    generated tensors directory under `name`; returns its absolute path.
    """
    path = os.path.abspath(os.path.join(_generated_dense_tensors_dir(), name))
    chunk_size = 65536
    with open(path, "w") as f:
        for start in range(0, len(values), chunk_size):
            if start:
                f.write(",")
            f.write(",".join(f"{float(v):.17g}" for v in values[start:start + chunk_size]))
        f.write("\n")
    return path


def write_dense_vector(val: float, size: int) -> str:
    """Fill a size-element vector with one repeated constant; returns its path."""
    return write_dense_values(f"generated_vector_{size}.vector", [val] * size)


def write_dense_matrix(val: float, m: int, n: int) -> str:
    """Fill an m x n matrix with one repeated constant; returns its path."""
    return write_dense_values(f"generated_matrix_{m}x{n}.matrix", [val] * (m * n))

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
        # Parse rows: [start, end] format (end is exclusive in code, but written as inclusive in YAML)
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
