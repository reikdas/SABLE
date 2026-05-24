import os
import pathlib
import re
from typing import Any, List, Tuple

import yaml

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

def _generated_dense_tensors_dir() -> str:
    return os.environ.get("SABLE_DENSE_TENSOR_DIR") or os.path.join(BASE_PATH, "Generated_dense_tensors")


def _write_repeated_values(path: str, val: float, size: int, chunk_size: int = 65536) -> None:
    value = str(val)
    with open(path, "w") as f:
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


def write_dense_vector(val: float, size: int):
    filename = f"generated_vector_{size}.vector"
    dir_name = _generated_dense_tensors_dir()
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    _write_repeated_values(os.path.join(dir_name, filename), val, size)

def write_dense_matrix(val: float, m: int, n: int):
    filename = f"generated_matrix_{m}x{n}.matrix"
    dir_name = _generated_dense_tensors_dir()
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    _write_repeated_values(os.path.join(dir_name, filename), val, n * m)

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
