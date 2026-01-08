import os
import pathlib
import re
from os import makedirs
from os.path import exists, join
from typing import List, Tuple

import yaml
from numpy import count_nonzero

from src.vbr import VBR

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

def write_vbr_matrix(filename: str, vbr_matrix: VBR, dir_name: str):
    
    assert(type(vbr_matrix) == VBR)
    assert(type(filename) == str)
    
    val = vbr_matrix.val
    indx = vbr_matrix.indx
    bindx = vbr_matrix.bindx
    rpntr = vbr_matrix.rpntr
    cpntr = vbr_matrix.cpntr
    bpntrb = vbr_matrix.bpntrb
    bpntre = vbr_matrix.bpntre
    
    if not exists(dir_name):
        makedirs(dir_name)
    with open(join(dir_name, filename+".vbr"), "w") as f:
        f.write(f"val=[{','.join(map(str, val))}]\n")
        f.write(f"indx=[{','.join(map(str, indx))}]\n")
        f.write(f"bindx=[{','.join(map(str, bindx))}]\n")
        f.write(f"rpntr=[{','.join(map(str, rpntr))}]\n")
        f.write(f"cpntr=[{','.join(map(str, cpntr))}]\n")
        f.write(f"bpntrb=[{','.join(map(str, bpntrb))}]\n")
        f.write(f"bpntre=[{','.join(map(str, bpntre))}]\n")

def write_dense_vector(val: float, size: int):
    filename = f"generated_vector_{size}.vector"
    dir_name = os.path.join(BASE_PATH, "Generated_dense_tensors")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, filename), "w") as f:
        x = [val] * size
        f.write(f"{','.join(map(str, x))}\n")

def write_dense_matrix(val: float, m: int, n: int):
    filename = f"generated_matrix_{m}x{n}.matrix"
    dir_name = os.path.join(BASE_PATH, "Generated_dense_tensors")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, filename), "w") as f:
        x = [val] * n * m
        f.write(f"{','.join(map(str, x))}\n")

def read_vbr(filename):
    with open(filename, "r") as f:
        val = list(map(float, f.readline().split("=")[1][1:-2].split(",")))
        indx = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bindx = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        rpntr = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        cpntr = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bpntrb = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bpntre = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
    return val, indx, bindx, rpntr, cpntr, bpntrb, bpntre

def read_vbrc(filename):
    with open(filename, "r") as f:
        l_val = f.readline().split("=")[1][1:-2]
        val: list[float] = []
        if l_val != "":
            val.extend(list(map(float, l_val.split(","))))
        l_val = f.readline().split("=")[1][1:-2]
        csr_val: list[float] = []
        if l_val != "":
            csr_val.extend(list(map(float, l_val.split(","))))
        l_i = f.readline().split("=")[1][1:-2]
        l_j = f.readline().split("=")[1][1:-2]
        indptr: list[int] = []
        indices: list[int] = []
        if l_i != "":
            indptr = list(map(int, l_i.split(",")))
            indices = list(map(int, l_j.split(",")))
        indx: list[int] = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bindx: list[int] = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        rpntr: list[int] = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        cpntr: list[int] = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bpntrb: list[int] = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bpntre: list[int] = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        ublocks: list[int] = []
        l = f.readline().split("=")[1][1:-2]
        if l != "":
            ublocks = list(map(int, l.split(",")))
    return val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val

def read_vector(filename):
    with open(filename, "r") as f:
        x = list(map(float, f.readline().split("=")[1][1:-2].split(",")))
    return x

def read_matrix(filename):
    with open(filename, "r") as f:
        x = list(map(float, f.readline().split("=")[1][1:-2].split(",")))
    return x

def write_mm_file(filename, M):
    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{M.shape[0]} {M.shape[1]} {count_nonzero(M)}\n")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i][j] != 0:
                    f.write(f"{i+1} {j+1} {M[i][j]}\n")

def cleanup(*args):
    for arg in args:
        os.rmdir(arg)

def parse_yaml_blocks(yaml_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse YAML file to extract dense block coordinates.
    
    Args:
        yaml_path: Path to YAML file containing block information
    
    Returns:
        List of dense block coordinates as (row_start, row_end, col_start, col_end)
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
