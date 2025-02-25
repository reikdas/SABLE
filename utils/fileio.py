import os
import pathlib
from os import makedirs
from os.path import exists, join

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
