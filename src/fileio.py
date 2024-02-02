from numpy import count_nonzero
from os import makedirs
from os.path import exists, join
from src.vbr import VBR

def write_vbr_matrix(filename: str, vbr_matrix: VBR):
    
    assert(type(vbr_matrix) == VBR)
    assert(type(filename) == str)
    
    val = vbr_matrix.val
    indx = vbr_matrix.indx
    bindx = vbr_matrix.bindx
    rpntr = vbr_matrix.rpntr
    cpntr = vbr_matrix.cpntr
    bpntrb = vbr_matrix.bpntrb
    bpntre = vbr_matrix.bpntre
    
    dir_name = "Generated_VBR"
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

def write_dense_vector(filename: str, val: float, size: int):
    dir_name = "Generated_Vector"
    if not exists(dir_name):
        makedirs(dir_name)
    with open(join(dir_name, filename+".vector"), "w") as f:
        x = [val] * size
        f.write(f"x=[{','.join(map(str, x))}]\n")

def write_dense_matrix(filename: str, val: float, m: int, n: int):
    dir_name = "Generated_Matrix"
    if not exists(dir_name):
        makedirs(dir_name)
    with open(join(dir_name, filename+".matrix"), "w") as f:
        x = [val] * n * m
        f.write(f"x=[{','.join(map(str, x))}]\n")

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
