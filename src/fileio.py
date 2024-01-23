from os import makedirs
from os.path import exists, join
from src.vbr import VBR

def write_vbr_matrix(filename: str, vbr_matrix: VBR):
    
    assert(type(vbr_matrix) == VBR)
    assert(type(filename) == str)
    
    x = vbr_matrix.x
    val = vbr_matrix.val
    indx = vbr_matrix.indx
    bindx = vbr_matrix.bindx
    rpntr = vbr_matrix.rpntr
    cpntr = vbr_matrix.cpntr
    bpntrb = vbr_matrix.bpntrb
    bpntre = vbr_matrix.bpntre
    
    dir_name = "Generated_Data"
    if not exists(dir_name):
        makedirs(dir_name)
    with open(join(dir_name, filename+".data"), "w") as f:
        f.write(f"x=[{','.join(map(str, x))}]\n")
        f.write(f"val=[{','.join(map(str, val))}]\n")
        f.write(f"indx=[{','.join(map(str, indx))}]\n")
        f.write(f"bindx=[{','.join(map(str, bindx))}]\n")
        f.write(f"rpntr=[{','.join(map(str, rpntr))}]\n")
        f.write(f"cpntr=[{','.join(map(str, cpntr))}]\n")
        f.write(f"bpntrb=[{','.join(map(str, bpntrb))}]\n")
        f.write(f"bpntre=[{','.join(map(str, bpntre))}]\n")

def read_data(filename):
    with open(filename, "r") as f:
        x = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        val = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        indx = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bindx = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        rpntr = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        cpntr = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bpntrb = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bpntre = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
    return x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre