import os
import numpy

from src.fileio import read_vbr, write_mm_file

'''
This file contains functionality to convert VBR matrices to Matrix Market format.
'''

def find_nonneg(l):
    for _, ele in enumerate(l):
        if ele != -1:
            return ele
    assert(False)
    return -1

def convert_all_vbr_to_mtx():
    dir_name = "Generated_MMarket"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_names = os.listdir("Generated_VBR")
    for filename in file_names:
        print("Converting", filename, "to Matrix Market format")
        vbr_to_mtx(filename, dir_name)

def vbr_to_mtx(filename: str, dir_name: str = "Generated_MMarket", vbr_dir="Generated_VBR"):
    assert(filename.endswith(".vbr"))
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(vbr_dir, filename))
    filename = filename[:-len(".vbr")]
    M = numpy.zeros((rpntr[-1], cpntr[-1]))
    count = 0
    bpntrb_start = find_nonneg(bpntrb)
    for a in range(len(rpntr) - 1):
        valid_cols = bindx[bpntrb[a]-bpntrb_start:bpntre[a]-bpntrb_start]
        for b in range(len(cpntr) - 1):
            if b in valid_cols:
                count2 = 0
                for j in range(cpntr[b], cpntr[b+1]):
                    for i in range(rpntr[a], rpntr[a+1]):
                        M[i][j] = val[indx[count]+count2]
                        count2 += 1
                count += 1
    write_mm_file(os.path.join(dir_name, filename + ".mtx"), M)
    return M
