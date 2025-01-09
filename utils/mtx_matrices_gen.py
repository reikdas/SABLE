import os

import numpy

from utils.fileio import read_vbr, write_mm_file

'''
This file contains functionality to convert VBR matrices to Matrix Market format.
'''

def find_nonneg(l):
    for _, ele in enumerate(l):
        if ele != -1:
            return ele
    assert(False)

def convert_all_vbr_to_mtx(dense_blocks_only: bool):
    if (dense_blocks_only):
        input_dir_name = "Generated_VBR"
        output_dir_name = "Generated_MMarket"
    else:
        input_dir_name = "Generated_VBR_Sparse"
        output_dir_name = "Generated_MMarket_Sparse"
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    for filename in os.listdir(input_dir_name):
        vbr_to_mtx(filename, output_dir_name, input_dir_name)

def vbr_to_mtx(filename: str, dir_name, vbr_dir):
    assert(filename.endswith(".vbr"))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
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
