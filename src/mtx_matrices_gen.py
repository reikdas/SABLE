import os
import numpy
from math import ceil
from multiprocessing import Process

from src.fileio import read_data, write_mm_file

'''
This file contains functionality to convert VBR matrices to Matrix Market format.
'''

NUM_PROCS = 30

def find_nonneg(l):
    for _, ele in enumerate(l):
        if ele != -1:
            return ele
    assert(False)
    return -1

def convert_all_vbr_to_mtx():
    dir_name = "Generated_Matrix"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_names = os.listdir("Generated_Data")
    n = len(file_names)
    procs = [-1 for _ in range(n)]
    
    for i in range(ceil(n / NUM_PROCS)):
        s = i * NUM_PROCS
        e = min((i + 1) * NUM_PROCS, n)
        
        for j in range(s, e):
            filename = file_names[j]
            p = Process(target=vbr_to_mtx, args=(filename, dir_name))
            procs[j] = p
            procs[j].start()
            
        for j in range(s, e):
            procs[j].join()
            print(f'{j}: Process {j} joined')

def vbr_to_mtx(filename: str, dir_name: str = "Generated_Matrix"):
    assert(filename.endswith(".data"))
    x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_data(os.path.join("Generated_Data", filename))
    filename = filename[:-5]
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
