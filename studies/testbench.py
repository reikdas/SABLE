# For benchmarking our generated code for single matrices after matrix generation

import subprocess
import sys
from argparse import ArgumentParser
from enum import Enum

from src.codegen import *


class Operation(Enum):
    SPMV = "spmv"
    SPMM = "spmm"

def exec_spmv(filename, threads):
    vbr_spmv_codegen(filename, threads=threads, dir_name="testing_cuda", vbr_dir="tests", dense_blocks_only=True)
    #subprocess.run(["gcc", "-O3", "-mprefer-vector-width=512", "-mavx", "-funroll-all-loops", "-march=native", "-pthread", "-o", "testbench", sys.argv[1]+".c"], cwd="Generated_SpMV")

def exec_spmv_cuda(filename):
    vbr_spmv_cuda_codegen(filename, vbr_dir="tests", dir_name="testing_cuda")

def exec_spmm(filename, threads, density_threshold):
    vbr_spmm_codegen(filename, threads=threads, density=density_threshold, dir_name="Generated_SpMM", vbr_dir="Generated_VBR")
    subprocess.run(["gcc", "-O3", "-mprefer-vector-width=512", "-mavx", "-funroll-all-loops", "-march=native", "-pthread", "-o", "testbench", sys.argv[1]+".c"], cwd="Generated_SpMM")
    sum = 0
    for _ in range(5):
        output = subprocess.check_output(["./testbench"], cwd="Generated_SpMM").decode("utf-8").split("\n")[0].split("=")[1]
        sum += float(output)
    print(sum/5)

def exec_spmm_cuda(filename):
    vbr_spmm_cuda_codegen(filename)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--cuda", action='store_true')
    parser.add_argument("-o", "--operation", type=Operation, choices=list(Operation), required=True)
    parser.add_argument("-t", "--threads", type=int, default=1)
    parser.add_argument("-d", "--density_threshold", type=int, default=0)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    if (args.cuda):
        if (args.operation == Operation.SPMV):
            exec_spmv_cuda(args.filename)
        elif (args.operation == Operation.SPMM):
            exec_spmv_cuda(args.filename)
        else:
            raise Exception("Unknown operation")
    else:
        if (args.operation == Operation.SPMV):
            exec_spmv(args.filename, args.threads)
        elif (args.operation == Operation.SPMM):
            exec_spmm(args.filename, args.threads, args.density_threshold)
        else:
            raise Exception("Unknown operation")
