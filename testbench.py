# For benchmarking our generated code for single matrices after matrix generation

import subprocess
import sys
from argparse import ArgumentParser
from enum import Enum

from src.codegen import vbr_spmm_codegen

class Operation(Enum):
    SPMV = "spmv"
    SPMM = "spmm"

def exec_spmv(filename, threads):
    raise NotImplementedError

def exec_spmm(filename, threads):
    vbr_spmm_codegen(filename, threads=threads)
    # Only execute on Tim Rogers' machine since it has AVX-512 instructions
    subprocess.run(["/usr/bin/gcc-8", "-O3", "-mprefer-vector-width=512", "-mavx", "-funroll-all-loops", "-march=native", "-pthread", "-o", "testbench", sys.argv[1]+".c"], cwd="Generated_SpMM")
    sum = 0
    for _ in range(5):
        output = subprocess.check_output(["./testbench"], cwd="Generated_SpMM").decode("utf-8").split("\n")[0].split("=")[1]
        sum += float(output)
    print(sum/5)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--operation", type=Operation, choices=list(Operation), required=True)
    parser.add_argument("-t", "--threads", type=int, default=1)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    if (args.operation == Operation.SPMV):
        exec_spmv(args.filename, args.threads)
    elif (args.operation == Operation.SPMM):
        exec_spmm(args.filename, args.threads)
    else:
        raise Exception("Unknown operation")
