import subprocess
import sys
from src.codegen import vbr_spmm_codegen

if __name__ == "__main__":
    vbr_spmm_codegen(sys.argv[1], threads=1)
    subprocess.run(["/usr/bin/gcc-8", "-O3", "-mprefer-vector-width=512", "-mavx", "-march=native", "-o", "testbench", sys.argv[1]+".c"], cwd="Generated_SpMM")
    sum = 0
    for i in range(5):
        output = subprocess.check_output(["./testbench"], cwd="Generated_SpMM").decode("utf-8").split("\n")[0].split("=")[1]
        sum += float(output)
    print(sum/5)
