import os
import subprocess

import numpy

from ops import vbr_spmm_codegen, vbr_spmv_codegen
from src.codegen import vbr_spmm_cuda_codegen, vbr_spmv_cuda_codegen
from src.mtx_matrices_gen import vbr_to_mtx


def cmp_file(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip()
            line2 = line2.strip()
            # Attempt to compare as floats if the lines contain numeric data
            try:
                # This will succeed if both lines are numeric
                if float(line1) != float(line2):
                    return False
            except ValueError:
                # If they aren't numeric, compare them as strings
                if line1 != line2:
                    return False

    return True

def test_setup_file():
    filename = "example.vbr"
    dense = vbr_to_mtx(filename, dir_name="tests", vbr_dir="tests")
    dense_canon = numpy.array([[ 4.,  2.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.,  1.],
                                [ 1.,  5.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0., -1.],
                                [ 0.,  0.,  6.,  1.,  2.,  2.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  2.,  7.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0., -1.,  2.,  9.,  3.,  0.,  0.,  0.,  0.,  0.],
                                [ 2.,  1.,  3.,  4.,  5., 10.,  4.,  3.,  2.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  4., 13.,  4.,  2.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  3.,  3., 11.,  3.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  7.,  0.,  0.],
                                [ 8.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 25.,  3.],
                                [-2.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  8., 12.]])
    assert(numpy.array_equal(dense, dense_canon))
    assert(cmp_file("tests/example.mtx", "tests/example-canon.mtx"))

def run_spmv(threads):
    test_setup_file()
    vbr_spmv_codegen(filename="example", density=0, dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c", "-march=native", "-O3", "-lpthread"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_spmv_cuda():
    test_setup_file()
    vbr_spmv_cuda_codegen(filename="example", dir_name="tests", vbr_dir="tests", density=0)
    subprocess.check_call(["nvcc", "-o", "example", "example.cu", "-O3"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_spmm(threads):
    test_setup_file()
    vbr_spmm_codegen(filename="example", density=0, dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c", "-march=native", "-O3", "-lpthread"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmm_canon.txt"))

def run_spmm_cuda():
    test_setup_file()
    vbr_spmm_cuda_codegen(filename="example", dir_name="tests", vbr_dir="tests", density=0)
    subprocess.check_call(["nvcc", "-o", "example", "example.cu", "-O3"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmm_canon.txt"))

def test_spmv():
    run_spmv(1)
    run_spmv(2)
    run_spmv(4)
    run_spmv(8)
    run_spmv(16)
    run_spmv_cuda()

def test_spmm():
    run_spmm(1)
    run_spmm(2)
    run_spmm(4)
    run_spmm(8)
    run_spmm(16)
    run_spmm_cuda()
