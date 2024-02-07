import os
from src.mtx_matrices_gen import vbr_to_mtx
from src.codegen import vbr_spmm_codegen
import subprocess

import numpy

def cmp_file(file1, file2):
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            for line1, line2 in zip(f1, f2):
                if line1!=line2:
                    return False
    return True

def test_spmm():
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
    mult = dense.dot(numpy.ones(dense.shape))
    mult_canon = numpy.array([[ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.],
                                [ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.],
                                [11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.],
                                [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
                                [13., 13., 13., 13., 13., 13., 13., 13., 13., 13., 13.],
                                [34., 34., 34., 34., 34., 34., 34., 34., 34., 34., 34.],
                                [23., 23., 23., 23., 23., 23., 23., 23., 23., 23., 23.],
                                [20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.],
                                [ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.],
                                [40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40.],
                                [21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21.]])
    assert(numpy.array_equal(mult, mult_canon))
    vbr_spmm_codegen(filename="example", dir_name="tests", threads=1, vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c", "-march=native", "-O3"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output-canon.txt"))
