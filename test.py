import os
import subprocess

import numpy
import scipy

from src.baseline import *
from src.codegen import *
from utils.convert_real_to_vbr import (convert_sparse_to_vbr,
                                       convert_vbr_to_compressed)
from utils.fileio import write_dense_matrix, write_dense_vector
from utils.mtx_matrices_gen import vbr_to_mtx


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
    write_dense_vector(1.0, 11)
    write_dense_matrix(1.0, 11, 512)

def test_partition():
    dense = numpy.array([[ 4.,  2.,  0.,  0.,  0.,  1.,  0.,  0.,  0., 0.,  1.],
                                [ 1.,  5.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0., -1.],
                                [ 0.,  0.,  6.,  1.,  2.,  2.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  2.,  7.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0., -1.,  2.,  9.,  3.,  0.,  0.,  0.,  0.,  0.],
                                [ 2.,  1.,  3.,  4.,  5., 10.,  4.,  3.,  2.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  4., 13.,  4.,  2.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  3.,  3., 11.,  3.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  7.,  0.,  0.],
                                [ 8.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  3.],
                                [-2.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 12.]])
    sparse = scipy.sparse.csc_matrix(dense)
    rpntr = [0, 2, 5, 6, 9, 11]
    cpntr = [0, 2, 5, 6, 9, 11]
    val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr(sparse, rpntr, cpntr, "example2", "tests")
    assert(val ==  [4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 0, 0.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 0.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 0, 0, 3.0, 12.0])
    assert(indx==[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51])
    assert(bindx==[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4])
    assert(bpntrb==[0, 3, 5, 9, 11])
    assert(bpntre==[3, 5, 9, 11, 13])

def run_spmv(threads):
    test_setup_file()
    vbr_spmv_codegen(filename="example", density=0, dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c", "-march=native", "-O3", "-lpthread"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_spmv_unroll(threads):
    test_setup_file()
    vbr_spmv_codegen(filename="example", density=80, dir_name="tests", threads=threads, vbr_dir="tests")
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

def run_spmm_libxsmm():
    filename = "example"
    dir_name = "tests"
    test_setup_file()
    vbr_path = os.path.join(dir_name, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    gen_spmm_libxsmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename, dir_name)
    subprocess.check_call([
    "gcc",
    "-o", "example",
    "example.c",
    "-march=native",
    "-O3",
    "-I", "/local/scratch/a/das160/libxsmm/include",
    "-L", "/local/scratch/a/das160/libxsmm/lib",
    "-lblas",
    "-lm"
], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmm_canon.txt"))

def run_spmm_cblas():
    filename = "example"
    dir_name = "tests"
    test_setup_file()
    vbr_path = os.path.join(dir_name, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    gen_spmm_cblas(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename, dir_name)
    subprocess.check_call([
    "gcc",
    "-o", "example",
    "example.c",
    "-march=native",
    "-O3",
    "-lblas",
], cwd="tests")
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

# def test_spmm_cuda_cublas():
#     test_setup_file()
#     vbr_spmm_cuda_codegen_cublas(filename="example", dir_name="tests", vbr_dir="tests", density=0)
#     subprocess.check_call(["nvcc", "-o", "example", "example.cu", "-O3", "-lcublas"], cwd="tests")
#     output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
#     with open(os.path.join("tests", "output.txt"), "w") as f:
#         f.write("\n".join(output))
#     assert(cmp_file("tests/output.txt", "tests/output_spmm_canon.txt"))

def run_nonzeros_spmv():
    test_setup_file()
    only_nonzeros_spmv(filename="example", dir_name="tests", vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c", "-march=native", "-O3"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_nonzeros_spmm():
    test_setup_file()
    only_nonzeros_spmm(filename="example", dir_name="tests", vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c", "-march=native", "-O3", "-lpthread"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmm_canon.txt"))

def run_compression():
    dense = numpy.array([[ 4.,  2.,  0.,  0.,  0.,  1.,  0.,  0.,  0., 0.,  1.],
                        [ 1.,  5.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0., -1.],
                        [ 0.,  0.,  6.,  1.,  2.,  2.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  2.,  7.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0., -1.,  2.,  9.,  3.,  0.,  0.,  0.,  0.,  0.],
                        [ 2.,  1.,  3.,  4.,  5., 10.,  4.,  3.,  2.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  4., 13.,  4.,  2.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  3.,  3., 11.,  3.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  7.,  0.,  0.],
                        [ 8.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  3.],
                        [-2.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 12.]])
    sparse = scipy.sparse.csc_matrix(dense)
    rpntr = [0, 2, 5, 6, 9, 11]
    cpntr = [0, 2, 5, 6, 9, 11]
    val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr(sparse, rpntr, cpntr, "example2", "tests")
    val2, indx2, bindx, bpntrb, bpntre, ublocks, coo_i, coo_j = convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, 80, "example2", "tests")
    assert(val2== [4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 3.0, 12.0])
    assert(indx2==[0.0, 4.0, 6.0, 8.0, 17.0, 19.0, 21.0, 24.0, 25.0, 28.0, 30.0, 39.0, 43.0, 45.0])
    assert(bindx==[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4])
    assert(bpntrb==[0, 3, 5, 9, 11])
    assert(bpntre==[3, 5, 9, 11, 13])
    assert(ublocks==[2, 4, 9, 12])
    assert(coo_i==[0, 1, 2, 4, 6, 7, 9, 10])
    assert(coo_j==[10, 10, 5, 5, 5, 5, 10, 10])

def run_compression_codegen():
    rpntr = [0, 2, 5, 6, 9, 11]
    cpntr = [0, 2, 5, 6, 9, 11]
    val2= [4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 3.0, 12.0]
    indx2=[0, 4, 6, 8, 17, 19, 21, 24, 25, 28, 30, 39, 43, 45]
    bindx=[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]
    bpntrb=[0, 3, 5, 9, 11]
    bpntre=[3, 5, 9, 11, 13]
    ublocks=[2, 4, 9, 12]
    coo_i = [0, 1, 2, 4, 6, 7, 9, 10]
    coo_j = [10, 10, 5, 5, 5, 5, 10, 10]
    gen_single_threaded_spmv_compressed(val2, indx2, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, coo_i, coo_j, "tests", "example", "tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c", "-march=native", "-O3", "-lpthread"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon_sparse.txt"))

def run_spmv_splitter(threads):
    test_setup_file()
    vbr_spmv_codegen(filename="example", density=0, dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["./../split_compile.sh", "example.c", "2"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests/split-and-binaries/example/").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def test_spmv():
    run_spmv(1)
    run_spmv(2)
    run_spmv(4)
    run_spmv(8)
    run_spmv(16)

def test_spmv_unroll():
    run_spmv_unroll(1)
    run_spmv_unroll(2)
    run_spmv_unroll(4)
    run_spmv_unroll(8)
    run_spmv_unroll(16)

def test_spmv_splitter():
    run_spmv_splitter(1)

def test_spmm():
    run_spmm(1)
    run_spmm(2)
    run_spmm(4)
    run_spmm(8)
    run_spmm(16)
#     run_spmm_libxsmm()
#     run_spmm_cblas()

# def test_spmv_cuda():
#     run_spmv_cuda()

# def test_spmm_cuda():
#     run_spmm_cuda()
    
def test_baselines():
    run_nonzeros_spmv()
    run_nonzeros_spmm()

def test_vbr_compression():
    run_compression()
    run_compression_codegen()
