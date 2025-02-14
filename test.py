import os
import subprocess

import numpy
import scipy

from src.baseline import *
from src.codegen import *
from src.consts import CFLAGS as CFLAGS
from src.autopartition import cut_indices2, similarity2
from utils.convert_real_to_vbr import (convert_sparse_to_vbr,
                                       convert_sparse_to_vbr_np,
                                       convert_vbr_to_compressed)
from utils.fileio import write_dense_matrix, write_dense_vector
from utils.mtx_matrices_gen import vbr_to_mtx
from utils.utils import extract_mul_nums
import cProfile
import timeit

import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/tmp/out.log', level = logging.INFO)

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

def test_read_vbr():
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(BASE_PATH, "tests", "example.vbr"))
    assert(val==[4.0,1.0,2.0,5.0,1.0,2.0,-1.0,0.0,1.0,-1.0,6.0,2.0,-1.0,1.0,7.0,2.0,2.0,1.0,9.0,2.0,0.0,3.0,2.0,1.0,3.0,4.0,5.0,10.0,4.0,3.0,2.0,4.0,3.0,0.0,13.0,3.0,2.0,4.0,11.0,0.0,2.0,3.0,7.0,8.0,-2.0,4.0,3.0,25.0,8.0,3.0,12.0])
    assert(indx==[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51])
    assert(bindx==[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4])
    assert(rpntr==[0, 2, 5, 6, 9, 11])
    assert(cpntr==[0, 2, 5, 6, 9, 11])
    assert(bpntrb==[0, 3, 5, 9, 11])
    assert(bpntre==[3, 5, 9, 11, 13])

def test_compression_full_dense():
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(BASE_PATH, "tests", "example.vbr"))
    val, indx, bindx, bpntrb, bpntre, ublocks, coo_i, coo_j, coo_val = convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, 0, "example", "tests")
    assert(val==[4.0,1.0,2.0,5.0,1.0,2.0,-1.0,0.0,1.0,-1.0,6.0,2.0,-1.0,1.0,7.0,2.0,2.0,1.0,9.0,2.0,0.0,3.0,2.0,1.0,3.0,4.0,5.0,10.0,4.0,3.0,2.0,4.0,3.0,0.0,13.0,3.0,2.0,4.0,11.0,0.0,2.0,3.0,7.0,8.0,-2.0,4.0,3.0,25.0,8.0,3.0,12.0])
    assert(indx==[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51])
    assert(bindx==[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4])
    assert(rpntr==[0, 2, 5, 6, 9, 11])
    assert(cpntr==[0, 2, 5, 6, 9, 11])
    assert(bpntrb==[0, 3, 5, 9, 11])
    assert(bpntre==[3, 5, 9, 11, 13])
    assert(ublocks==[])
    assert(coo_i==[])
    assert(coo_j==[])
    assert(coo_val==[])


def test_compression():
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
    val2, indx2, bindx, bpntrb, bpntre, ublocks, coo_i, coo_j, coo_val = convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, 80, "example2", "tests")
    assert(val2== [4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0])
    assert(indx2==[0, 4, 6, 15, 17, 20, 21, 24, 33, 37])
    assert(bindx==[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4])
    assert(bpntrb==[0, 3, 5, 9, 11])
    assert(bpntre==[3, 5, 9, 11, 13])
    assert(ublocks==[2, 4, 9, 12])
    assert(coo_i==[0, 1, 2, 4, 6, 7, 9, 10])
    assert(coo_j==[10, 10, 5, 5, 5, 5, 10, 10])
    assert(coo_val==[1.0, -1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 12.0])

def test_read_compression():
    test_compression()
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, coo_i, coo_j, coo_val = read_vbrc(os.path.join(BASE_PATH, "tests", "example2.vbrc"))
    assert(val== [4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0])
    assert(indx==[0, 4, 6, 15, 17, 20, 21, 24, 33, 37])
    assert(bindx==[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4])
    assert(rpntr==[0, 2, 5, 6, 9, 11])
    assert(cpntr==[0, 2, 5, 6, 9, 11])
    assert(bpntrb==[0, 3, 5, 9, 11])
    assert(bpntre==[3, 5, 9, 11, 13])
    assert(ublocks==[2, 4, 9, 12])
    assert(coo_i==[0, 1, 2, 4, 6, 7, 9, 10])
    assert(coo_j==[10, 10, 5, 5, 5, 5, 10, 10])
    assert(coo_val==[1.0, -1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 12.0])

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
    vbr_spmv_codegen(filename="example", dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c"] + CFLAGS, cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_spmv_multi_out(threads):
    test_setup_file()
    vbr_spmv_codegen(filename="example", dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c", "-march=native", "-O3", "-lpthread"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[0]
    output = extract_mul_nums(output)
    assert(len(output)==5)

def run_spmv_unroll(threads):
    test_compression()
    vbr_spmv_codegen("example2", "tests", "tests", threads)
    subprocess.check_call(["gcc", "-o", "example2", "example2.c", "-march=native", "-O3", "-lpthread"], cwd="tests")
    output = subprocess.check_output(["./example2"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon_sparse.txt"))

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
    subprocess.check_call(["gcc", "-o", "example", "example.c"] + CFLAGS, cwd="tests")
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
    "-I", "/local/scratch/a/das160/libxsmm/include",
    "-L", "/local/scratch/a/das160/libxsmm/lib",
    "-lblas",
    "-lm"
] + CFLAGS, cwd="tests")
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
    "-lblas",
]+CFLAGS, cwd="tests")
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
    subprocess.check_output(["g++", "-o", "csr-spmv", "csr-spmv.cpp"] + CFLAGS, cwd=os.path.join(BASE_PATH, "src"))
    baseline_output = subprocess.run(["./csr-spmv", os.path.join(BASE_PATH, "tests", "example-canon.mtx"), str(1), str(1), os.path.join(BASE_PATH, "Generated_dense_tensors", "generated_vector_11.vector")], capture_output=True, cwd=os.path.join(BASE_PATH, "src"))
    output = baseline_output.stdout.decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_nonzeros_spmm():
    test_setup_file()
    only_nonzeros_spmm(filename="example", dir_name="tests", vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c"] + CFLAGS, cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmm_canon.txt"))

def run_spmv_splitter(threads):
    test_setup_file()
    vbr_spmv_codegen(filename="example", dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["./../split_compile.sh", "example.c", "2"], cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests/split-and-binaries/example/").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_spmv_unroll_splitter(threads):
    test_compression()
    vbr_spmv_codegen(filename="example2", dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["./../split_compile.sh", "example2.c", "2"], cwd="tests")
    output = subprocess.check_output(["./example2"], cwd="tests/split-and-binaries/example2/").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon_sparse.txt"))

def test_spmv():
    run_spmv(1)
#     run_spmv(2)
#     run_spmv(4)
#     run_spmv(8)
#     run_spmv(16)

def test_spmv_multi_out():
    run_spmv_multi_out(1)
#     run_spmv_multi_out(2)
#     run_spmv_multi_out(4)
#     run_spmv_multi_out(8)
#     run_spmv_multi_out(16)

def test_spmv_unroll():
    run_spmv_unroll(1)
#     run_spmv_unroll(2)
#     run_spmv_unroll(4)
#     run_spmv_unroll(8)
#     run_spmv_unroll(16)

def test_spmv_splitter():
    run_spmv_splitter(1)
    # run_spmv_splitter(2)
    # run_spmv_splitter(4)
    # run_spmv_splitter(8)
    # run_spmv_splitter(16)

def test_spmv_unroll_splitter():
    run_spmv_unroll_splitter(1)
    # run_spmv_unroll_splitter(2)
    # run_spmv_unroll_splitter(4)
    # run_spmv_unroll_splitter(8)
    # run_spmv_unroll_splitter(16)

# def test_spmm():
#     run_spmm(1)
#     run_spmm(2)
#     run_spmm(4)
#     run_spmm(8)
#     run_spmm(16)
#     run_spmm_libxsmm()
#     run_spmm_cblas()

# def test_spmv_cuda():
#     run_spmv_cuda()

# def test_spmm_cuda():
#     run_spmm_cuda()

def test_baselines():
    run_nonzeros_spmv()
    run_nonzeros_spmm()

def test_vbr_conversion():
    A = scipy.sparse.csc_matrix([
        [1.0,2.0,0.0,0.0],
        [3.0,4.0,0.0,0.0],
        [0.0,0.0,5.0,6.0],
        [0.0,0.0,7.0,8.0]]
    )

    cpntr = [0,2,4]
    rpntr = [0,1,2,4]

    val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr_np(A, rpntr, cpntr, "dummy", "tests")

    assert(np.array_equal(val, [1,2,3,4,5,7,6,8]))
    assert(np.array_equal(indx, [0,2,4,8]))
    assert(np.array_equal(bindx, [0,0,1]))
    assert(np.array_equal(bpntrb, [0,1,2]))
    assert(np.array_equal(bpntre, [1,2,3]))

def test_partition_vals_real():
    logger.info("running test to check vbr conversion") 

    # read matrix from mm-market format
    mtx_path = os.path.join(BASE_PATH, "tests", "Franz8.mtx")
    mtx = scipy.io.mmread(mtx_path)

    # convert to scipy csc
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    A_nnz = A.nnz

    # get indices of VBR partitions
    cpntr, rpntr = cut_indices2(A, 0.2, similarity2)
    logger.info("completed sub-block generation")
    val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr(A, rpntr, cpntr, "Franz8", "tests")
    val2, indx2, bindx2, bpntrb2, bpntre2 = convert_sparse_to_vbr_np(A, rpntr, cpntr, "Franz8_np", "tests")

    # check nnz
    val_nnz = len([x for x in val if x != 0])
    val2_nnz = len([x for x in val2 if x != 0])
    assert(val_nnz == val2_nnz)
    assert(A_nnz == val2_nnz)

    assert(np.array_equal(val, val2))
    assert(np.array_equal(indx, indx2))
    assert(np.array_equal(bindx, bindx2))
    assert(np.array_equal(bpntrb, bpntrb2))
    assert(np.array_equal(bpntre, bpntre2))
