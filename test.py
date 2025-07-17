import os
import subprocess
from typing import List, Union

import numpy
import pytest
import scipy

from src.autopartition import cut_indices2, similarity2
from src.codegen import *
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from utils.convert_real_to_vbr import (convert_sparse_to_vbr,
                                       convert_sparse_to_vbrc,
                                       vbrc_matrix_gen)
from utils.fileio import read_vbr, read_vbrc, write_dense_matrix, write_dense_vector
from utils.mtx_matrices_gen import vbr_to_mtx
from utils.utils import extract_mul_nums

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
    assert(numpy.array_equal(val,[4.0,1.0,2.0,5.0,1.0,2.0,-1.0,0.0,1.0,-1.0,6.0,2.0,-1.0,1.0,7.0,2.0,2.0,1.0,9.0,2.0,0.0,3.0,2.0,1.0,3.0,4.0,5.0,10.0,4.0,3.0,2.0,4.0,3.0,0.0,13.0,3.0,2.0,4.0,11.0,0.0,2.0,3.0,7.0,8.0,-2.0,4.0,3.0,25.0,8.0,3.0,12.0]))
    assert(numpy.array_equal(indx,[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51]))
    assert(numpy.array_equal(bindx,[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]))
    assert(numpy.array_equal(rpntr,[0, 2, 5, 6, 9, 11]))
    assert(numpy.array_equal(cpntr,[0, 2, 5, 6, 9, 11]))
    assert(numpy.array_equal(bpntrb,[0, 3, 5, 9, 11]))
    assert(numpy.array_equal(bpntre,[3, 5, 9, 11, 13]))


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
    val2, indx2, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_sparse_to_vbrc(sparse, rpntr, cpntr, "example2", "tests", 80)
    assert(numpy.array_equal(val2,[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0]))
    assert(numpy.array_equal(indx2,[0, 4, 6, 15, 17, 20, 21, 24, 33, 37]))
    assert(numpy.array_equal(bindx,[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]))
    assert(numpy.array_equal(bpntrb,[0, 3, 5, 9, 11]))
    assert(numpy.array_equal(bpntre,[3, 5, 9, 11, 13]))
    assert(numpy.array_equal(ublocks,[2, 4, 9, 12]))
    assert(numpy.array_equal(indptr, [0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8]))
    assert(numpy.array_equal(indices,[10, 10, 5, 5, 5, 5, 10, 10]))
    assert(numpy.array_equal(csr_val,[1.0, -1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 12.0]))

def test_read_compression():
    test_compression()
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(os.path.join(BASE_PATH, "tests", "example2.vbrc"))
    assert(numpy.array_equal(val,[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0]))
    assert(numpy.array_equal(indx,[0, 4, 6, 15, 17, 20, 21, 24, 33, 37]))
    assert(numpy.array_equal(bindx,[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]))
    assert(numpy.array_equal(rpntr,[0, 2, 5, 6, 9, 11]))
    assert(numpy.array_equal(cpntr,[0, 2, 5, 6, 9, 11]))
    assert(numpy.array_equal(bpntrb,[0, 3, 5, 9, 11]))
    assert(numpy.array_equal(bpntre,[3, 5, 9, 11, 13]))
    assert(numpy.array_equal(ublocks,[2, 4, 9, 12]))
    assert(numpy.array_equal(indptr, [0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8]))
    assert(numpy.array_equal(indices,[10, 10, 5, 5, 5, 5, 10, 10]))
    assert(numpy.array_equal(csr_val,[1.0, -1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 12.0]))

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
    assert(numpy.array_equal(val,[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 0, 0.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 0.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 0, 0, 3.0, 12.0]))
    assert(numpy.array_equal(indx,[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51]))
    assert(numpy.array_equal(bindx,[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]))
    assert(numpy.array_equal(bpntrb,[0, 3, 5, 9, 11]))
    assert(numpy.array_equal(bpntre,[3, 5, 9, 11, 13]))
    # Since density=0, all blocks should be dense, so ublocks, indptr, indices, csr_val should be empty
    assert(len(ublocks) == 0)
    assert(len(indptr) == 0)
    assert(len(indices) == 0)
    assert(len(csr_val) == 0)

def run_spmv(threads):
    test_setup_file()
    vbr_spmv_codegen(filename="example", dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c"] + CFLAGS + MKL_FLAGS, cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_spmv_py():
    test_setup_file()
    vbr_spmv_codegen_python(filename="example", dir_name="tests", vbr_dir="tests")
    output = subprocess.check_output(["python3", "example.py"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_spmv_multi_out(threads):
    test_setup_file()
    vbr_spmv_codegen(filename="example", dir_name="tests", threads=threads, vbr_dir="tests")
    subprocess.check_call(["gcc", "-o", "example", "example.c"] + CFLAGS, cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[0]
    output = extract_mul_nums(output)
    assert(len(output)==5)

def run_spmv_unroll(threads):
    test_compression()
    vbr_spmv_codegen("example2", "tests", "tests", threads)
    subprocess.check_call(["gcc", "-o", "example2", "example2.c"] + CFLAGS+MKL_FLAGS, cwd="tests")
    output = subprocess.check_output(["./example2"], cwd="tests").decode("utf-8").split("\n")
    if "warning" in output[0].lower():
        output = output[2:]
    else:
        output = output[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon_sparse.txt"))

def run_spmv_unroll_py():
    test_compression()
    vbr_spmv_codegen_python(filename="example2", dir_name="tests", vbr_dir="tests")
    output = subprocess.check_output(["python3", "example2.py"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon_sparse.txt"))

def test_spmv():
    run_spmv(1)
    run_spmv(2)
    run_spmv(4)
    run_spmv(8)
    run_spmv(16)

def test_spmv_py():
    run_spmv_py()

def test_spmv_multi_out():
    run_spmv_multi_out(1)
    run_spmv_multi_out(2)
    run_spmv_multi_out(4)
    run_spmv_multi_out(8)
    run_spmv_multi_out(16)

def test_spmv_unroll():
    run_spmv_unroll(1)
    run_spmv_unroll(2)
    run_spmv_unroll(4)
    run_spmv_unroll(8)
    run_spmv_unroll(16)

def test_spmv_unroll_py():
    run_spmv_unroll_py()

@pytest.mark.skip(reason="Git cannot store Franz8_canon.vbr")
def test_partition_vals_real():
    # read matrix from mm-market format
    mtx_path = os.path.join(BASE_PATH, "tests", "Franz8.mtx")
    mtx = scipy.io.mmread(mtx_path)

    # convert to scipy csc
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    A_nnz = A.nnz

    # get indices of VBR partitions
    cpntr, rpntr = cut_indices2(A, 0.2, similarity2)
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(BASE_PATH, "tests", "Franz8_canon.vbr"))
    val2, indx2, bindx2, bpntrb2, bpntre2 = convert_sparse_to_vbr(A, rpntr, cpntr, "Franz8", "tests")

    # check nnz
    val_nnz = len([x for x in val if x != 0])
    val2_nnz = len([x for x in val2 if x != 0])
    assert(val_nnz == val2_nnz)
    assert(A_nnz == val2_nnz)

    assert(numpy.array_equal(val, val2))
    assert(numpy.array_equal(indx, indx2))
    assert(numpy.array_equal(bindx, bindx2))
    assert(numpy.array_equal(bpntrb, bpntrb2))
    assert(numpy.array_equal(bpntre, bpntre2))

def test_vbrc_matrix_gen():
    """Test vbrc_matrix_gen function by generating a small matrix and verifying it can be read back."""
    # Generate a small 6x6 matrix with 2x2 blocks
    # 2 row splits, 2 col splits = 4 blocks total
    # 2 dense blocks, 50% zeros in dense blocks, 1 sparse block
    filename = vbrc_matrix_gen(
        m=6, n=6, 
        partitioning="uniform", 
        row_split=2, col_split=2, 
        num_dense=2, perc_dense_zeros=50, 
        num_sparse=1, 
        dense_blocks_only=False, 
        vbr_dir="tests", 
        density=80  # Use density threshold instead of ML model
    )
    
    # Read back the generated file
    vbrc_path = os.path.join("tests", f"{filename}.vbrc")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbrc_path)
    
    # Verify basic structure
    assert len(rpntr) == 3  # 2 splits + 1
    assert len(cpntr) == 3  # 2 splits + 1
    assert rpntr[0] == 0 and rpntr[-1] == 6
    assert cpntr[0] == 0 and cpntr[-1] == 6
    
    # Verify block structure
    assert len(bpntrb) == 2  # 2 row splits
    assert len(bpntre) == 2  # 2 row splits
    
    # Verify that we have some data
    assert len(val) > 0 or len(csr_val) > 0
    
    # Verify that indx starts with 0
    assert indx[0] == 0
    
    # Verify that bindx contains valid column indices
    assert all(0 <= idx < 2 for idx in bindx)  # 2 column splits
    
    # Verify that ublocks, indptr, indices, csr_val are consistent
    if len(ublocks) > 0:
        assert len(indptr) > 0
        assert len(indices) > 0
        assert len(csr_val) > 0
        # indptr should have length equal to number of rows + 1
        assert len(indptr) == 7  # 6 rows + 1
    
    # Clean up the generated file
    if os.path.exists(vbrc_path):
        os.remove(vbrc_path)
    # Also clean up the directory if it's empty
    if os.path.exists("tests") and not os.listdir("tests"):
        os.rmdir("tests")
