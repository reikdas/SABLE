import gc
import os
import pathlib

import numpy as np
import scipy
from scipy.sparse import csr_matrix, csc_matrix
from numba import njit

from utils.convert_real_to_vbr import convert_sparse_to_vbr, convert_vbr_to_compressed

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")


def cut_indices1(A, cut_threshold, similarity):
    col_indices = [0]  # Start with the first column index
    row_indices = [0]  # Start with the first row index

    # Check column similarities
    for i in range(A.shape[1] - 1):
        if similarity(A[:, i].toarray().ravel(), A[:, i + 1].toarray().ravel()) < cut_threshold:
            col_indices.append(i + 1)
    col_indices.append(A.shape[1])

    # Check row similarities
    for i in range(A.shape[0] - 1):
        if similarity(A[i, :].toarray().ravel(), A[i + 1, :].toarray().ravel()) < cut_threshold:
            row_indices.append(i + 1)
    row_indices.append(A.shape[0])

    return col_indices, row_indices

def cut_indices2(A_, cut_threshold, similarity, run = 3):
    #run is Max consecutive low-similarity columns to merge
    A = (A_ != 0).astype(int)  # Converts to binary (0,1) integers
    col_indices = [0]
    row_indices = [0]
    # Check column similarities
    i = 0
    run_idx = 1

    while i < A.shape[1] - 1:
        if similarity(A[:, i].toarray().ravel(), A[:, i + run_idx].toarray().ravel()) >= cut_threshold:
            i += 1
            continue
        else:
            run_idx += 1
            while (
                i + run_idx < A.shape[1] 
                and run_idx < run  # Ensure we don’t exceed max run length
                and similarity(A[:, i ].toarray().ravel(), A[:, i + run_idx].toarray().ravel()) < cut_threshold
            ):
                run_idx += 1
            
            if run_idx == run:
                col_indices.append(i+1)
            else: 
                i += run_idx
                run_idx = 1
                continue
                
            i += 1
            entered = False
            while (
                i < A.shape[1] -1
                and similarity(A[:, i].toarray().ravel(), A[:, i + 1].toarray().ravel()) < cut_threshold
            ):
                i += 1
                entered = True
            if entered:
                col_indices.append(i)
        run_idx = 1
    
    i = 0
    run_idx = 1

    while i < A.shape[0] - 1:
        if similarity(A[i, :].toarray().ravel(), A[i + run_idx, :].toarray().ravel()) >= cut_threshold:
            i += 1
            continue
        else:
            run_idx += 1
            while (
                i + run_idx < A.shape[0] 
                and run_idx < run  # Ensure we don’t exceed max run length
                and similarity(A[i , :].toarray().ravel(), A[i + run_idx, :].toarray().ravel()) < cut_threshold
            ):
                run_idx += 1

            if run_idx == run:
                row_indices.append(i+1)
            else:    
                i += run_idx
                run_idx = 1
                continue

            i += 1
            entered = False
            while (
                i < A.shape[0] - 1
                and similarity(A[i, :].toarray().ravel(), A[i + 1, :].toarray().ravel()) < cut_threshold
            ):
                i += 1
                entered = True
            if entered:
                row_indices.append(i)
        run_idx = 1
        
    if col_indices[-1] != A.shape[1]:
        col_indices.append(A.shape[1])
    if row_indices[-1] != A.shape[0]:
        row_indices.append(A.shape[0])
    return col_indices, row_indices




def similarity1(a, b):
    if np.count_nonzero(a) == 0 or np.count_nonzero(b) == 0:
        return 0
    return (a.dot(b) +a[1:].dot(b[:-1]) +a[:-1].dot(b[1:])) / (3*max(np.count_nonzero(a), np.count_nonzero(b)))

def similarity2(a, b):
    if np.count_nonzero(a) == 0 or np.count_nonzero(b) == 0:
        return 0
    return max(a.dot(b),a[1:].dot(b[:-1]),a[:-1].dot(b[1:])) / max(np.count_nonzero(a), np.count_nonzero(b))

def similarity3(a, b):
    if np.count_nonzero(a) == 0 or np.count_nonzero(b) == 0:
        return 0
    return (a.dot(b) +a[1:].dot(b[:-1]) +a[:-1].dot(b[1:])) / (3*max(np.count_nonzero(a), np.count_nonzero(b),60))

def similarity4(a, b):
    if np.count_nonzero(a) == 0 or np.count_nonzero(b) == 0:
        return 0
    return max(a.dot(b),a[1:].dot(b[:-1]),a[:-1].dot(b[1:])) / max(np.count_nonzero(a), np.count_nonzero(b), 20)

import numpy as np
from numba import njit

@njit
def fast_similarity(indices1, indices2, max_index):
    len1, len2 = len(indices1), len(indices2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Fast intersection without extra allocations
    i1 = i2 = 0
    common = shifted1 = shifted2 = 0

    while i1 < len1 and i2 < len2:
        if indices1[i1] == indices2[i2]:
            common += 1
            i1 += 1
            i2 += 1
        elif indices1[i1] < indices2[i2]:
            i1 += 1
        else:
            i2 += 1

    # Shifted forward (indices1 - 1 in indices2)
    i1 = i2 = 0
    while i1 < len1 and i2 < len2:
        if indices1[i1] - 1 == indices2[i2]:
            shifted1 += 1
            i1 += 1
            i2 += 1
        elif indices1[i1] - 1 < indices2[i2]:
            i1 += 1
        else:
            i2 += 1

    # Shifted backward (indices1 + 1 in indices2)
    i1 = i2 = 0
    while i1 < len1 and i2 < len2:
        if indices1[i1] + 1 == indices2[i2]:
            shifted2 += 1
            i1 += 1
            i2 += 1
        elif indices1[i1] + 1 < indices2[i2]:
            i1 += 1
        else:
            i2 += 1

    denominator = 3 * max(len1, len2)
    return (common + shifted1 + shifted2) / denominator if denominator > 0 else 0.0


def cut_indices2_sparse(A_, cut_threshold, run=3):
    """
    A_ is assumed to be a CSR sparse matrix.
    The function computes cut indices for columns and rows based on a custom similarity measure.
    """
    # Convert to binary and ensure CSR format for rows
    A_bin = (A_ != 0).astype(np.int8)
    n_rows, n_cols = A_bin.shape

    # Convert to CSC for fast column slicing
    A_csc = A_bin.tocsc()
    col_nz = [A_csc.indices[A_csc.indptr[j]:A_csc.indptr[j + 1]] for j in range(n_cols)]

    def sim_cols(i, j):
        return fast_similarity(col_nz[i], col_nz[j], n_rows)
    
    col_indices = [0]
    i = 0
    while i < n_cols - 1:
        if i % 1000 == 0:
            print(f"Processing column: {i}", flush=True)
        run_idx = 1
        while (i + run_idx < n_cols and run_idx < run and 
               sim_cols(i, i + run_idx) < cut_threshold):
            run_idx += 1

        if run_idx == run:
            col_indices.append(i + 1)
        else:
            i += run_idx
            continue

        i += 1
        while (i < n_cols - 1 and sim_cols(i, i + 1) < cut_threshold):
            i += 1
        col_indices.append(i)

    if col_indices[-1] != n_cols:
        col_indices.append(n_cols)

    # Use CSR for rows
    row_nz = [A_bin.indices[A_bin.indptr[i]:A_bin.indptr[i + 1]] for i in range(n_rows)]

    def sim_rows(i, j):
        return fast_similarity(row_nz[i], row_nz[j], n_cols)
    
    row_indices = [0]
    i = 0
    while i < n_rows - 1:
        run_idx = 1
        while (i + run_idx < n_rows and run_idx < run and 
               sim_rows(i, i + run_idx) < cut_threshold):
            run_idx += 1

        if run_idx == run:
            row_indices.append(i + 1)
        else:
            i += run_idx
            continue

        i += 1
        while (i < n_rows - 1 and sim_rows(i, i + 1) < cut_threshold):
            i += 1
        row_indices.append(i)

    if row_indices[-1] != n_rows:
        row_indices.append(n_rows)

    return col_indices, row_indices
# Example usage
# cut_threshold = 0.2
# col_indices, row_indices = cut_indices(A, cut_threshold)
# print("Column indices:", col_indices)
# print("Row indices:", row_indices)

# cut_threshold = 0.05
# col_indices, row_indices = cut_indices(A, cut_threshold)
# print("Column indices:", col_indices)
# print("Row indices:", row_indices)

# # Example usage
# cut_threshold = 0.4
# col_indices, row_indices = cut_indices(A, cut_threshold)
# print("Column indices:", col_indices)
# print("Row indices:", row_indices)

def my_convert_dense_to_vbr(file_info, cut_threshold, cut_indices, similarity):
    src_path, dest_path = file_info
    mtx = scipy.io.mmread(src_path)
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    del mtx
    gc.collect()
    cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
    val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr(A, rpntr, cpntr, pathlib.Path(src_path).resolve().stem, pathlib.Path(dest_path).resolve().parent)
    val, indx, bindx, bpntrb, bpntre, ublocks, coo_i, coo_j = convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, 8, pathlib.Path(src_path).resolve().stem, pathlib.Path(dest_path).resolve().parent)
    return val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, ublocks, coo_i, coo_j

# def partition_dlmc(mtx_dir, vbr_dir):
#     src_dir = pathlib.Path(os.path.join(BASE_PATH, mtx_dir))
#     dest_dir = pathlib.Path(os.path.join(BASE_PATH, vbr_dir))
#     parallel_dispatch(src_dir, dest_dir, cpu_count(), my_convert_dense_to_vbr, ".mtx", ".vbr")

# def partition_suitesparse():
#     src_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
#     dest_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr"))
#     parallel_dispatch(src_dir, dest_dir, cpu_count(), my_convert_dense_to_vbr, ".mtx", ".vbr")

# if __name__ == "__main__":
#     # partition_dlmc("Real_mtx", "Real_vbr")
#     partition_suitesparse()
