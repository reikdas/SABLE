import os
import pathlib
from multiprocessing import cpu_count

import numpy as np
import scipy
from scripts.convert_real_to_vbr import convert_sparse_to_vbr
from scripts.smtx_to_mtx import parallel_dispatch

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

# Read the matrix from a .mtx file
# matrix = mmread("/Users/amir/Documents/Pratyush/matrices/bcspwr06.mtx")
# boolean_matrix = matrix != 0
# A = csr_matrix(boolean_matrix)

# def similarity(a, b):
#     return a.dot(b) / max(np.count_nonzero(a), np.count_nonzero(b))

def cut_indices(A, cut_threshold):
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

# def adjusted_cut_indices(A, cut_threshold):
#     ##This one if sees low similarity, it will take one extra step for comparison
#     col_indices = [0]  # Start with the first column index
#     row_indices = [0]  # Start with the first row index

#     # Check column similarities
#     i = 0
#     while i < A.shape[1] - 1:
#         if similarity(A[:, i].toarray().ravel(), A[:, i + 1].toarray().ravel()) < cut_threshold:
#             if i + 2 < A.shape[1] and similarity(A[:, i].toarray().ravel(), A[:, i + 2].toarray().ravel()) < cut_threshold:
#                 col_indices.append(i + 1)
#                 i += 2  # Skip the next column index
#             else:
#                 col_indices.append(i + 1)
#         i += 1

#     if col_indices[-1] != A.shape[1]:
#         col_indices.append(A.shape[1])

#     # Check row similarities
#     i = 0
#     while i < A.shape[0] - 1:
#         if similarity(A[i, :].toarray().ravel(), A[i + 1, :].toarray().ravel()) < cut_threshold:
#             if i + 2 < A.shape[0] and similarity(A[i, :].toarray().ravel(), A[i + 2, :].toarray().ravel()) < cut_threshold:
#                 row_indices.append(i + 1)
#                 i += 2  # Skip the next row index
#             else:
#                 row_indices.append(i + 1)
#         i += 1

#     if row_indices[-1] != A.shape[0]:
#         row_indices.append(A.shape[0])

#     return col_indices, row_indices

def similarity(a, b):
    # RuntimeWarning: invalid value encountered in long_scalars
    return (a.dot(b) + a[1:].dot(b[:-1])+a[:-1].dot(b[1:])) / (3*max(np.count_nonzero(a), np.count_nonzero(b)))

# Example usage
# cut_threshold = 0.01
# col_indices, row_indices = cut_indices(A, cut_threshold)
# print("Column indices:", col_indices)
# print("Row indices:", row_indices)

# cut_threshold = 0.05
# col_indices, row_indices = cut_indices(A, cut_threshold)
# print("Column indices:", col_indices)
# print("Row indices:", row_indices)

# # Example usage
# cut_threshold = 0.04
# col_indices, row_indices = cut_indices(A, cut_threshold)
# print("Column indices:", col_indices)
# print("Row indices:", row_indices)

def my_convert_dense_to_vbr(file_info):
    src_path, dest_path = file_info
    mtx = scipy.io.mmread(src_path)
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    # print(src_path, A.shape[0], A.shape[1], A.nnz)
    cut_threshold = 0.2
    cpntr, rpntr = cut_indices(A, cut_threshold)
    convert_sparse_to_vbr(A, rpntr, cpntr, pathlib.Path(src_path).resolve().stem, pathlib.Path(dest_path).resolve().parent)

def partition_dlmc():
    src_dir = pathlib.Path(os.path.join(BASE_PATH, "Real_mtx"))
    dest_dir = pathlib.Path(os.path.join(BASE_PATH, "Real_vbr"))
    parallel_dispatch(src_dir, dest_dir, cpu_count(), my_convert_dense_to_vbr, ".mtx", ".vbr")

def partition_suitesparse():
    src_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
    dest_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr"))
    parallel_dispatch(src_dir, dest_dir, cpu_count(), my_convert_dense_to_vbr, ".mtx", ".vbr")

if __name__ == "__main__":
    # partition_dlmc()
    partition_suitesparse()
