import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import os
import pathlib
import scipy
from convert_real_to_vbr import convert_dense_to_vbr

src_dir = "Real_mtx"
dst_dir = "Real_vbr"

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

def similarity(a, b):
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

if __name__ == "__main__":
    for filename in os.listdir(f"{BASE_PATH}/{src_dir}"):
        mtx = scipy.io.mmread(f'{BASE_PATH}/{src_dir}/{filename}')
        A = scipy.sparse.csr_matrix(mtx)
        cut_threshold = 0.2
        cpntr, rpntr = cut_indices(A, cut_threshold)
        convert_dense_to_vbr(mtx.todense(), rpntr, cpntr, filename[:-4], dst_dir)
