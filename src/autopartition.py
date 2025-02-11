import gc
import os
import pathlib

import numpy as np
import scipy

# from utils.convert_real_to_vbr import convert_sparse_to_vbr, convert_vbr_to_compressed

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
    A = A_!=0
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
    return (a.dot(b) +a[1:].dot(b[:-1]) +a[:-1].dot(b[1:])) / (3*max(np.count_nonzero(a), np.count_nonzero(b),5))


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

if __name__ == "__main__":
    cut_indices = cut_indices2
    similarity = similarity2
    dense = np.array([[ 4.,  2.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.,  1.],
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
    A = scipy.sparse.csc_matrix(dense)
    cpntr, rpntr = cut_indices(A, 0.2, similarity)