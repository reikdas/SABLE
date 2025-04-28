import os
import pathlib

import numpy as np
import numba as nb
import scipy

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

def cut_indices2(A, cut_threshold, similarity):
    col_indices = [0]  # Start with the first column index
    row_indices = [0]  # Start with the first row index
    
    # Check column similarities
    i = 0
    while i < A.shape[1] - 1:
        if A[:, i].nnz == 0:  # Start of an empty column sequence
            # Skip all consecutive empty columns
            while i < A.shape[1] - 1 and A[:, i + 1].nnz == 0:
                i += 1
            col_indices.append(i + 1)  # Mark the end of the empty block
        elif similarity(A[:, i].toarray().ravel(), A[:, i + 1].toarray().ravel()) < cut_threshold:
            col_indices.append(i + 1)
        i += 1
    
    if col_indices[-1] != A.shape[1]:
        col_indices.append(A.shape[1])

    # Check row similarities
    i = 0
    while i < A.shape[0] - 1:
        if A[i, :].nnz == 0:  # Start of an empty row sequence
            # Skip all consecutive empty rows
            while i < A.shape[0] - 1 and A[i + 1, :].nnz == 0:
                i += 1
            row_indices.append(i + 1)  # Mark the end of the empty block
        elif similarity(A[i, :].toarray().ravel(), A[i + 1, :].toarray().ravel()) < cut_threshold:
            row_indices.append(i + 1)
        i += 1
    
    if row_indices[-1] != A.shape[0]:
        row_indices.append(A.shape[0])

    return col_indices, row_indices

def similarity1(a, b):
    return (a.dot(b) + a[1:].dot(b[:-1])+a[:-1].dot(b[1:])) / (3*max(np.count_nonzero(a), np.count_nonzero(b)))

def similarity2(a, b):
    return max(a.dot(b),a[1:].dot(b[:-1]),a[:-1].dot(b[1:])) / max(np.count_nonzero(a), np.count_nonzero(b))

@nb.njit
def similarity2_numba(a, b):
    # Numba-accelerated version of similarity2
    count_a = np.count_nonzero(a)
    count_b = np.count_nonzero(b)
    if count_a == 0 or count_b == 0:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot_main = np.dot(a, b)
    
    if len(a) > 1 and len(b) > 1:
        dot_forward = np.dot(a[1:], b[:-1])
        dot_backward = np.dot(a[:-1], b[1:])
        max_dot = max(dot_main, dot_forward, dot_backward)
    else:
        max_dot = dot_main
    
    return max_dot / max(count_a, count_b)

def cut_indices2_fast(A, cut_threshold, similarity):
    """
    Optimized version of cut_indices2
    """
    # if similarity.__name__ == 'similarity1':
    #     sim_func = similarity1_numba
    # elif similarity.__name__ == 'similarity2':
    #     sim_func = similarity2_numba
    # else:
    #     sim_func = similarity  
    sim_func = similarity

    # Convert to optimized formats
    A_csr = A.tocsr() if not scipy.sparse.isspmatrix_csr(A) else A
    A_csc = A.tocsc() if not scipy.sparse.isspmatrix_csc(A) else A
    
    # Caches for dense arrays - sliding window
    col_cache = {}
    row_cache = {}
    
    col_indices = [0]
    row_indices = [0]
    
    # Column processing
    i = 0
    while i < A.shape[1] - 1:
        
        if A_csc[:, i].nnz == 0:
            # Skip consecutive empty columns
            while i < A.shape[1] - 1 and A_csc[:, i + 1].nnz == 0:
                i += 1
            col_indices.append(i + 1)
        else:
            if i not in col_cache:
                col_cache[i] = A_csc[:, i].toarray().ravel()
            if i+1 not in col_cache:
                col_cache[i+1] = A_csc[:, i+1].toarray().ravel()
            
            if sim_func(col_cache[i], col_cache[i+1]) < cut_threshold:
                col_indices.append(i + 1)
                
            # Cache
            if len(col_cache) > 10:
                keys_to_remove = sorted([k for k in col_cache if k < i-5])
                for k in keys_to_remove:
                    del col_cache[k]
        i += 1
    
    # last column 
    if col_indices[-1] != A.shape[1]:
        col_indices.append(A.shape[1])
    
    # Row processing 
    i = 0
    while i < A.shape[0] - 1:
        
        if A_csr[i, :].nnz == 0:
            # Skip consecutive empty rows
            while i < A.shape[0] - 1 and A_csr[i + 1, :].nnz == 0:
                i += 1
            row_indices.append(i + 1)
        else:
            if i not in row_cache:
                row_cache[i] = A_csr[i, :].toarray().ravel()
            if i+1 not in row_cache:
                row_cache[i+1] = A_csr[i+1, :].toarray().ravel()
            
            if sim_func(row_cache[i], row_cache[i+1]) < cut_threshold:
                row_indices.append(i + 1)
                
            # Cache 
            if len(row_cache) > 10: 
                keys_to_remove = sorted([k for k in row_cache if k < i-5])
                for k in keys_to_remove:
                    del row_cache[k]
        i += 1
    
    # last row 
    if row_indices[-1] != A.shape[0]:
        row_indices.append(A.shape[0])
    
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
