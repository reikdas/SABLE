import gc
import scipy
import numpy

from src.vbr import VBR

def del_stuff(arr):
    '''
    Deletes the input array and calls the garbage collector.
    '''
    assert type(arr) == list
    for a in arr:
        del a
    gc.collect()

def get_accumulated_values(arr: list) -> list:
    '''
    Returns the accumulated values of the input array.
    Example:
    get_accumulated_values([1, 2, 3, 4]) -> [0, 1, 3, 6, 10]
    '''
    assert type(arr) == list
    acc = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        acc[i+1] += acc[i] + arr[i]
    return acc

def correct_widths(widths: list, dim: int) -> list:
    '''
    Summation of widths should be equal to the dimension of the matrix. If not equal, this function appends the difference to the widths.
    Example:
    correct_widths([2, 3, 1, 3], 10) -> [2, 3, 1, 3, 1]
    '''
    assert type(widths) == list
    sum_widths = sum(widths)
    assert sum_widths <= dim
    if sum_widths != dim:
        widths.append(dim - sum_widths)
    return widths

def convert_dense_to_vbr(dense, row_widths, col_widths):
    '''
    Converts a dense matrix to a VBR matrix.
    Example:
    Inputs:
    dense = [
        [ 4.  2. | 0.  0.  0. | 1. | 0.  0.  0. |-1.  1.]
        [ 1.  5. | 0.  0.  0. | 2. | 0.  0.  0. | 0. -1.]
        -------------------------------------------------
        [ 0.  0. | 6.  1.  2. | 2. | 0.  0.  0. | 0.  0.]
        [ 0.  0. | 2.  7.  1. | 0. | 0.  0.  0. | 0.  0.]
        [ 0.  0. |-1.  2.  9. | 3. | 0.  0.  0. | 0.  0.]
        -------------------------------------------------
        [ 2.  1. | 3.  4.  5. |10. | 4.  3.  2. | 0.  0.]
        -------------------------------------------------
        [ 0.  0. | 0.  0.  0. | 4. |13.  4.  2. | 0.  0.]
        [ 0.  0. | 0.  0.  0. | 3. | 3. 11.  3. | 0.  0.]
        [ 0.  0. | 0.  0.  0. | 0. | 2.  0.  7. | 0.  0.]
        -------------------------------------------------
        [ 8.  4. | 0.  0.  0. | 0. | 0.  0.  0. |25.  3.]
        [-2.  3. | 0.  0.  0. | 0. | 0.  0.  0. | 8. 12.]
    ]
    row_widths = [2, 3, 1, 3, 2]
    col_widths = [2, 3, 1, 3, 2]
    Returns:
    VBR(val=[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, -1.0, 0.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 0.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 25.0, 8.0, 3.0, 12.0],
        indx=[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51],
        bindx=[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4],
        rpntr=[0, 2, 5, 6, 9, 11],
        cpntr=[0, 2, 5, 6, 9, 11],
        bpntrb=[0, 3, 5, 9, 11],
        bpntre=[3, 5, 9, 11, 13])
    '''
    
    assert type(dense) == numpy.ndarray
    
    val = []
    indx = [0]
    bindx = []
    rpntr = []
    cpntr = []
    bpntrb = []
    bpntre = []
    
    row_widths = correct_widths(row_widths, dense.shape[0])
    col_widths = correct_widths(col_widths, dense.shape[1])
    
    rpntr = get_accumulated_values(row_widths)
    cpntr = get_accumulated_values(col_widths)

    num_blocks = 0    
    for r, rw in enumerate(row_widths):
        blocks_in_row_partition = 0
        blocks_found = False
        for c, cw in enumerate(col_widths):
            r0 = rpntr[r]
            r1 = rpntr[r+1]
            c0 = cpntr[c]
            c1 = cpntr[c+1]
            
            non_zero_element_found = False
            values = []
            for j in range(c0, c1):
                for i in range(r0, r1):
                    if dense[i, j] != 0:
                        non_zero_element_found = True
                        blocks_found = True
                    values.append(dense[i, j])
                        
            if non_zero_element_found:
                blocks_in_row_partition += 1
                val.extend(values)
                indx.append(len(val))
                bindx.append(c)
                
        if blocks_found:
            bpntrb.append(num_blocks)
            bpntre.append(num_blocks + blocks_in_row_partition)
            num_blocks += blocks_in_row_partition
        else:
            bpntrb.append(-1)
            bpntre.append(-1)
            
    return VBR(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)

def convert_mtx_to_vbr(coo: scipy.sparse._coo.coo_matrix, row_widths: list = None, col_widths: list = None) -> None:
    '''
    Converts a matrix in Matrix Market format to VBR format.
    '''
    assert type(coo) == scipy.sparse._coo.coo_matrix
    dense = coo.toarray()
    
    if row_widths is None:
        row_widths = [2 for _ in range(0, dense.shape[0]//2)]
    if col_widths is None:
        col_widths = [2 for _ in range(0, dense.shape[1]//2)]
    
    return convert_dense_to_vbr(dense, row_widths, col_widths)
