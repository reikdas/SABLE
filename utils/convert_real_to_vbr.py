import os
import pathlib
from typing import List, Optional, Tuple

import joblib
import numpy
import scipy
from scipy.io import mmread
from scipy.sparse import spmatrix
from random import sample

def cumsum_list(l):
    '''
    method to calculate cumulative sum list of a list
    [1,2,3] -> [0,1,3,6]
    '''
    assert(type(l) == list)
    l2 = [0]
    for elem in l:
        l2.append(l2[-1] + elem)
    return l2

def random_splits(n, a):
    if n <= 0 or a <= 0:
        raise ValueError("Both n and a must be positive integers.")

    # Generate a list of 'a-1' random numbers between 1 and n-1
    split_numbers = sorted(sample(range(1, n), a - 1))

    # Calculate the differences between consecutive split numbers
    differences = [split_numbers[0]] + [split_numbers[i] - split_numbers[i - 1] for i in range(1, a - 1)] + [n - split_numbers[-1]]

    return differences

from src.consts import *

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")


def _generate_vbrc_data(
    rpntr: List[int], 
    cpntr: List[int], 
    density: Optional[float] = None,
    block_processor=None
) -> Tuple[List[float], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[float]]:
    """
    Shared helper function to generate VBRC data structures.
    
    Args:
        rpntr: Row pointer array
        cpntr: Column pointer array  
        density: Density threshold for unrolling (if None, uses ML model)
        block_processor: Function that processes each block and returns (block_vals, dense_elems, idxs_i, idxs_j, block_sx, block_sy)
    
    Returns:
        Tuple of VBRC data structures: (val2, indx2, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    """
    # Load ML model if density is None
    if density is None:
        model = joblib.load(os.path.join(BASE_PATH, "models", "density_threshold_spmv.pkl"))
    
    val2: List[float] = []
    indx2: List[int] = [0]
    bindx: List[int] = []
    bpntrb: List[int] = []
    bpntre: List[int] = []
    ublocks: List[int] = []
    coo_i: List[int] = []
    coo_j: List[int] = []
    coo_val: List[float] = []
    
    block_count = 0
    
    for r_i in range(len(rpntr) - 1):
        row_start_block = block_count
        r_start, r_end = rpntr[r_i], rpntr[r_i + 1]
        
        for c_i in range(len(cpntr) - 1):
            c_start, c_end = cpntr[c_i], cpntr[c_i + 1]
            
            # Use the provided block processor to get block data
            result = block_processor(r_start, r_end, c_start, c_end, r_i, c_i)
            if result is None:
                continue  # Skip empty blocks
                
            block_vals, dense_elems, idxs_i, idxs_j, block_sx, block_sy = result
            size = block_sx * block_sy
            
            dense_count = len(dense_elems)
            calc_density = (dense_count / size) * 100
            
            # Decision logic
            if density is not None:
                unroll = calc_density <= density
            else:
                unroll = model.predict([[block_sx, block_sy, calc_density]])[0] == 0
            
            if not unroll:
                # Keep as dense block
                val2.extend(block_vals)
                indx2.append(len(val2))
                bindx.append(c_i)
            else:
                # Unroll to CSR
                coo_val.extend(dense_elems)
                ublocks.append(block_count)
                coo_i.extend(idxs_i)
                coo_j.extend(idxs_j)
                bindx.append(c_i)
            
            block_count += 1
        
        # Update row pointers
        if row_start_block < block_count:
            bpntrb.append(row_start_block)
            bpntre.append(block_count)
        else:
            # Empty row - mark with -1
            bpntrb.append(-1)
            bpntre.append(-1)
    
    # Create CSR representation for unrolled blocks
    if len(coo_i) > 0:
        if (rpntr[-1]-1) not in coo_i or (cpntr[-1]-1) not in coo_j:
            coo_i.append(rpntr[-1]-1)
            coo_j.append(cpntr[-1]-1)
            coo_val.append(0.0)
        csr = scipy.sparse.coo_array((coo_val, (coo_i, coo_j))).tocsr()
        indptr = csr.indptr.tolist()
        assert(len(indptr) == (rpntr[-1]+1))
        indices = csr.indices.tolist()
        csr_val = csr.data.tolist()
    else:
        csr_val = []
        indptr = []
        indices = []
    
    return val2, indx2, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val


def _write_vbrc_file(
    fname: str, 
    dst_dir: str, 
    val2: List[float], 
    indx2: List[int], 
    bindx: List[int], 
    rpntr: List[int], 
    cpntr: List[int], 
    bpntrb: List[int], 
    bpntre: List[int], 
    ublocks: List[int], 
    indptr: List[int], 
    indices: List[int], 
    csr_val: List[float]
) -> None:
    """Write VBRC data to file."""
    # Ensure output directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Write to file
    with open(os.path.join(dst_dir, f"{fname}.vbrc"), "w") as f:
        f.write(f"val=[{','.join(map(str, val2))}]\n")
        f.write(f"csr_val=[{','.join(map(str, csr_val))}]\n")
        f.write(f"indptr=[{','.join(map(str, indptr))}]\n")
        f.write(f"indices=[{','.join(map(str, indices))}]\n")
        f.write(f"indx=[{','.join(map(str, indx2))}]\n")
        f.write(f"bindx=[{','.join(map(str, bindx))}]\n")
        f.write(f"rpntr=[{','.join(map(str, rpntr))}]\n")
        f.write(f"cpntr=[{','.join(map(str, cpntr))}]\n")
        f.write(f"bpntrb=[{','.join(map(str, bpntrb))}]\n")
        f.write(f"bpntre=[{','.join(map(str, bpntre))}]\n")
        f.write(f"ublocks=[{','.join(map(str, ublocks))}]\n")


def convert_sparse_to_vbrc(
    mat: spmatrix, 
    rpntr: List[int], 
    cpntr: List[int], 
    fname: str, 
    dst_dir: str, 
    density: Optional[float] = None
) -> Tuple[List[float], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[float]]:
    """Convert sparse matrix to VBRC format."""
    
    def block_processor(r_start, r_end, c_start, c_end, r_i, c_i):
        # Extract the block
        block = mat[r_start:r_end, c_start:c_end]
        nnz = block.nnz
        
        # Only process non-empty blocks
        if nnz == 0:
            return None
            
        block_sx, block_sy = block.shape
        
        # Calculate density and decide whether to unroll
        dense_elems = []
        idxs_i = []
        idxs_j = []
        
        # Extract non-zero elements and their indices
        for idx_j in range(c_start, c_end):
            for idx_i in range(r_start, r_end):
                val = block[idx_i - r_start, idx_j - c_start]
                if val != 0.0:
                    dense_elems.append(val)
                    idxs_i.append(idx_i)
                    idxs_j.append(idx_j)
        
        block_vals = block.todense().flatten(order='F').A1
        return block_vals, dense_elems, idxs_i, idxs_j, block_sx, block_sy
    
    # Generate VBRC data
    val2, indx2, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = _generate_vbrc_data(
        rpntr, cpntr, density, block_processor
    )
    
    # Write to file
    _write_vbrc_file(fname, dst_dir, val2, indx2, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    
    return val2, indx2, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val



def convert_sparse_to_vbr(mat: spmatrix, rpntr: List[int], cpntr: List[int], fname: str, dst_dir: str) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    '''
    Converts a matrix to a VBR matrix. If matrix provided is sparse, should be
    csc / csr (allowing slicing). The return is a set of arrays describing all
    the blocks that are fully materialized and stored (note zeros in these
    materialized blocks will be stored).

    Inputs:
    matrix = [
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
    Returns:
    VBR(val=[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, -1.0, 0.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 0.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 25.0, 8.0, 3.0, 12.0],
        indx=[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51],
        bindx=[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4],
        rpntr=[0, 2, 5, 6, 9, 11],
        cpntr=[0, 2, 5, 6, 9, 11],
        bpntrb=[0, 3, 5, 9, 11],
        bpntre=[3, 5, 9, 11, 13])
    '''

    val: List[float] = []
    indx: List[int] = [0]
    bindx: List[int] = []
    bpntrb: List[int] = []
    bpntre: List[int] = []
    
    block_count = 0
    
    for r_i in range(len(rpntr) - 1):
        row_start_block = block_count
        r_start, r_end = rpntr[r_i], rpntr[r_i + 1]
        
        for c_i in range(len(cpntr) - 1):
            c_start, c_end = cpntr[c_i], cpntr[c_i + 1]
            
            # Extract the block
            block = mat[r_start:r_end, c_start:c_end]
            nnz = block.nnz
            
            # Only process non-empty blocks
            if nnz > 0:
                block_vals = block.todense().flatten(order='F').A1
                val.extend(block_vals)
                indx.append(len(val))
                bindx.append(c_i)
                block_count += 1
        
        # Update row pointers
        if row_start_block < block_count:
            bpntrb.append(row_start_block)
            bpntre.append(block_count)
        else:
            # Empty row - mark with -1
            bpntrb.append(-1)
            bpntre.append(-1)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # save to file
    with open(os.path.join(dst_dir, f"{fname}.vbr"), "w") as f:
        f.write(f"val=[{','.join(map(str, val))}]\n")
        f.write(f"indx=[{','.join(map(str, indx))}]\n")
        f.write(f"bindx=[{','.join(map(str, bindx))}]\n")
        f.write(f"rpntr=[{','.join(map(str, rpntr))}]\n")
        f.write(f"cpntr=[{','.join(map(str, cpntr))}]\n")
        f.write(f"bpntrb=[{','.join(map(str, bpntrb))}]\n")
        f.write(f"bpntre=[{','.join(map(str, bpntre))}]\n")

    return numpy.array(val), numpy.array(indx), numpy.array(bindx), numpy.array(bpntrb), numpy.array(bpntre)

def _generate_vbrc_filename(m: int, n: int, partitioning: str, row_split: int, col_split: int, num_dense: int, perc_dense_zeros: int, num_sparse: int, dense_blocks_only: bool) -> str:
    """
    Generate the filename for a VBRC matrix without creating the actual VBRC data.
    
    Args:
        m: Number of rows in the matrix
        n: Number of columns in the matrix
        partitioning: Type of partitioning ("uniform" or "nonuniform")
        row_split: Number of row splits
        col_split: Number of column splits
        num_dense: Number of dense blocks
        perc_dense_zeros: Percentage of zeros in dense blocks
        num_sparse: Number of sparse blocks
        dense_blocks_only: Whether to generate only dense blocks
    
    Returns:
        str: The filename for the VBRC matrix (without extension)
    """
    if dense_blocks_only:
        filename: str = f"Matrix_{m}_{n}_{row_split}_{col_split}_{num_dense}_{perc_dense_zeros}_{partitioning}"
    else:
        filename: str = f"Matrix_{m}_{n}_{row_split}_{col_split}_{num_dense}_{perc_dense_zeros}_{num_sparse}_{partitioning}"
    
    return filename


def vbrc_matrix_gen(m: int, n: int, partitioning: str, row_split: int, col_split: int, num_dense: int, perc_dense_zeros: int, num_sparse: int, dense_blocks_only: bool, vbr_dir: str = "Generated_VBR", density: Optional[float] = None) -> str:
    """
    Generate a VBRC matrix directly in a single iteration without any intermediate formats.
    
    This function creates VBRC format directly from the matrix generation process,
    eliminating all intermediate steps and data structures.
    
    Args:
        m: Number of rows in the matrix
        n: Number of columns in the matrix
        partitioning: Type of partitioning ("uniform" or "nonuniform")
        row_split: Number of row splits
        col_split: Number of column splits
        num_dense: Number of dense blocks
        perc_dense_zeros: Percentage of zeros in dense blocks
        num_sparse: Number of sparse blocks
        dense_blocks_only: Whether to generate only dense blocks
        vbr_dir: Directory to save the .vbrc file (no automatic suffixes will be added)
        density: Density threshold for unrolling (if None, uses ML model)
    
    Returns:
        str: The filename of the generated .vbrc file (without extension)
    """
    
    # Generate partitioning
    assert(m % row_split == 0)
    assert(n % col_split == 0)
    if partitioning == "nonuniform":
        rpntr = cumsum_list(random_splits(m, row_split))
        cpntr = cumsum_list(random_splits(n, col_split))
    elif partitioning == "uniform":
        rpntr = [x for x in range(0, m+1, m//row_split)]
        cpntr = [x for x in range(0, n+1, n//col_split)]
    else:
        assert(False)
    
    num_blocks = row_split * col_split

    # Randomly choose dense blocks
    dense_blocks = sample([x for x in range(num_blocks)], num_dense)

    remaining_blocks = [x for x in range(num_blocks) if x not in dense_blocks]
    sparse_blocks = sample(remaining_blocks, num_sparse)
    dense_in_sparse_blocks = 10

    all_blocks = dense_blocks + sparse_blocks
    all_blocks.sort() # Easier handling of bpntrb/bpntre/bindx

    # Create a mapping from (row, col) to block data
    block_data_map = {}
    for block in all_blocks:
        new_row = block//col_split
        col_idx = block%col_split
        r_start, r_end = rpntr[new_row], rpntr[new_row + 1]
        c_start, c_end = cpntr[col_idx], cpntr[col_idx + 1]
        block_sx = r_end - r_start
        block_sy = c_end - c_start
        block_size = block_sx * block_sy
        
        # Generate block values
        if block in dense_blocks:
            zeros = sample([x for x in range(block_size)], (block_size * perc_dense_zeros) // 100)
        else:
            # Ensure we don't try to sample more elements than available
            num_zeros = max(0, block_size - dense_in_sparse_blocks)
            zeros = sample([x for x in range(block_size)], num_zeros)
        zeros = set(zeros)
        
        # Create block values array
        block_vals = []
        for index in range(block_size):
            if index in zeros:
                block_vals.append(0.0)
            else:
                block_vals.append(1.0)
        
        # Calculate density and extract non-zero elements
        dense_elems = []
        idxs_i = []
        idxs_j = []
        
        # Extract non-zero elements and their indices
        for idx in range(len(block_vals)):
            if block_vals[idx] != 0.0:
                # Convert linear index to 2D indices (Fortran order)
                row_idx = idx % block_sx
                col_idx_2d = idx // block_sx
                dense_elems.append(block_vals[idx])
                idxs_i.append(r_start + row_idx)
                idxs_j.append(c_start + col_idx_2d)
        
        block_data_map[(new_row, col_idx)] = (block_vals, dense_elems, idxs_i, idxs_j, block_sx, block_sy)
    
    def block_processor(r_start, r_end, c_start, c_end, r_i, c_i):
        # Find the block in our pre-generated data
        if (r_i, c_i) in block_data_map:
            return block_data_map[(r_i, c_i)]
        return None  # Empty block
    
    # Generate VBRC data using shared helper
    val2, indx2, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = _generate_vbrc_data(
        rpntr, cpntr, density, block_processor
    )
    
    # Generate filename using helper function
    filename = _generate_vbrc_filename(m, n, partitioning, row_split, col_split, num_dense, perc_dense_zeros, num_sparse, dense_blocks_only)
    
    # Write to file using shared helper
    _write_vbrc_file(filename, vbr_dir, val2, indx2, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    
    return filename


if __name__ == "__main__":
    d = {}
    d["bcsstk04.mtx"] = [
        [0, 5, 11, 17, 24, 30, 35, 42, 45, 54, 60, 65, 71, 77, 84, 90, 95, 101, 108, 114, 120, 125, 132],
        [0, 5, 11, 17, 24, 30, 35, 42, 45, 54, 60, 65, 71, 77, 84, 90, 95, 101, 108, 114, 120, 125, 132]
    ]
    d["bcspwr06.mtx"] = [
        [0, 208, 288, 340, 392, 445, 500, 576, 840, 995, 1032, 1088, 1118, 1154, 1236, 1296, 1340, 1356, 1424, 1454],
        [0, 208, 288, 340, 392, 445, 500, 576, 840, 995, 1032, 1088, 1118, 1154, 1236, 1296, 1340, 1356, 1424, 1454]
    ]
    d["bibd_12_5.mtx"] = [
        [0, 10, 20, 30, 40, 66],
        [0, 118, 203, 260, 295, 315, 330, 360, 414, 470, 504, 540, 595, 631, 651, 666, 701, 721, 736, 756, 771, 792],
    ]
    d["bibd_12_4.mtx"] = [
        [0, 10, 20, 30, 37, 45, 66],
        [0, 45, 81, 109, 130, 165, 201, 250, 265, 285, 313, 334, 349, 359, 369, 390, 425, 460, 480, 495]
    ]
    d["bibd_9_5.mtx"] = [
        [0, 8, 15, 20, 26, 36],
        [0, 15, 35, 55, 70, 90, 105, 120, 126]
    ]
    d["bibd_11_5.mtx"] = [
        [0, 10, 19, 35, 55],
        [0, 30, 50, 82, 103, 205, 335, 462],
    ]
    for mtx_name, (rpntr, cpntr) in d.items():
        filename = pathlib.Path(os.path.join(BASE_PATH, "manual_mtx", mtx_name))
        dst_dir = pathlib.Path(os.path.join(BASE_PATH, "manual_vbr"))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        mtx = mmread(filename)
        convert_sparse_to_vbr(scipy.sparse.csc_matrix(mtx), rpntr, cpntr, filename.resolve().stem, str(dst_dir))
