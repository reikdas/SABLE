import os
import pathlib
from typing import List, Tuple

import numpy
import scipy
from scipy.sparse import spmatrix

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

def _compute_partitioning_from_dense_blocks(
    mat: spmatrix,
    dense_blocks: List[Tuple[int, int, int, int]]
) -> Tuple[List[int], List[int]]:
    """
    Compute rpntr and cpntr partitioning from dense block coordinates.

    Args:
        mat: Sparse matrix
        dense_blocks: List of dense block coordinates as (row_start, row_end, col_start, col_end)

    Returns:
        Tuple of (rpntr, cpntr) partitioning arrays
    """
    # Collect all unique row and column boundaries from dense blocks
    row_boundaries = set([0, mat.shape[0]])  # Always include start and end
    col_boundaries = set([0, mat.shape[1]])  # Always include start and end

    for dense_r_start, dense_r_end, dense_c_start, dense_c_end in dense_blocks:
        row_boundaries.add(dense_r_start)
        row_boundaries.add(dense_r_end)
        col_boundaries.add(dense_c_start)
        col_boundaries.add(dense_c_end)

    # Sort and convert to lists
    rpntr = sorted(row_boundaries)
    cpntr = sorted(col_boundaries)

    return rpntr, cpntr


def convert_sparse_to_vbrc_with_blocks(
    mat: spmatrix,
    dense_blocks: List[Tuple[int, int, int, int]]
) -> Tuple[List[float], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[float]]:
    """
    Convert sparse matrix to VBRC format using specified dense block coordinates.
    The partitioning (rpntr, cpntr) is computed from the dense block boundaries.

    Args:
        mat: Sparse matrix to convert
        dense_blocks: List of dense block coordinates as (row_start, row_end, col_start, col_end)
                     where row_end and col_end are exclusive (e.g., [1409, 1944) means rows 1409 to 1943)

    Returns:
        Tuple of VBRC data structures: (val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    """
    # Compute partitioning from dense block boundaries
    rpntr, cpntr = _compute_partitioning_from_dense_blocks(mat, dense_blocks)

    def block_processor(r_start, r_end, c_start, c_end, r_i, c_i):
        # Extract the block
        block = mat[r_start:r_end, c_start:c_end]
        nnz = block.nnz

        # Only process non-empty blocks
        if nnz == 0:
            return None

        block_sx, block_sy = block.shape

        # Check if this partitioning block is contained within any specified dense block
        # Since partitioning is computed from dense block boundaries, we check containment
        is_dense_block = False
        for dense_r_start, dense_r_end, dense_c_start, dense_c_end in dense_blocks:
            # Partitioning block is contained if it's within the dense block boundaries
            if (dense_r_start <= r_start and r_end <= dense_r_end and
                dense_c_start <= c_start and c_end <= dense_c_end):
                is_dense_block = True
                break

        # Extract non-zero elements and their indices using sparse operations
        coo_block = block.tocoo()
        dense_elems = coo_block.data.tolist()
        idxs_i = (coo_block.row + r_start).tolist()
        idxs_j = (coo_block.col + c_start).tolist()

        block_vals = block.todense().flatten(order='F').A1

        return block_vals, dense_elems, idxs_i, idxs_j, block_sx, block_sy, is_dense_block

    # Generate VBRC data structures
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

            # Process the block
            result = block_processor(r_start, r_end, c_start, c_end, r_i, c_i)
            if result is None:
                continue  # Skip empty blocks

            block_vals, dense_elems, idxs_i, idxs_j, block_sx, block_sy, is_dense_block = result

            if is_dense_block:
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

    return val2, indx2, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val



def analyze_dense_blocks(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks) -> List[dict]:
    """
    Analyze dense blocks in a matrix using VBR data structure.

    Args:
        val: Dense block values array
        indx: Index array for dense blocks
        bindx: Block index array
        rpntr: Row pointer array
        cpntr: Column pointer array
        bpntrb: Block pointer begin array
        bpntre: Block pointer end array
        ublocks: Unrolled (sparse) blocks list

    Returns:
        List of dictionaries containing dense block information:
        - rows: number of rows in block
        - cols: number of columns in block
        - density_percent: calculated density percentage
        - nnz: number of non-zeros in block
    """
    dense_blocks = []

    # Count dense blocks using same logic as code generation
    nnz_block = 0
    count = 0

    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    # This is a dense block
                    i_extent = rpntr[a+1] - rpntr[a]
                    j_extent = cpntr[b+1] - cpntr[b]

                    # Calculate density based on the block size and actual non-zero count
                    block_size = i_extent * j_extent
                    # Determine slice of `val` for this dense block (indx is ordered by dense blocks)
                    start = indx[count]
                    end = indx[count+1] if (count+1) < len(indx) else len(val)
                    block_vals = val[start:end]
                    # `val` contains explicit zeros for dense blocks; count real non-zeros
                    block_nnz = int(numpy.count_nonzero(block_vals))
                    calc_density = (block_nnz / block_size) * 100 if block_size > 0 else 0

                    dense_blocks.append({
                        "rows": i_extent,
                        "cols": j_extent,
                        "density_percent": calc_density,
                        "nnz": block_nnz,
                    })
                    count += 1
                nnz_block += 1

    return dense_blocks
