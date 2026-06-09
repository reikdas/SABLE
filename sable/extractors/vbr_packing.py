from __future__ import annotations

import scipy.sparse


Block = tuple[int, int, int, int]
VBRCData = tuple[
    list[float],  # val           (packed dense-block values, column-major)
    list[int],    # indx          (offsets into val, one per packed block + 1)
    list[int],    # bindx         (block-column of each packed block)
    list[int],    # rpntr
    list[int],    # cpntr
    list[int],    # bpntrb        (CSR-style row pointer into bindx/indx, len nbrows+1)
    list[int],    # csr_indptr    (residual)
    list[int],    # csr_indices   (residual)
    list[float],  # csr_val       (residual)
]


def _compute_partitioning_from_blocks(
    mat: scipy.sparse.spmatrix,
    blocks: list[Block],
) -> tuple[list[int], list[int]]:
    row_boundaries = {0, mat.shape[0]}
    col_boundaries = {0, mat.shape[1]}

    for row_start, row_end, col_start, col_end in blocks:
        row_boundaries.add(row_start)
        row_boundaries.add(row_end)
        col_boundaries.add(col_start)
        col_boundaries.add(col_end)

    return sorted(row_boundaries), sorted(col_boundaries)


def convert_matrix_to_vbrc_with_blocks(
    mat: scipy.sparse.spmatrix,
    blocks: list[Block],
) -> VBRCData:
    rpntr, cpntr = _compute_partitioning_from_blocks(mat, blocks)

    def block_processor(r_start: int, r_end: int, c_start: int, c_end: int):
        block = mat[r_start:r_end, c_start:c_end]
        if block.nnz == 0:
            return None

        is_selected_block = any(
            selected_r_start <= r_start
            and r_end <= selected_r_end
            and selected_c_start <= c_start
            and c_end <= selected_c_end
            for selected_r_start, selected_r_end, selected_c_start, selected_c_end in blocks
        )

        coo_block = block.tocoo()
        residual_elems = coo_block.data.tolist()
        idxs_i = (coo_block.row + r_start).tolist()
        idxs_j = (coo_block.col + c_start).tolist()
        block_vals = block.toarray().flatten(order="F")

        return block_vals, residual_elems, idxs_i, idxs_j, is_selected_block

    val: list[float] = []
    indx: list[int] = [0]
    bindx: list[int] = []
    bpntrb: list[int] = []
    coo_i: list[int] = []
    coo_j: list[int] = []
    coo_val: list[float] = []
    packed_count = 0

    # VBR holds *only* the selected dense blocks.  Everything else (including
    # non-selected sub-blocks of the partition) goes to the CSR residual, so
    # there is no ublocks list.  bpntrb is a standard CSR-style row pointer of
    # length nbrows+1 (row x's packed blocks are bindx[bpntrb[x]:bpntrb[x+1]]);
    # an empty block-row is just a zero-length range, so no -1 sentinel and no
    # separate bpntre is needed.
    for row_idx in range(len(rpntr) - 1):
        bpntrb.append(packed_count)
        r_start, r_end = rpntr[row_idx], rpntr[row_idx + 1]

        for col_idx in range(len(cpntr) - 1):
            c_start, c_end = cpntr[col_idx], cpntr[col_idx + 1]
            result = block_processor(r_start, r_end, c_start, c_end)
            if result is None:
                continue

            block_vals, residual_elems, idxs_i, idxs_j, is_selected_block = result
            if is_selected_block:
                val.extend(block_vals)
                indx.append(len(val))
                bindx.append(col_idx)
                packed_count += 1
            else:
                coo_val.extend(residual_elems)
                coo_i.extend(idxs_i)
                coo_j.extend(idxs_j)

    bpntrb.append(packed_count)

    if coo_i:
        if (rpntr[-1] - 1) not in coo_i or (cpntr[-1] - 1) not in coo_j:
            coo_i.append(rpntr[-1] - 1)
            coo_j.append(cpntr[-1] - 1)
            coo_val.append(0.0)
        csr = scipy.sparse.coo_array((coo_val, (coo_i, coo_j))).tocsr()
        indptr = csr.indptr.tolist()
        assert len(indptr) == rpntr[-1] + 1
        indices = csr.indices.tolist()
        csr_val = csr.data.tolist()
    else:
        indptr = []
        indices = []
        csr_val = []

    return val, indx, bindx, rpntr, cpntr, bpntrb, indptr, indices, csr_val
