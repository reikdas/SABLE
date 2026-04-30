from __future__ import annotations

import numpy
import scipy.sparse


Block = tuple[int, int, int, int]
VBRCData = tuple[
    list[float],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[float],
]


def _compute_partitioning_from_dense_blocks(
    mat: scipy.sparse.spmatrix,
    dense_blocks: list[Block],
) -> tuple[list[int], list[int]]:
    row_boundaries = {0, mat.shape[0]}
    col_boundaries = {0, mat.shape[1]}

    for row_start, row_end, col_start, col_end in dense_blocks:
        row_boundaries.add(row_start)
        row_boundaries.add(row_end)
        col_boundaries.add(col_start)
        col_boundaries.add(col_end)

    return sorted(row_boundaries), sorted(col_boundaries)


def convert_sparse_to_vbrc_with_blocks(
    mat: scipy.sparse.spmatrix,
    dense_blocks: list[Block],
) -> VBRCData:
    rpntr, cpntr = _compute_partitioning_from_dense_blocks(mat, dense_blocks)

    def block_processor(r_start: int, r_end: int, c_start: int, c_end: int):
        block = mat[r_start:r_end, c_start:c_end]
        if block.nnz == 0:
            return None

        is_dense_block = any(
            dense_r_start <= r_start
            and r_end <= dense_r_end
            and dense_c_start <= c_start
            and c_end <= dense_c_end
            for dense_r_start, dense_r_end, dense_c_start, dense_c_end in dense_blocks
        )

        coo_block = block.tocoo()
        dense_elems = coo_block.data.tolist()
        idxs_i = (coo_block.row + r_start).tolist()
        idxs_j = (coo_block.col + c_start).tolist()
        block_vals = block.todense().flatten(order="F").A1

        return block_vals, dense_elems, idxs_i, idxs_j, is_dense_block

    val: list[float] = []
    indx: list[int] = [0]
    bindx: list[int] = []
    bpntrb: list[int] = []
    bpntre: list[int] = []
    ublocks: list[int] = []
    coo_i: list[int] = []
    coo_j: list[int] = []
    coo_val: list[float] = []
    block_count = 0

    for row_idx in range(len(rpntr) - 1):
        row_start_block = block_count
        r_start, r_end = rpntr[row_idx], rpntr[row_idx + 1]

        for col_idx in range(len(cpntr) - 1):
            c_start, c_end = cpntr[col_idx], cpntr[col_idx + 1]
            result = block_processor(r_start, r_end, c_start, c_end)
            if result is None:
                continue

            block_vals, dense_elems, idxs_i, idxs_j, is_dense_block = result
            if is_dense_block:
                val.extend(block_vals)
                indx.append(len(val))
            else:
                coo_val.extend(dense_elems)
                ublocks.append(block_count)
                coo_i.extend(idxs_i)
                coo_j.extend(idxs_j)
            bindx.append(col_idx)
            block_count += 1

        if row_start_block < block_count:
            bpntrb.append(row_start_block)
            bpntre.append(block_count)
        else:
            bpntrb.append(-1)
            bpntre.append(-1)

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

    return val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val


def analyze_dense_blocks(
    val: list[float],
    indx: list[int],
    bindx: list[int],
    rpntr: list[int],
    cpntr: list[int],
    bpntrb: list[int],
    bpntre: list[int],
    ublocks: list[int],
) -> list[dict[str, float | int]]:
    dense_blocks: list[dict[str, float | int]] = []
    nnz_block = 0
    dense_count = 0

    for row_idx in range(len(rpntr) - 1):
        if bpntrb[row_idx] == -1:
            continue
        valid_cols = bindx[bpntrb[row_idx] : bpntre[row_idx]]
        for col_idx in range(len(cpntr) - 1):
            if col_idx not in valid_cols:
                continue
            if nnz_block not in ublocks:
                rows = rpntr[row_idx + 1] - rpntr[row_idx]
                cols = cpntr[col_idx + 1] - cpntr[col_idx]
                block_size = rows * cols
                start = indx[dense_count]
                end = indx[dense_count + 1] if dense_count + 1 < len(indx) else len(val)
                block_nnz = int(numpy.count_nonzero(val[start:end]))
                density_percent = (block_nnz / block_size) * 100 if block_size else 0.0
                dense_blocks.append(
                    {
                        "rows": rows,
                        "cols": cols,
                        "density_percent": density_percent,
                        "nnz": block_nnz,
                    }
                )
                dense_count += 1
            nnz_block += 1

    return dense_blocks
