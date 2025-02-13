import gc
import os
import pathlib

import numpy
import scipy
from scipy.io import mmread

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

def convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, density, fname, dst_dir):
    val2: list[float] = []
    indx2: list[int] = [0]
    ublocks: list[int] = []
    coo_i: list[int] = []
    coo_j: list[int] = []
    coo_val: list[float] = []
    count: int = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                sparse_count = 0
                count2 = 0
                dense_elems = []
                idxs_i = []
                idxs_j = []
                for idx_j in range(cpntr[b], cpntr[b+1]):
                    for idx_i in range(rpntr[a], rpntr[a+1]):
                        if val[indx[count]+count2] == 0.0:
                            sparse_count+=1
                        else:
                            dense_elems += [val[indx[count]+count2]]
                            idxs_i.append(idx_i)
                            idxs_j.append(idx_j)
                        count2+=1
                dense_count = len(dense_elems)
                if (dense_count/(dense_count + sparse_count))*100 > density:
                    val2.extend(val[indx[count]:indx[count+1]])
                    indx2.append(len(val2))
                else:
                    coo_val.extend(dense_elems)
                    ublocks.append(count)
                    coo_i.extend(idxs_i)
                    coo_j.extend(idxs_j)
                count+=1
    with open(os.path.join(dst_dir, f"{fname}.vbrc"), "w") as f:
        f.write(f"val=[{','.join(map(str, val2))}]\n")
        f.write(f"coo_val=[{','.join(map(str, coo_val))}]\n")
        f.write(f"indx=[{','.join(map(str, indx2))}]\n")
        f.write(f"bindx=[{','.join(map(str, bindx))}]\n")
        f.write(f"rpntr=[{','.join(map(str, rpntr))}]\n")
        f.write(f"cpntr=[{','.join(map(str, cpntr))}]\n")
        f.write(f"bpntrb=[{','.join(map(str, bpntrb))}]\n")
        f.write(f"bpntre=[{','.join(map(str, bpntre))}]\n")
        f.write(f"ublocks=[{','.join(map(str, ublocks))}]\n")
        f.write(f"coo_i=[{','.join(map(str, coo_i))}]\n")
        f.write(f"coo_j=[{','.join(map(str, coo_j))}]\n")
    return val2, indx2, bindx, bpntrb, bpntre, ublocks, coo_i, coo_j, coo_val

def convert_sparse_to_vbr(csc_mat, rpntr, cpntr, fname, dst_dir):
    '''
    Converts a CSC matrix to a VBR matrix.
    Example:
    Inputs:
    dense = [
        [ 4.  2. | 0.  0.  0. | 1. | 0.  0.  0. |0.  1.]
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
        [ 8.  4. | 0.  0.  0. | 0. | 0.  0.  0. |0.  3.]
        [-2.  3. | 0.  0.  0. | 0. | 0.  0.  0. | 0. 12.]
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
    
    val = []
    indx = [0]
    bindx = []
    bpntrb = []
    bpntre = []

    # Dictionary to store partition mappings
    d = {}

    row_part_lookup = {}
    for i in range(len(rpntr) - 1):
        start, end = rpntr[i], rpntr[i + 1]
        for idx in range(start, end):
            row_part_lookup[idx] = (start, end)
    
    # Find containing partitions and build dictionary
    for c in range(0, len(cpntr)-1):
        for j in range(cpntr[c], cpntr[c+1]):
            ridxs = csc_mat.indices[csc_mat.indptr[j]:csc_mat.indptr[j+1]]
            dense_parts = {row_part_lookup[num] for num in ridxs}         
            if len(dense_parts) > 0:
                d[(cpntr[c], cpntr[c+1])] = dense_parts
                break
    
    all_row_parts = list(zip(rpntr[:-1], rpntr[1:]))

    col_part_to_index = {(cpntr[i], cpntr[i+1]): i for i in range(len(cpntr)-1)}
    
    val = []
    num_blocks = 0
    # Process each row partition
    for row_part in all_row_parts:
        # Find all column partitions that contain this row partition
        relevant_cols = sorted(
            col_part for col_part, row_parts in d.items() 
            if row_part in row_parts
        )
        if len(relevant_cols) > 0:
            bpntrb.append(num_blocks)
            bpntre.append(num_blocks + len(relevant_cols))
            num_blocks += len(relevant_cols)
            for col_part in relevant_cols:
                # Find the index of col_part in cpntr partitions
                bindx.append(col_part_to_index[col_part])
                col_start, col_end = col_part
                row_start, row_end = row_part
                
                # Extract the submatrix for this partition
                submat = csc_mat[row_start:row_end, col_start:col_end]
                
                # Convert to dense array and flatten in column-major order (F-order)
                try:
                    submat_values = submat.toarray().flatten(order='F')
                except numpy._core._exceptions._ArrayMemoryError:
                    print(f"Skipping {fname} due to memory error")
                    return None, None, None, None, None
                val.extend(submat_values)
                del submat
                del submat_values
                gc.collect()
                indx.append(len(val))
        else:
            bpntrb.append(-1)
            bpntre.append(-1)
            continue
    
    with open(os.path.join(dst_dir, f"{fname}.vbr"), "w") as f:
        f.write(f"val=[{','.join(map(str, val))}]\n")
        f.write(f"indx=[{','.join(map(str, indx))}]\n")
        f.write(f"bindx=[{','.join(map(str, bindx))}]\n")
        f.write(f"rpntr=[{','.join(map(str, rpntr))}]\n")
        f.write(f"cpntr=[{','.join(map(str, cpntr))}]\n")
        f.write(f"bpntrb=[{','.join(map(str, bpntrb))}]\n")
        f.write(f"bpntre=[{','.join(map(str, bpntre))}]\n")
    return val, indx, bindx, bpntrb, bpntre

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
        convert_sparse_to_vbr(scipy.sparse.csc_matrix(mtx), rpntr, cpntr, filename.resolve().stem, dst_dir)
