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

def convert_sparse_to_vbr(mat, rpntr, cpntr, fname, dst_dir):
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
    Returns:
    VBR(val=[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, -1.0, 0.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 0.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 25.0, 8.0, 3.0, 12.0],
        indx=[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51],
        bindx=[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4],
        rpntr=[0, 2, 5, 6, 9, 11],
        cpntr=[0, 2, 5, 6, 9, 11],
        bpntrb=[0, 3, 5, 9, 11],
        bpntre=[3, 5, 9, 11, 13])
    '''
    dense = mat.todense()
    val = []
    indx = [0]
    bindx = []
    bpntrb = []
    bpntre = []

    num_blocks = 0    
    for r in range(len(rpntr)-1):
        blocks_in_row_partition = 0
        blocks_found = False
        for c in range(len(cpntr)-1):
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
