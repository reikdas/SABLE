from random import sample

from src.fileio import write_vbr_matrix
from src.vbr import VBR

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

def vbr_matrix_gen(m: int, n: int, partitioning: str, row_split: int, col_split: int, num_dense: int, perc_zeros: int) -> None:
    assert(m%row_split == 0)
    assert(n%col_split == 0)
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
    dense_blocks.sort() # Easier handling of bpntrb/bpntre/bindx

    val = []
    indx = [0]
    bindx = []
    bpntrb = []
    bpntre = []
    curr_row = 0
    numzeros = 0
    for dense_block in dense_blocks:
        new_row = dense_block//col_split
        col_idx = dense_block%col_split
        block_size = (rpntr[new_row+1] - rpntr[new_row]) * (cpntr[col_idx+1] - cpntr[col_idx])
        zeros = sample([x for x in range(block_size)], (block_size * perc_zeros) // 100)
        zeros = set(zeros)
        numzeros += len(zeros)
        
        for index in range(block_size):
            if index in zeros:
                val.append(0)
            else:
                val.append(1.0)
        
        indx.append(indx[-1] + block_size)
        if new_row != curr_row:
            if curr_row == 0 and len(bpntrb) == 0:
                for _ in range(curr_row, new_row):
                    bpntrb.append(-1)
                    bpntre.append(-1)
            else:
                if (len(bpntrb) > 0):
                    bpntre.append(len(bindx))
                for _ in range(curr_row+1, new_row):
                    bpntrb.append(-1)
                    bpntre.append(-1)
            curr_row = new_row
            if (len(bpntrb) > 0):
                bpntrb.append(len(bindx))
        if (len(bpntrb) == 0):
            bpntrb.append(0)
        bindx.append(dense_block%col_split)
    bpntre.append(len(bindx))
    while (len(bpntrb) < len(rpntr) -1):
        bpntrb.append(-1)
        bpntre.append(-1)
    filename = f"Matrix_{m}_{n}_{row_split}_{col_split}_{num_dense}_{perc_zeros}_{partitioning}"
    
    vbr_matrix = VBR(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)
    write_vbr_matrix(filename, vbr_matrix)
