import random
import numpy
import sys

def codegen(x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre):
    print("#include <stdio.h>\n")
    print("int main() {")
    print(f"\tint y[{len(x)}] = {{0}};")
    print(f"\tint x[] = {{{','.join(map(str, x))}}};")
    print(f"\tint val[] = {{{','.join(map(str, val))}}};")
    print(f"\tint indx[] = {{{','.join(map(str, indx))}}};")
    print(f"\tint bindx[] = {{{','.join(map(str, bindx))}}};")
    print(f"\tint rpntr[] = {{{','.join(map(str, rpntr))}}};")
    print(f"\tint cpntr[] = {{{','.join(map(str, cpntr))}}};")
    print(f"\tint bpntrb[] = {{{','.join(map(str, bpntrb))}}};")
    print(f"\tint bpntre[] = {{{','.join(map(str, bpntre))}}};")
    print("\tint count = 0;\n")
    count = 0
    for a in range(len(rpntr)-1):
        valid_cols = bindx[bpntrb[a]-bpntrb[0]:bpntre[a]-bpntrb[0]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                print("\tcount = 0;")
                print(f"\tfor (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{")
                print(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{")
                print(f"\t\t\ty[i] += val[{indx[count]}+count] * x[j];")
                print("\t\t\tcount++;")
                print("\t\t}")
                print("\t}")
                count+=1
    print(f"\tfor (int i=0; i<{len(x)}; i++) {{")
    print("\t\tprintf(\"%d \", y[i]);")
    print("\t}\n")
    print("\tprintf(\"\\n\");\n")
    print("}\n")

def find_nonneg(l):
    for i, ele in enumerate(l):
        if ele != -1:
            return ele
    assert(False)
    return -1

def gen_matrix(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre):
    M = numpy.zeros((rpntr[-1], cpntr[-1]))
    count = 0
    bpntrb_start = find_nonneg(bpntrb)
    for a in range(len(rpntr) - 1):
        valid_cols = bindx[bpntrb[a]-bpntrb_start:bpntre[a]-bpntrb_start]
        for b in range(len(cpntr) - 1):
            if b in valid_cols:
                count2 = 0
                for j in range(cpntr[b], cpntr[b+1]):
                    for i in range(rpntr[a], rpntr[a+1]):
                        M[i][j] = val[indx[count]+count2]
                        count2 += 1
                count += 1
    print(M)

def gen_random():
    row_widths = [1, 2, 3] # Rows are split equally into row_widths
    col_widths = [1, 2, 3] # Cols are split equally into col_widths
    val = [1]*1000 # Adjust for larger m and n
    m = 30 # Number of rows
    n = 30 # Number of columns
    for row_width in row_widths:
        for col_width in col_widths:
            rpntr = [x for x in range(0, m+row_width, row_width)]
            cpntr = [x for x in range(0, n+col_width, col_width)]
            assert(m%row_width == 0)
            assert(n%col_width == 0)
            blocks_in_row = m//row_width
            blocks_in_col = n//col_width
            num_blocks = blocks_in_row * blocks_in_col
            for num_dense in range(1, num_blocks//5): # Only 20% of blocks can be dense
                # Randomly choose dense blocks
                dense_blocks = random.choices([x for x in range(num_blocks)], k=num_dense)
                dense_blocks.sort() # Easier handling of bpntrb/bpntre/bindx
                bindx = []
                bpntrb = []
                bpntre = []
                curr_row = 0
                for dense_block in dense_blocks:
                    new_row = dense_block//blocks_in_col
                    if new_row != curr_row:
                        if curr_row == 0 and len(bpntrb) == 0:
                            for i in range(curr_row, new_row):
                                bpntrb.append(-1)
                                bpntre.append(-1)
                        else:
                            if (len(bpntrb) > 0):
                                bpntre.append(len(bindx))
                            for i in range(curr_row+1, new_row):
                                bpntrb.append(-1)
                                bpntre.append(-1)
                        curr_row = new_row
                        if (len(bpntrb) > 0):
                            bpntrb.append(len(bindx))
                    if (len(bpntrb) == 0):
                        bpntrb.append(0)
                    bindx.append(dense_block%blocks_in_col)
                bpntre.append(len(bindx))
                while (len(bpntrb) < len(rpntr) -1):
                    bpntrb.append(-1)
                    bpntre.append(-1)
                indx = [0]
                for i in range(num_dense):
                    indx.append((i+1)*(row_width * col_width))
                print("indx = ", indx)
                print("bindx = ", bindx)
                print("rpntr = ", rpntr)
                print("cpntr = ", cpntr)
                print("bpntrb = ", bpntrb)
                print("bpntre = ", bpntre)
                print("Dense blocks = ", dense_blocks)
                gen_matrix(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)
                print("--------------------")
                
def sparskit_mat():
    """
    [[4 2 0 0 0 1 0 0 0 -1 1]
    [1 5 0 0 0 2 0 0 0 0 -1]
    [0 0 6 1 2 2 0 0 0 0 0]
    [0 0 2 7 1 0 0 0 0 0 0]
    [0 0 -1 2 9 3 0 0 0 0 0]
    [2 1 3 4 5 10 4 3 2 0 0]
    [0 0 0 0 0 4 13 4 2 0 0]
    [0 0 0 0 0 3 3 11 3 0 0]
    [0 0 0 0 0 0 2 0 7 0 0]
    [8 4 0 0 0 0 0 0 0 25 3]
    [-2 3 0 0 0 0 0 0 0 8 12]]
    """
    val = [4, 1, 2, 5, 1, 2, -1, 0, 1, -1, 6, 2, -1, 1, 7, 2, 2, 1, 9, 2, 0, 3, 2, 1, 3, 4, 5, 10, 4, 3, 2, 4, 3, 0, 13, 3, 2, 4, 11, 0, 2, 3, 7, 8, -2, 4, 3, 25, 8, 3, 12]
    indx = [0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51]
    bindx = [0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]
    rpntr = [0, 2, 5, 6, 9, 11]
    cpntr = [0, 2, 5, 6, 9, 11]
    bpntrb = [0, 3, 5, 9, 11]
    bpntre = [3, 5, 9, 11, 13]
    # codegen([1]*11, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)
    gen_matrix(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)

if __name__ == "__main__":
    # sparskit_mat()
    gen_random()
