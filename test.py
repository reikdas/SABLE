import random
import numpy
import sys
import os

def codegen(filename, x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre):
    dir_name = "Generated_SpMV"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, filename+".data"), "w") as f:
        f.write(f"x=[{','.join(map(str, x))}]\n")
        f.write(f"val=[{','.join(map(str, val))}]\n")
        f.write(f"indx=[{','.join(map(str, indx))}]\n")
        f.write(f"bindx=[{','.join(map(str, bindx))}]\n")
        f.write(f"rpntr=[{','.join(map(str, rpntr))}]\n")
        f.write(f"cpntr=[{','.join(map(str, cpntr))}]\n")
        f.write(f"bpntrb=[{','.join(map(str, bpntrb))}]\n")
        f.write(f"bpntre=[{','.join(map(str, bpntre))}]\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.write("#include <stdio.h>\n")
        f.write("#include <sys/time.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <assert.h>\n\n")
        f.write("int main() {\n")
        f.write(f"\tFILE *file = fopen(\"{filename}.data\", \"r\");\n")
        f.write("\tif (file == NULL) { printf(\"Error opening file\"); return 1; }\n")
        f.write(f"\tint* y = (int*)calloc({len(x)}, sizeof(int));\n")
        f.write(f"\tint* x = (int*)calloc({len(x) + 1}, sizeof(int));\n")
        f.write(f"\tint* val = (int*)calloc({len(val) + 1}, sizeof(int));\n")
        f.write(f"\tint* indx = (int*)calloc({len(indx) + 1}, sizeof(int));\n")
        f.write(f"\tint* bindx = (int*)calloc({len(bindx) + 1}, sizeof(int));\n")
        f.write(f"\tint* rpntr = (int*)calloc({len(rpntr) + 1}, sizeof(int));\n")
        f.write(f"\tint* cpntr = (int*)calloc({len(cpntr) + 1}, sizeof(int));\n")
        f.write(f"\tint* bpntrb = (int*)calloc({len(bpntrb) + 1}, sizeof(int));\n")
        f.write(f"\tint* bpntre = (int*)calloc({len(bpntre) + 1}, sizeof(int));\n")
        f.write("\tchar c;\n")
        f.write(f"\tint x_size=0, val_size=0, indx_size=0, bindx_size=0, rpntr_size=0, cpntr_size=0, bpntrb_size=0, bpntre_size=0;\n")
        for variable in ["x", "val", "indx", "bindx", "rpntr", "cpntr", "bpntrb", "bpntre"]:
            f.write('''
    assert(fscanf(file, "{0}=[%d", &{0}[{0}_size]) == 1);
    {0}_size++;
    while (1) {{
        assert(fscanf(file, \"%c\", &c) == 1);
        if (c == ',') {{
            assert(fscanf(file, \"%d\", &{0}[{0}_size]) == 1);
            {0}_size++;
        }} else if (c == ']') {{
            break;
        }} else {{
            assert(0);
        }}
    }}
    fscanf(file, \"%c\", &c);
    assert(c=='\\n');\n'''.format(variable))
        f.write("\tfclose(file);\n")
        f.write("\tint count = 0;\n")
        f.write("\tstruct timeval t1;\n")
        f.write("\tgettimeofday(&t1, NULL);\n")
        f.write("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
        count = 0
        for a in range(len(rpntr)-1):
            valid_cols = bindx[bpntrb[a]-bpntrb[0]:bpntre[a]-bpntrb[0]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    f.write("\tcount = 0;\n")
                    f.write(f"\tfor (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{\n")
                    f.write(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{\n")
                    f.write(f"\t\t\ty[i] += val[{indx[count]}+count] * x[j];\n")
                    f.write("\t\t\tcount++;\n")
                    f.write("\t\t}\n")
                    f.write("\t}\n")
                    count+=1
        f.write("\tstruct timeval t2;\n")
        f.write("\tgettimeofday(&t2, NULL);\n")
        f.write("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
        f.write(f"\tfor (int i=0; i<{len(x)}; i++) {{\n")
        f.write("\t\tprintf(\"%d \", y[i]);\n")
        f.write("\t}\n")
        f.write("\tprintf(\"\\n\");\n")
        f.write("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
        f.write("}\n")

def write_mm_file(filename, M):
    dir_name = "Generated_Matrices"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, filename), 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{M.shape[0]} {M.shape[1]} {numpy.count_nonzero(M)}\n")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i][j] != 0:
                    f.write(f"{i+1} {j+1} {M[i][j]}\n")

def find_nonneg(l):
    for _, ele in enumerate(l):
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
    # print(M)
    return M

def gen_random() -> None:
    row_widths = [50, 100] # Rows are split equally into row_widths
    col_widths = [50, 100] # Cols are split equally into col_widths
    m = 500 # Number of rows
    n = 500 # Number of columns
    val = []
    indx = [0]
    for row_width in row_widths:
        for col_width in col_widths:
            rpntr = [x for x in range(0, m+row_width, row_width)]
            cpntr = [x for x in range(0, n+col_width, col_width)]
            assert(m%row_width == 0)
            assert(n%col_width == 0)
            blocks_in_row = m//row_width
            blocks_in_col = n//col_width
            num_blocks = blocks_in_row * blocks_in_col
            #for num_dense in range(1, num_blocks//5): # Only 20% of blocks can be dense
            for num_dense in [num_blocks//5]:
                # Randomly choose dense blocks
                dense_blocks = random.sample([x for x in range(num_blocks)], num_dense)
                dense_blocks.sort() # Easier handling of bpntrb/bpntre/bindx
                bindx = []
                bpntrb = []
                bpntre = []
                curr_row = 0
                nzeros = 0
                for dense_block in dense_blocks:
                    new_row = dense_block//blocks_in_col
                    col_idx = dense_block//blocks_in_row
                    block_size = (rpntr[new_row+1] - rpntr[new_row]) * (cpntr[col_idx+1] - cpntr[col_idx])
                    zeros = random.sample([x for x in range(block_size)], k=block_size//5) # Only 20% of elements can be zero
                    nzeros += (len(zeros))
                    for index in range(block_size):
                        if index in zeros:
                            val.append(0)
                        else:
                            val.append(1)
                    indx.append(indx[-1] + block_size)
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
                # print("indx = ", indx)
                # print("bindx = ", bindx)
                # print("rpntr = ", rpntr)
                # print("cpntr = ", cpntr)
                # print("bpntrb = ", bpntrb)
                # print("bpntre = ", bpntre)
                # print("Dense blocks = ", dense_blocks)
                filename = f"Matrix_{row_width}_{col_width}_{num_dense}_{nzeros}z"
                codegen(filename, [1]*(m+col_width+1), val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)
                M = gen_matrix(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)
                # print(M)
                # print("--------------------")
                write_mm_file(filename + ".mtx", M)

# def numpy_to_csr(matrix):
#     # Convert the list of lists to a CSR matrix
#     csr_matrix_representation = csr_matrix(matrix)
#     return csr_matrix_representation.tocsr()
                
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
    codegen("foo.c", [1]*11, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)
    # gen_matrix(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)

if __name__ == "__main__":
    # sparskit_mat()
    gen_random()
