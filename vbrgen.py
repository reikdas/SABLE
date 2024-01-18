import random
import numpy
import sys
import os

def datagen(filename, x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre):
    dir_name = "Generated_Data"
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

def read_data(filename):
    with open(filename, "r") as f:
        x = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        val = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        indx = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bindx = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        rpntr = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        cpntr = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bpntrb = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
        bpntre = list(map(int, f.readline().split("=")[1][1:-2].split(",")))
    return x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre

def spmv_codegen():
    dir_name = "Generated_SpMV"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for filename in os.listdir("Generated_Data"):
        assert(filename.endswith(".data"))
        rel_path = os.path.join("Generated_Data", filename)
        x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_data(rel_path)
        filename = filename[:-5]
        with open(os.path.join(dir_name, filename+".c"), "w") as f:
            f.write("#include <stdio.h>\n")
            f.write("#include <sys/time.h>\n")
            f.write("#include <stdlib.h>\n")
            f.write("#include <assert.h>\n\n")
            f.write("int main() {\n")
            f.write(f"\tFILE *file = fopen(\"{os.path.abspath(rel_path)}\", \"r\");\n")
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
                if bpntrb[a] == -1:
                    continue
                valid_cols = bindx[bpntrb[a]:bpntre[a]]
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
            f.write("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
            f.write(f"\tfor (int i=0; i<{len(x)}; i++) {{\n")
            f.write("\t\tprintf(\"%d\\n\", y[i]);\n")
            f.write("\t}\n")
            f.write("}\n")

def write_mm_file(filename, M):
    with open(filename, 'w') as f:
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

def mtx_gen():
    dir_name = "Generated_Matrix"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for filename in os.listdir("Generated_Data"):
        assert(filename.endswith(".data"))
        x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_data(os.path.join("Generated_Data", filename))
        filename = filename[:-5]
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
        write_mm_file(os.path.join(dir_name, filename + ".mtx"), M)

def random_splits(n, a):
    if n <= 0 or a <= 0:
        raise ValueError("Both n and a must be positive integers.")

    # Generate a list of 'a-1' random numbers between 1 and n-1
    split_numbers = sorted(random.sample(range(1, n), a - 1))

    # Calculate the differences between consecutive split numbers
    differences = [split_numbers[0]] + [split_numbers[i] - split_numbers[i - 1] for i in range(1, a - 1)] + [n - split_numbers[-1]]

    return differences

def cumsum_list(l):
    l2 = [0]
    for elem in l:
        l2.append(l2[-1] + elem)
    return l2

def gen_data() -> None:
    num_row_splits = [50, 100]
    num_col_splits = [50, 100]
    rows = [5000] # Number of rows
    cols = [5000] # Number of columns
    for m in rows:
        for n in cols:
            for row_split in num_row_splits:
                for col_split in num_col_splits:
                    assert(m%row_split == 0)
                    assert(n%col_split == 0)
                    rpntr = cumsum_list(random_splits(m-1, row_split))
                    cpntr = cumsum_list(random_splits(n-1, col_split))
                    num_blocks = row_split * col_split
                    for num_dense in [(num_blocks*20)//100, (num_blocks*15)//100, (num_blocks*10)//100, (num_blocks*5)//100, (num_blocks*1)//100]:
                        # Randomly choose dense blocks
                        dense_blocks = random.sample([x for x in range(num_blocks)], num_dense)
                        dense_blocks.sort() # Easier handling of bpntrb/bpntre/bindx
                        for perc_zeros in [50, 40, 30, 20, 10, 0]:
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
                                zeros = random.sample([x for x in range(block_size)], (block_size * perc_zeros) // 100)
                                numzeros += len(zeros)
                                for index in range(block_size):
                                    if index in zeros:
                                        val.append(0)
                                    else:
                                        val.append(1)
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
                            # print("indx = ", indx)
                            # print("bindx = ", bindx)
                            # print("rpntr = ", rpntr)
                            # print("cpntr = ", cpntr)
                            # print("bpntrb = ", bpntrb)
                            # print("bpntre = ", bpntre)
                            # print("Dense blocks = ", dense_blocks)
                            filename = f"Matrix_{m}_{n}_{row_split}_{col_split}_{num_dense}_{perc_zeros}z"
                            datagen(filename, [1]*(m+col_split+1), val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)

if __name__ == "__main__":
    gen_data()
    spmv_codegen()
    mtx_gen()
