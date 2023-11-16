import random
import numpy

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
            if (b+1) in valid_cols:
                print("\tcount = 0;")
                print(f"\tfor (int j = {cpntr[b]-1}; j < {cpntr[b+1]-1}; j++) {{")
                print(f"\t\tfor (int i = {rpntr[a]-1}; i < {rpntr[a+1]-1}; i++) {{")
                print(f"\t\t\ty[i] += val[{indx[count]-1}+count] * x[j];")
                print("\t\t\tcount++;")
                print("\t\t}")
                print("\t}")
                count+=1
    print(f"\tfor (int i=0; i<{len(x)}; i++) {{")
    print("\t\tprintf(\"%d \", y[i]);")
    print("\t}\n")
    print("\tprintf(\"\\n\");\n")
    print("}\n")

def gen_matrix(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre):
    M = numpy.zeros((rpntr[-1]-1, cpntr[-1]-1))
    count = 0
    for a in range(len(rpntr) - 1):
        valid_cols = bindx[bpntrb[a]-bpntrb[0]:bpntre[a]-bpntrb[0]]
        for b in range(len(cpntr) - 1):
            if (b+1) in valid_cols:
                count2 = 0
                for j in range(cpntr[b]-1, cpntr[b+1]-1):
                    for i in range(rpntr[a]-1, rpntr[a+1]-1):
                        M[i][j] = val[indx[count]-1+count2]
                        count2 += 1
                count += 1
    print(M)

# def gen_random():
#     row_widths = [1]
#     col_widths = [1]
#     val = [1]*100
#     m = 10
#     n = 10
#     for row_width in row_widths:
#         for col_width in col_widths:
#             rpntr = [x for x in range(0, m, row_width)]
#             cpntr = [x for x in range(0, n, col_width)]
#             assert((m*n)%(row_width*col_width) == 0)
#             num_blocks = (m*n)//(row_width*col_width)
#             for num_dense in range(1, num_blocks):
#                 dense_blocks = random.choices([x for x in range(num_blocks)], k=num_dense)
#                 dense_blocks.sort()
#                 bindx = []
#                 bpntrb = []
#                 bpntre = []
#                 for dense_block in dense_blocks:
#                     bindx.append(dense_block%m)
                
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
    indx = [1, 5, 7, 11, 20, 23, 25, 28, 29, 32, 35, 44, 48, 52]
    bindx = [1, 3, 5, 2, 3, 1, 2, 3, 4, 3, 4, 1, 5]
    rpntr = [1, 3, 6, 7, 10, 12]
    cpntr = [1, 3, 6, 7, 10, 12]
    bpntrb = [1, 4, 6, 10, 12]
    bpntre = [4, 6, 10, 12, 14]
    # codegen([1]*11, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)
    gen_matrix(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)

if __name__ == "__main__":
    sparskit_mat()
