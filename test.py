def codegen(x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre):
    print(f"""
x = {x}
y = [0] * {len(x)}
val = {val}
indx = {indx}
bindx = {bindx}
rpntr = {rpntr}
cpntr = {cpntr}
bpntrb = {bpntrb}
bpntre = {bpntre}
    """)
    count = 0
    for a in range(len(rpntr)-1):
        valid_cols = bindx[bpntrb[a]-bpntrb[0]:bpntre[a]-bpntrb[0]]
        for b in range(len(cpntr)-1):
            if (b+1) in valid_cols:
                print("count2 = 0")
                print(f"for j in range({cpntr[b]-1}, {cpntr[b+1]-1}):")
                print(f"\tfor i in range({rpntr[a]-1}, {rpntr[a+1]-1}):")
                print(f"\t\ty[i] += val[{indx[count]-1}+count2] * x[j]")
                print("\t\tcount2+=1")
                count+=1
    print("print(y)")


if __name__ == "__main__":
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
    codegen([1]*11, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre)
