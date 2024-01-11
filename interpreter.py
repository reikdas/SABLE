from vbrgen import read_data

def interpret(filename):
    x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_data(filename)
    y = [0] * len(x)
    count1 = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                count2 = 0
                for j in range(cpntr[b], cpntr[b+1]):
                    for i in range(rpntr[a], rpntr[a+1]):
                        y[i] += val[indx[count1] + count2] * x[j]
                        count2 += 1
                count1 += 1
    return y
