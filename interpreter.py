from vbrgen import read_data
from functools import partial

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

                # def spMVDot(denseBlock):
                #     def vectorDot(i):
                #         ewise_product: [cpntr[b+1] - cpntr[b]] = build(cpntr[b], cpntr[b+1], lambda j: val[indx[denseBlock]+(j-cpntr[b])*rpntr[a+1]+i] * x[j])
                #         res: float = fold(len(ewise_product), lambda carry, idx: carry + ewise_product[idx], init_carry=0)
                #         return res

                #     return build(rpntr[a], rpntr[a+1], vectorDot) # y

                # build(0, len(indx), spMVDot)
    return y
