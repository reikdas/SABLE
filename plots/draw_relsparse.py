import matplotlib.pyplot as plt
import os
import pathlib

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..", "..")

plt.rcParams['figure.dpi'] = 300

def plot_relmatrix(matrix_name):
    zerosl = [0, 10, 20, 30, 40, 50, 75]
    names = []
    # t_mine_uniform = []
    # t_mine_uniform_bad = []
    t_mine_nonuniform = []
    t_mine_nonuniform_bad = []
    # t_psc_uniform = []
    t_psc_nonuniform = []
    for zeros in zerosl:
        names.append(matrix_name + "_" + str(zeros)+"_200")
    for name in names:
        with open(f"{BASE_PATH}/SABLE/results/benchmarks_spmv_1_rel.csv", "r") as f:
            for line in f:
                parts = line.split(",")
                # if parts[0] == name+"_uniform":
                #     t_mine_uniform.append(sum(float(x) for x in parts[1:])/len(parts[1:]))
                if parts[0] == name+"_nonuniform":
                    t_mine_nonuniform.append(sum(float(x) for x in parts[1:])/len(parts[1:]))
        with open(f"{BASE_PATH}/SABLE/results/benchmarks_spmv_1_rel_bad.csv", "r") as f:
            for line in f:
                parts = line.split(",")
                # if parts[0] == name+"_uniform":
                #     t_mine_uniform_bad.append(sum(float(x) for x in parts[1:])/len(parts[1:]))
                if parts[0] == name+"_nonuniform":
                    t_mine_nonuniform_bad.append(sum(float(x) for x in parts[1:])/len(parts[1:]))
        with open(f"{BASE_PATH}/SABLE/partially-strided-codelet/bench_executor_1thrds_rel.csv", "r") as f:
            for line in f:
                parts = line.split(",")
                # if parts[0] == name+"_uniform":
                #     t_psc_uniform.append(sum(float(x) for x in parts[1:])/len(parts[1:]))
                if parts[0] == name+"_nonuniform":
                    t_psc_nonuniform.append(sum(float(x) for x in parts[1:])/len(parts[1:]))
    speedup_nonuniform = [t_psc_nonuniform[i]/t_mine_nonuniform[i] for i in range(len(t_mine_nonuniform))]
    speedup_nonuniform_bad = [t_psc_nonuniform[i]/t_mine_nonuniform_bad[i] for i in range(len(t_mine_nonuniform_bad))]
    plt.plot(zerosl, speedup_nonuniform_bad, label="SABLE vs PSC")
    plt.plot(zerosl, speedup_nonuniform, label="SABLE vs PSC (Unrolled loops for sparse blocks)")
    plt.xlabel("Percentage of zeros per dense block", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    plt.subplots_adjust(bottom=0.23)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.axhline(y=1, color='b', linestyle='--')
    plt.savefig("relsparse_spmv.pdf")
                    
if __name__ == "__main__":
    plot_relmatrix("Matrix_10000_10000_50_50_300")
