import re
import pathlib

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

def extract_number(name_value_tuple):
    name = name_value_tuple[0]  # Get the name part of the tuple
    return int(name.split('_')[-3])  # Extract the number preceding "_0_"

def gen_table(op):
    if op == "spmm":
        repo = "sparse-register-tiling/results"
        baseline = "SpReg"
    elif op == "spmv":
        repo = "partially-strided-codelet"
        baseline = "PSC"
    else:
        raise Exception("Unknown operation")
    threads = [1, 16]
    thread_vars_mine = {}
    # Read mine
    for thread in threads:
        with open(f"{BASE_PATH}/SABLE/results/benchmarks_{op}_{thread}.csv", "r") as f:
            thread_vars_mine[f"mine_{thread}_uniform_0"] = []
            thread_vars_mine[f"mine_{thread}_uniform_20"] = []
            thread_vars_mine[f"mine_{thread}_uniform_50"] = []
            thread_vars_mine[f"mine_{thread}_nonuniform_0"] = []
            thread_vars_mine[f"mine_{thread}_nonuniform_20"] = []
            thread_vars_mine[f"mine_{thread}_nonuniform_50"] = []
            thread_vars_mine[f"mine_{thread}_uniform_0_names"] = []
            thread_vars_mine[f"mine_{thread}_uniform_20_names"] = []
            thread_vars_mine[f"mine_{thread}_uniform_50_names"] = []
            thread_vars_mine[f"mine_{thread}_nonuniform_0_names"] = []
            thread_vars_mine[f"mine_{thread}_nonuniform_20_names"] = []
            thread_vars_mine[f"mine_{thread}_nonuniform_50_names"] = []
            lines = f.readlines()
            for line in lines:
                parts = line.split(",")
                name = parts[0]
                t = sum(float(x) for x in parts[1:])/len(parts[1:])
                if name.endswith("_0_uniform"):
                    thread_vars_mine[f"mine_{thread}_uniform_0"].append(t)
                    thread_vars_mine[f"mine_{thread}_uniform_0_names"].append(name)
                elif name.endswith("_20_uniform"):
                    thread_vars_mine[f"mine_{thread}_uniform_20"].append(t)
                    thread_vars_mine[f"mine_{thread}_uniform_20_names"].append(name)
                elif name.endswith("_50_uniform"):
                    thread_vars_mine[f"mine_{thread}_uniform_50"].append(t)
                    thread_vars_mine[f"mine_{thread}_uniform_50_names"].append(name)
                if name.endswith("_0_nonuniform"):
                    thread_vars_mine[f"mine_{thread}_nonuniform_0"].append(t)
                    thread_vars_mine[f"mine_{thread}_nonuniform_0_names"].append(name)
                elif name.endswith("_20_nonuniform"):
                    thread_vars_mine[f"mine_{thread}_nonuniform_20"].append(t)
                    thread_vars_mine[f"mine_{thread}_nonuniform_20_names"].append(name)
                elif name.endswith("_50_nonuniform"):
                    thread_vars_mine[f"mine_{thread}_nonuniform_50"].append(t)
                    thread_vars_mine[f"mine_{thread}_nonuniform_50_names"].append(name)
    # Read PSC
    thread_vars_psc = {}
    for thread in threads:
        with open(f"{BASE_PATH}/{repo}/bench_executor_{thread}thrds.csv", "r") as f:
            thread_vars_psc[f"psc_{thread}_uniform_0"] = []
            thread_vars_psc[f"psc_{thread}_uniform_20"] = []
            thread_vars_psc[f"psc_{thread}_uniform_50"] = []
            thread_vars_psc[f"psc_{thread}_nonuniform_0"] = []
            thread_vars_psc[f"psc_{thread}_nonuniform_20"] = []
            thread_vars_psc[f"psc_{thread}_nonuniform_50"] = []
            thread_vars_psc[f"psc_{thread}_uniform_0_names"] = []
            thread_vars_psc[f"psc_{thread}_uniform_20_names"] = []
            thread_vars_psc[f"psc_{thread}_uniform_50_names"] = []
            thread_vars_psc[f"psc_{thread}_nonuniform_0_names"] = []
            thread_vars_psc[f"psc_{thread}_nonuniform_20_names"] = []
            thread_vars_psc[f"psc_{thread}_nonuniform_50_names"] = []
            lines = f.readlines()
            for line in lines:
                parts = line.split(",")
                name = parts[0]
                t = sum(float(x) for x in parts[1:])/len(parts[1:])
                if name.endswith("_0_uniform"):
                    thread_vars_psc[f"psc_{thread}_uniform_0"].append(t)
                    thread_vars_psc[f"psc_{thread}_uniform_0_names"].append(name)
                elif name.endswith("_20_uniform"):
                    thread_vars_psc[f"psc_{thread}_uniform_20"].append(t)
                    thread_vars_psc[f"psc_{thread}_uniform_20_names"].append(name)
                elif name.endswith("_50_uniform"):
                    thread_vars_psc[f"psc_{thread}_uniform_50"].append(t)
                    thread_vars_psc[f"psc_{thread}_uniform_50_names"].append(name)
                elif name.endswith("_0_nonuniform"):
                    thread_vars_psc[f"psc_{thread}_nonuniform_0"].append(t)
                    thread_vars_psc[f"psc_{thread}_nonuniform_0_names"].append(name)
                elif name.endswith("_20_nonuniform"):
                    thread_vars_psc[f"psc_{thread}_nonuniform_20"].append(t)
                    thread_vars_psc[f"psc_{thread}_nonuniform_20_names"].append(name)
                elif name.endswith("_50_nonuniform"):
                    thread_vars_psc[f"psc_{thread}_nonuniform_50"].append(t)
                    thread_vars_psc[f"psc_{thread}_nonuniform_50_names"].append(name)

    for zero in [0, 20, 50]:
        mine_1_uniform_pairs = sorted(zip(thread_vars_mine[f"mine_1_uniform_{zero}_names"], thread_vars_mine[f"mine_1_uniform_{zero}"]), key=extract_number)
        thread_vars_mine[f"mine_1_uniform_{zero}_names"], thread_vars_mine[f"mine_1_uniform_{zero}"] = zip(*mine_1_uniform_pairs)
        mine_16_uniform_pairs = sorted(zip(thread_vars_mine[f"mine_16_uniform_{zero}_names"], thread_vars_mine[f"mine_16_uniform_{zero}"]), key=extract_number)
        thread_vars_mine[f"mine_16_uniform_{zero}_names"], thread_vars_mine[f"mine_16_uniform_{zero}"] = zip(*mine_16_uniform_pairs)
        psc_1_uniform_pairs = sorted(zip(thread_vars_psc[f"psc_1_uniform_{zero}_names"], thread_vars_psc[f"psc_1_uniform_{zero}"]), key=extract_number)
        thread_vars_psc[f"psc_1_uniform_{zero}_names"], thread_vars_psc[f"psc_1_uniform_{zero}"] = zip(*psc_1_uniform_pairs)
        psc_16_uniform_pairs = sorted(zip(thread_vars_psc[f"psc_16_uniform_{zero}_names"], thread_vars_psc[f"psc_16_uniform_{zero}"]), key=extract_number)
        thread_vars_psc[f"psc_16_uniform_{zero}_names"], thread_vars_psc[f"psc_16_uniform_{zero}"] = zip(*psc_16_uniform_pairs)
        mine_1_nonuniform_pairs = sorted(zip(thread_vars_mine[f"mine_1_nonuniform_{zero}_names"], thread_vars_mine[f"mine_1_nonuniform_{zero}"]), key=extract_number)
        thread_vars_mine[f"mine_1_nonuniform_{zero}_names"], thread_vars_mine[f"mine_1_nonuniform_{zero}"] = zip(*mine_1_nonuniform_pairs)
        mine_16_nonuniform_pairs = sorted(zip(thread_vars_mine[f"mine_16_nonuniform_{zero}_names"], thread_vars_mine[f"mine_16_nonuniform_{zero}"]), key=extract_number)
        thread_vars_mine[f"mine_16_nonuniform_{zero}_names"], thread_vars_mine[f"mine_16_nonuniform_{zero}"] = zip(*mine_16_nonuniform_pairs)
        psc_1_nonuniform_pairs = sorted(zip(thread_vars_psc[f"psc_1_nonuniform_{zero}_names"], thread_vars_psc[f"psc_1_nonuniform_{zero}"]), key=extract_number)
        thread_vars_psc[f"psc_1_nonuniform_{zero}_names"], thread_vars_psc[f"psc_1_nonuniform_{zero}"] = zip(*psc_1_nonuniform_pairs)
        psc_16_nonuniform_pairs = sorted(zip(thread_vars_psc[f"psc_16_nonuniform_{zero}_names"], thread_vars_psc[f"psc_16_nonuniform_{zero}"]), key=extract_number)
        thread_vars_psc[f"psc_16_nonuniform_{zero}_names"], thread_vars_psc[f"psc_16_nonuniform_{zero}"] = zip(*psc_16_nonuniform_pairs)

    with open(f"{op}_speedup_table.txt", "w") as f:
        f.write(f"Matrix & {baseline} 1 Thread (0% & 20% & 50%) & SABLE 1 Thread (0% & 20% & 50%) & {baseline} 16 Threads (0% & 20% & 50%) & SABLE 16 Threads (0% & 20% & 50%)\n")
        for formity in ["uniform", "nonuniform"]:
            for i, matrix_name in enumerate(thread_vars_mine[f"mine_1_{formity}_0_names"]):
                f.write(matrix_name + " & ")
                for zero in [0, 20, 50]:
                    val = thread_vars_psc[f"psc_1_{formity}_{zero}"][i]/1000
                    f.write(f"{val:.2f}" + " & ")
                for zero in [0, 20, 50]:
                    val = thread_vars_psc[f"psc_1_{formity}_{zero}"][i]/thread_vars_mine[f"mine_1_{formity}_{zero}"][i]
                    f.write(f"{val:.2f}" + "x & ")
                for zero in [0, 20, 50]:
                    val = thread_vars_psc[f"psc_1_{formity}_{zero}"][i]/thread_vars_psc[f"psc_16_{formity}_{zero}"][i]
                    f.write(f"{val:.2f}" + "x & ")
                for j, zero in enumerate([0, 20, 50]):
                    val = thread_vars_psc[f"psc_1_{formity}_{zero}"][i]/thread_vars_mine[f"mine_16_{formity}_{zero}"][i]
                    if j == 2:
                        f.write(f"{val:.2f}" + "x")
                    else:
                        f.write(f"{val:.2f}" + "x & ")
                f.write("\n")

if __name__ == "__main__":
    gen_table("spmv")
    gen_table("spmm")
