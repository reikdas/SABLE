import re

BASE_PATH="/home/das160"

def extract_number(name_value_tuple):
    name = name_value_tuple[0]  # Get the name part of the tuple
    return int(name.split('_')[-3])  # Extract the number preceding "_0_"

def draw_executor():
    threads = [1, 16]
    thread_vars_mine = {}
    # Read mine
    for thread in threads:
        with open(f"{BASE_PATH}/VBR-SpMV/results/benchmarks_spmm_{thread}.csv", "r") as f:
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
        with open(f"{BASE_PATH}/sparse-register-tiling/results/bench_executor_{thread}thrds.csv", "r") as f:
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

    for formity in ["uniform", "nonuniform"]:
        for i, matrix_name in enumerate(thread_vars_mine[f"mine_1_{formity}_0_names"]):
            print(matrix_name + " & ", end="")
            for zero in [0, 20, 50]:
                val = thread_vars_psc[f"psc_1_{formity}_{zero}"][i]/1000
                print(f"{val:.2f}" + " & ", end="")
            for zero in [0, 20, 50]:
                val = thread_vars_psc[f"psc_1_{formity}_{zero}"][i]/thread_vars_mine[f"mine_1_{formity}_{zero}"][i]
                print(f"{val:.2f}" + "x & ", end="")
            for zero in [0, 20, 50]:
                val = thread_vars_psc[f"psc_1_{formity}_{zero}"][i]/thread_vars_psc[f"psc_16_{formity}_{zero}"][i]
                print(f"{val:.2f}" + "x & ", end="")
            for zero in [0, 20, 50]:
                val = thread_vars_psc[f"psc_1_{formity}_{zero}"][i]/thread_vars_mine[f"mine_16_{formity}_{zero}"][i]
                print(f"{val:.2f}" + "x & ", end="")
            print()

if __name__ == "__main__":
    draw_executor()
