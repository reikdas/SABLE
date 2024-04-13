from matplotlib import pyplot as plt
import numpy as np  # Import numpy for evenly spaced sampling
import os
import pathlib

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..", "..")

plt.rcParams['figure.dpi'] = 300

ROWS = 10000
COLS = 10000

def extract_number(name_value_tuple):
    name = name_value_tuple[0]  # Get the name part of the tuple
    name_split = name.split("_")
    row_split = int(name_split[1])
    col_split = int(name_split[2])
    num_blocks = int(name_split[3])
    return (ROWS//row_split)*(COLS//col_split)*num_blocks

if __name__ == "__main__":
    threads = [1, 2, 4, 8, 16]
    thread_vars = {}
    for thread in threads:
        with open(f"{BASE_PATH}/SABLE/results/benchmarks_spmv_{thread}.csv", "r") as f:
            # thread_vars[f"{thread}_uniform_0"] = []
            # thread_vars[f"{thread}_uniform_0_names"] = []
            thread_vars[f"{thread}_nonuniform_0"] = []  # Initialize for non-uniform
            thread_vars[f"{thread}_nonuniform_0_names"] = []  # Initialize for non-uniform
            lines = f.readlines()
            for line in lines:
                parts = line.split(",")
                name = parts[0]
                t = sum(float(x) for x in parts[1:])/len(parts[1:])
                # if name.endswith("_0_uniform"):
                #     thread_vars[f"{thread}_uniform_0"].append(t)
                #     cleaned_name = name.replace("_0_uniform", "")
                #     cleaned_name = cleaned_name.replace(f"_{ROWS}_{COLS}", "")
                #     thread_vars[f"{thread}_uniform_0_names"].append(cleaned_name)
                if name.endswith("_0_nonuniform"):
                    thread_vars[f"{thread}_nonuniform_0"].append(t)
                    cleaned_name = name.replace("_0_nonuniform", "")
                    cleaned_name = cleaned_name.replace(f"_{ROWS}_{COLS}", "")
                    thread_vars[f"{thread}_nonuniform_0_names"].append(cleaned_name)

        # Sorting and storing uniform data
        # uniform_pairs = sorted(zip(thread_vars[f"{thread}_uniform_0_names"], thread_vars[f"{thread}_uniform_0"]), key=extract_number)
        # thread_vars[f"{thread}_uniform_0_names"], thread_vars[f"{thread}_uniform_0"] = zip(*uniform_pairs)

        # Sorting and storing non-uniform data
        non_uniform_pairs = sorted(zip(thread_vars[f"{thread}_nonuniform_0_names"], thread_vars[f"{thread}_nonuniform_0"]), key=extract_number)
        thread_vars[f"{thread}_nonuniform_0_names"], thread_vars[f"{thread}_nonuniform_0"] = zip(*non_uniform_pairs)

    # Normalize according to 1 thread for both uniform and non-uniform
    # norm_values_uniform = {thread: [thread_vars["1_uniform_0"][i]/thread_vars[f"{thread}_uniform_0"][i] for i in range(len(thread_vars["1_uniform_0"]))]
    #                for thread in threads if thread != 1}
    norm_values_nonuniform = {thread: [thread_vars["1_nonuniform_0"][i]/thread_vars[f"{thread}_nonuniform_0"][i] for i in range(len(thread_vars["1_nonuniform_0"]))]
                   for thread in threads if thread != 1}
    
    # Define the number of matrices to plot
    n = 6  # Example value
    
    # Assuming uniform and non-uniform have the same matrices and thus the same indices can be used
    total_matrices = len(thread_vars["1_nonuniform_0_names"])
    indices_to_sample = np.round(np.linspace(0, total_matrices - 1, n)).astype(int)
    sampled_matrix_names = [thread_vars["1_nonuniform_0_names"][i] for i in indices_to_sample]
    
    # Plot for uniform data
    # plt.figure()
    # for i, matrix_name in enumerate(sampled_matrix_names):
    #     y_axis = [norm_values_uniform[thread][indices_to_sample[i]] for thread in threads[1:]]  # Adjusted to use threads[1:]
    #     plt.plot(threads[1:], y_axis, label=matrix_name, marker='o')  # Adjusted for clarity

    # plt.xlabel('Number of Threads')
    # plt.ylabel('Normalized Performance (1 Thread Baseline)')
    # plt.title('Performance Scaling with Threads')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    # plt.subplots_adjust(bottom=0.3)
    # plt.savefig("images/parspmv_uniform.pdf")
    
    # Plot for non-uniform data
    plt.figure()
    for i, matrix_name in enumerate(sampled_matrix_names):
        y_axis = [norm_values_nonuniform[thread][indices_to_sample[i]] for thread in threads[1:]]  # Adjusted to use threads[1:]
        plt.plot(threads[1:], y_axis, label=matrix_name, marker='o')  # Adjusted for clarity

    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    # plt.title('Performance Scaling with Threads')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("parspmv_nonuniform.pdf")
