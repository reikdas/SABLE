import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pathlib
from matplotlib import rcParams

# Use Type 1 fonts (vector fonts) in PDF
rcParams.update({
    "pdf.fonttype": 42,       # Use TrueType fonts (better for compatibility)
    "ps.fonttype": 42,        # Same for PS
    "font.family": "serif",   # Use a serif font (or "sans-serif" if preferred)
    "font.serif": ["Times"],  # Or use "Palatino", "Georgia", etc.
    "text.usetex": True,     # Set to True if using LaTeX for text rendering
    "figure.dpi": 300         # High-res figures
})

# Set base path
FILEPATH = pathlib.Path(__file__).resolve().parent
base_path = os.path.join(FILEPATH)

# File paths
files = {
    "SABLE_1": "results/res_1.csv",
    "SABLE_2": "results/res_2.csv",
    "SABLE_4": "results/res_4.csv",
    "SABLE_8": "results/res_8.csv",
    "MKL": "results/mkl-spmv-merged-results.csv",
    "PSC": "partially-strided-codelet/bench_executor_SPMV_merged.csv",
    "CSR": "results/csr-spmv-merged-results.csv"
}

# Load data
dataframes = {
    label: pd.read_csv(os.path.join(base_path, fname)) for label, fname in files.items()
}

# Baseline from SABLE 1-thread
baseline = dataframes["SABLE_1"].set_index("Filename")["SABLE(ns)"]
sable_2 = dataframes["SABLE_2"].set_index("Filename")["SABLE(ns)"]
sable_4 = dataframes["SABLE_4"].set_index("Filename")["SABLE(ns)"]
sable_8 = dataframes["SABLE_8"].set_index("Filename")["SABLE(ns)"]

scalable_matrices = ["brainpc2", "bloweybl", "nd12k", "std1_Jac2"]

# Compute speedups
def compute_speedup(df, baseline, time_col="SABLE(ns)", name_col="Filename"):
    return baseline / df.set_index(name_col)[time_col]

speedups = {
    "SABLE_1": pd.Series(1.0, index=scalable_matrices),
    "SABLE_2": compute_speedup(dataframes["SABLE_2"], baseline).loc[scalable_matrices],
    "SABLE_4": compute_speedup(dataframes["SABLE_4"], baseline).loc[scalable_matrices],
    "SABLE_8": compute_speedup(dataframes["SABLE_8"], baseline).loc[scalable_matrices],
}

mkl_df = dataframes["MKL"].set_index("Matrix").loc[scalable_matrices]
psc_df = dataframes["PSC"].set_index("Matrix").loc[scalable_matrices]
csr_df = dataframes["CSR"].set_index("Matrix").loc[scalable_matrices]

for t in [1, 2, 4, 8]:
    speedups[f"MKL_{t}"] = baseline.loc[scalable_matrices] / mkl_df[f"{t}thread"]
    speedups[f"PSC_{t}"] = baseline.loc[scalable_matrices] / psc_df[f"{t} Threads"]
    speedups[f"CSR_{t}"] = baseline.loc[scalable_matrices] / csr_df[f"{t}thread"]

# Plot all in one row
pdf_path = os.path.join(base_path, "par.pdf")
with PdfPages(pdf_path) as pdf:
    num_matrices = len(scalable_matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(5 * num_matrices, 4), squeeze=False)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    threads = [1, 2, 4, 8]
    for idx, matrix in enumerate(scalable_matrices):
        ax = axes[0][idx]
        for impl in ["SABLE", "MKL", "PSC", "CSR"]:
            y = [speedups[f"{impl}_{t}"].get(matrix, float("nan")) for t in threads]
            ax.plot(threads, y, marker='o', label=impl)
        ax.axhline(y=1, color='gray', linestyle='--')
        ax.set_title(matrix, fontsize=20)
        ax.set_xlabel("Threads", fontsize=18)
        ax.set_ylabel("Speedup", fontsize=18)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1', '2', '3', '4', '5', '6', '7', '8'], fontsize=16)
        # set y tick labels fontsize to 16
        if (idx == 0):
            ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5], ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'], fontsize=16)
        elif (idx == 1):
            ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3], ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0'], fontsize=16)
        elif (idx == 2):
            ax.set_yticks([1, 2, 3, 4, 5, 6, 7], ['1', '2', '3', '4', '5', '6', '7'], fontsize=16)
        elif (idx == 3):
            ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=16)
        ax.grid(True)
        if idx == 0:
            ax.legend(fontsize=14)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"Saved horizontal row of plots to: {pdf_path}")
