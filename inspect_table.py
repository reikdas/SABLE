import pandas as pd

# Load SABLE inspection times
sable_df = pd.read_csv("results/suitesparse_inspect.csv")

# Load partially-strided codelet (LCM I/E) times
psc_df = pd.read_csv("partially-strided-codelet/bench_inspector_1thrds_SPMV.csv")

# Convert Time(ns) to ms
psc_df["Time(ms)"] = psc_df["Time(ns)"] / 1e6

# Drop rows where Matrix name ends with '_db'
psc_df = psc_df[~psc_df['Matrix'].str.endswith('_db')]

# Merge the dataframes
merged = sable_df.merge(
    psc_df, how='inner', left_on='Filename', right_on='Matrix'
)

# Prepare final DataFrame
final = pd.DataFrame({
    "Matrix": merged["Filename"],
    "CSR to VBR-C": merged["Codegen(ms)"],
    "Generate binary": merged["Compile(ms)"],
    "LCM I/E": (merged["Time(ms)"]).round(2)
})

# Sort if you want, e.g., by Matrix name
final = final.sort_values("Matrix")

# Generate LaTeX table
with open("inspect_table.tex", "w") as f:
    f.write("\\begin{table}\n")
    f.write("    \\caption{Inspection time breakdown in milliseconds (ms)}\n")
    f.write("    \\label{table:inspector}\n")
    f.write("    \\scriptsize\n")
    f.write("    \\begin{tabular}{lrrr}\n")
    f.write("    \\toprule\n")
    f.write("    \\textbf{Matrix} & \\multicolumn{2}{c}{\\textbf{SABLE}} & \\textbf{LCM I/E} \\\n")
    f.write("    \\cmidrule(lr){2-3}\n")
    f.write("           & \\textbf{CSR to VBR-C} & \\textbf{Generate binary} & \\\n")
    f.write("    \\midrule\n")

    for _, row in final.iterrows():
        matrix_name = row['Matrix'].replace('_', '\\_')
        f.write(f"    {matrix_name} & {int(row['CSR to VBR-C'])} & {int(row['Generate binary'])} & {row['LCM I/E']:.2f} \\\n")

    f.write("    \\bottomrule\n")
    f.write("    \\end{tabular}\n")
    f.write("\\end{table}\n")
