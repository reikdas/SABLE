import pandas as pd

# Read all the CSV files
sable = pd.read_csv('results/res_1.csv')
psc = pd.read_csv('partially-strided-codelet/bench_executor_1thrds_SPMV.csv')
csr5 = pd.read_csv('Benchmark_SpMV_using_CSR5/CSR5_avx2/bench_1thrds.csv')
csr = pd.read_csv('results/csr-spmv-suitesparse_1thrd.csv')
mkl = pd.read_csv('results/mkl-spmv-suitesparse_1thrd.csv')

# Fix column names to be consistent
psc = psc.rename(columns={'Matrix': 'Matrix', 'Time(ns)': 'PSC_Time(ns)'})
csr5 = csr5.rename(columns={'Matrix': 'Matrix', 'Time(ns)': 'CSR5_Time(ns)'})
csr = csr.rename(columns={'Matrix': 'Matrix', 'Time(ns)': 'CSR_Time(ns)'})
mkl = mkl.rename(columns={'Matrix': 'Matrix', 'Time(ns)': 'MKL_Time(ns)'})
sable = sable.rename(columns={'Filename': 'Matrix', 'SABLE(ns)': 'SABLE(ns)'})

# Merge all dataframes
merged = sable.merge(psc, on='Matrix', how='inner')
merged = merged.merge(csr5, on='Matrix', how='inner')
merged = merged.merge(csr, on='Matrix', how='inner')
merged = merged.merge(mkl, on='Matrix', how='inner')

# Calculate speedups
merged['Speedup over PSC'] = merged['PSC_Time(ns)'] / merged['SABLE(ns)']
merged['Speedup over CSR5'] = merged['CSR5_Time(ns)'] / merged['SABLE(ns)']
merged['Speedup over CSR'] = merged['CSR_Time(ns)'] / merged['SABLE(ns)']
merged['Speedup over MKL'] = merged['MKL_Time(ns)'] / merged['SABLE(ns)']

# Select and reorder columns
final = merged[['Matrix', 'SABLE(ns)', 'Speedup over PSC', 'Speedup over CSR5', 'Speedup over CSR', 'Speedup over MKL']]

# Save to CSV
final.to_csv('single.csv', index=False)

print("Merged file 'single.csv' created successfully.")
