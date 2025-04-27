import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
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

# Load the CSV file
file_path = "single.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Sort by matrix name
df = df.sort_values("Matrix")

# Extract relevant columns after sorting
matrices = df['Matrix']
speedup_mkl = df['Speedup over MKL']
speedup_lcm = df['Speedup over PSC']
speedup_csr5 = df['Speedup over CSR5']
speedup_csr = df['Speedup over CSR']

# Compute geometric means
geo_mkl = gmean(speedup_mkl[speedup_mkl > 0])
geo_lcm = gmean(speedup_lcm[speedup_lcm > 0])
geo_csr5 = gmean(speedup_csr5[speedup_csr5 > 0])
geo_csr = gmean(speedup_csr[speedup_csr > 0])

# Setup for grouped bar chart
num_matrices = len(matrices)
x = np.arange(num_matrices)
width = 0.2

# Create the plot
plt.figure(figsize=(14, 6))
plt.bar(x - 1.5*width, speedup_mkl, width, label=f'Over MKL (Geo-mean: {geo_mkl:.2f})')
plt.bar(x - 0.5*width, speedup_lcm, width, label=f'Over LCM I/E (Geo-mean: {geo_lcm:.2f})')
plt.bar(x + 0.5*width, speedup_csr5, width, label=f'Over CSR5 (Geo-mean: {geo_csr5:.2f})')
plt.bar(x + 1.5*width, speedup_csr, width, label=f'Over CSR (Geo-mean: {geo_csr:.2f})')

plt.yscale('log')

# Add horizontal line at y=1
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)

# Add labels and legend
plt.xlabel('Matrix', fontsize=18)
plt.ylabel('Speedup (log)', fontsize=16)
plt.title('Speedup of SABLE Over Various Baselines (1 Thread)', fontsize=20)
plt.xticks(x, matrices, rotation=90, fontsize=14)
plt.yticks([1, 2, 3, 4, 5, 6], ['1', '2', '3', '4', '5', '6'], fontsize=16)
plt.legend(fontsize=12, loc='upper center')
plt.grid(True, axis='y')

plt.ylim(bottom=None, top=6)
plt.tight_layout()

# Save the plot
plt.savefig("single.pdf", dpi=300)
