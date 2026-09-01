#!/usr/bin/env python3

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(here, os.pardir, 'results', 'inspection',
                         'inspection_vbr_csr_spmv.json')
out_path = os.path.join(here, 'images', 'block_discovery.pdf')

with open(data_path) as f:
    entries = json.load(f)

assert len(entries) == 55, f'expected 55 matrices, got {len(entries)}'

saturation_times = sorted(
    max(b['found_at_seconds'] for b in e['phases']['vbr']['blocks'])
    for e in entries
)
total = len(saturation_times)
counts = range(1, total + 1)

fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.scatter(saturation_times, counts, s=8, alpha=0.8, edgecolors='none',
           zorder=3)

ax.set_xlabel('Time of last block discovery (seconds)', fontsize=7)
ax.set_ylabel(f'Matrices saturated (of {total})', fontsize=7)
ax.tick_params(labelsize=6)
ax.grid(True, linewidth=0.3, alpha=0.5)

fig.savefig(out_path, bbox_inches='tight', dpi=300)
print(f'wrote {out_path} ({total} matrices)')
