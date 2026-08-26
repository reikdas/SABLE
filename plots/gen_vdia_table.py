#!/usr/bin/env python3
"""Generate LaTeX tables for VDIA evaluation results.

Produces two tables in the style of the VBR+CSR table (Table 1):
  1. VDIA+CSR vs CSR  (per-backend times and speedups)
  2. VDIA+VBR+CSR vs VBR+CSR  (per-backend times and speedups)
"""

import json
import os
import sys
import math

from fukaya_results import EXCLUDED_MATRICES, load_fukaya


SPMV_BACKENDS = ['naive', 'mkl', 'spv8']
SPMM_BACKENDS = ['naive', 'mkl']

BACKEND_DISPLAY = {
    'naive': 'Naive', 'mkl': 'MKL', 'spv8': 'SpV8',
}


def load_json(path):
    with open(path) as f:
        return {m['matrix_name']: m for m in json.load(f)
                if m['matrix_name'] not in EXCLUDED_MATRICES}


def load_data(results_dir):
    files = {
        'spmv_vdia_csr': 'spmv_vdia_csr_d075.json',
        'spmm_vdia_csr': 'spmm_vdia_csr_d075.json',
        'spmv_vdia_vbr_csr': 'spmv_vdia_vbr_csr_d075.json',
        'spmm_vdia_vbr_csr': 'spmm_vdia_vbr_csr_d075.json',
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found", file=sys.stderr)
            continue
        data[key] = load_json(path)
    data['spmv_vdia_csr'].update(load_fukaya(results_dir, 'spmv'))
    data['spmm_vdia_csr'].update(load_fukaya(results_dir, 'spmm'))
    return data


def format_cell(per_backend, backend, all_backends_data, bold_best=True):
    """Format cell as: time (speedup×).

    Underlines the lowest VDIA time across backends.
    Bolds the highest speedup across backends.
    """
    if backend not in per_backend:
        return '---'
    info = per_backend[backend]
    time_us = info['vdia_us']
    speedup = info['speedup']

    time_str = f'{time_us:,.0f}'
    speedup_str = f'{speedup:.2f}'

    is_best_time = True
    is_best_speedup = True
    for b, bdata in all_backends_data.items():
        if b == backend:
            continue
        if bdata['vdia_us'] < time_us:
            is_best_time = False
        if bdata['speedup'] > speedup:
            is_best_speedup = False

    if is_best_time and len(all_backends_data) > 1:
        time_str = f'\\underline{{{time_str}}}'

    if is_best_speedup and bold_best and len(all_backends_data) > 1:
        speedup_str = f'\\textbf{{{speedup_str}}}'

    return f'{time_str} ({speedup_str}$\\times$)'


def generate_table(data, spmv_key, spmm_key, caption, label,
                   comparison_label):
    """Generate one VDIA table (either VDIA+CSR or VDIA+VBR+CSR)."""
    ref = data[spmv_key]
    matrices = sorted(ref.keys(), key=lambda n: ref[n]['nnz'])

    n_spmv = len(SPMV_BACKENDS)
    n_spmm = len(SPMM_BACKENDS)
    n_total = 6 + n_spmv + n_spmm  # 6 property cols

    col_spec = 'l|rrrrr|' + 'r' * n_spmv + '|' + 'r' * n_spmm
    spmv_start = 7
    spmv_end = spmv_start + n_spmv - 1
    spmm_start = spmv_end + 1
    spmm_end = spmm_start + n_spmm - 1

    lines = []
    lines.append(r'\begin{table*}[t!]')
    lines.append(f'  \\caption{{{caption}}}')
    lines.append(f'    \\label{{{label}}}')
    lines.append(r'    \centering')
    lines.append(r'    \resizebox{\textwidth}{!}{')
    lines.append(f'    \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')

    lines.append(
        f'    & & & & & & '
        f'\\multicolumn{{{n_spmv + n_spmm}}}{{c}}'
        f'{{\\textbf{{{comparison_label}}}}} \\\\'
    )
    lines.append(f'    \\cmidrule(l){{{spmv_start}-{spmm_end}}}')

    lines.append(
        f'    & & & & & & '
        f'\\multicolumn{{{n_spmv}}}{{c|}}{{\\textbf{{SpMV}}}} & '
        f'\\multicolumn{{{n_spmm}}}{{c}}{{\\textbf{{SpMM}}}} \\\\'
    )
    lines.append(
        f'    \\cmidrule(lr){{{spmv_start}-{spmv_end}}} '
        f'\\cmidrule(l){{{spmm_start}-{spmm_end}}}'
    )

    spmv_headers = ' & '.join(
        f'\\textbf{{{BACKEND_DISPLAY[b]}}}' for b in SPMV_BACKENDS
    )
    spmm_headers = ' & '.join(
        f'\\textbf{{{BACKEND_DISPLAY[b]}}}' for b in SPMM_BACKENDS
    )
    lines.append(
        f'    \\textbf{{Matrix}} & \\textbf{{Rows}} & \\textbf{{Cols}} '
        f'& \\textbf{{Nnz}} & \\textbf{{\\shortstack{{Band\\\\cov.\\ (\\%)}}}}'
        f' & \\textbf{{Segs}} '
        f'& {spmv_headers} & {spmm_headers} \\\\'
    )
    lines.append(r'    \midrule')

    for name in matrices:
        m = ref[name]
        tex_name = name.replace('_', r'\_')

        props = [
            f'    {tex_name}',
            f'{m["rows"]:,}',
            f'{m["cols"]:,}',
            f'{m["nnz"]:,}',
            f'{m["band_nnz_pct"]:.1f}',
            str(m['n_segments']),
        ]

        spmv_pb = data[spmv_key][name]['per_backend']
        spmv_cells = [
            format_cell(spmv_pb, b, spmv_pb) for b in SPMV_BACKENDS
        ]

        spmm_pb = data[spmm_key][name]['per_backend']
        spmm_cells = [
            format_cell(spmm_pb, b, spmm_pb) for b in SPMM_BACKENDS
        ]

        lines.append(
            ' & '.join(props + spmv_cells + spmm_cells) + r' \\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    }')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)


def print_stats(data, key, label):
    speedups = [m['best_vs_best']['speedup'] for m in data[key].values()]
    wins = sum(1 for s in speedups if s >= 1.0)
    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print(f'{label}: {wins}/{len(speedups)} wins, '
          f'geo mean {geo:.4f}x, '
          f'range [{min(speedups):.4f}, {max(speedups):.4f}]',
          file=sys.stderr)


if __name__ == '__main__':
    results_dir = os.path.join(
        os.path.dirname(__file__), 'results'
    )
    data = load_data(results_dir)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

    if mode in ('vdia_csr', 'both'):
        table1 = generate_table(
            data, 'spmv_vdia_csr', 'spmm_vdia_csr',
            caption=(
                'Speedup of VDIA+CSR over CSR on 28 SuiteSparse matrices '
                'with diagonal band structure. '
                '\\textit{Band cov.}\\ is the percentage of non-zeros '
                'residing in diagonal band segments. '
                '\\textit{Segs}\\ is the number of VDIA segments. '
                'Times are in $\\mu$s; the highest speedup per benchmark '
                'is in bold and the lowest time is underlined; '
                '--- marks dispatches not measured for a matrix.'
            ),
            label='table:vdia_csr',
            comparison_label='VDIA+CSR speedup over CSR',
        )
        if mode == 'vdia_csr':
            print(table1)
        else:
            with open(os.path.join(
                os.path.dirname(__file__), 'vdia_csr_table.tex'
            ), 'w') as f:
                f.write(table1 + '\n')
            print(f'Wrote vdia_csr_table.tex', file=sys.stderr)

    if mode in ('vdia_vbr_csr', 'both'):
        table2 = generate_table(
            data, 'spmv_vdia_vbr_csr', 'spmm_vdia_vbr_csr',
            caption=(
                'Speedup of VDIA+VBR+CSR over VBR+CSR on 24 SuiteSparse '
                'matrices with diagonal band structure. '
                '\\textit{Band cov.}\\ is the percentage of non-zeros '
                'residing in diagonal band segments. '
                '\\textit{Segs}\\ is the number of VDIA segments. '
                'Times are in $\\mu$s; the highest speedup per benchmark '
                'is in bold and the lowest time is underlined.'
            ),
            label='table:vdia_vbr_csr',
            comparison_label='VDIA+VBR+CSR speedup over VBR+CSR',
        )
        if mode == 'vdia_vbr_csr':
            print(table2)
        else:
            with open(os.path.join(
                os.path.dirname(__file__), 'vdia_vbr_csr_table.tex'
            ), 'w') as f:
                f.write(table2 + '\n')
            print(f'Wrote vdia_vbr_csr_table.tex', file=sys.stderr)

    print('\n% --- Statistics ---', file=sys.stderr)
    print_stats(data, 'spmv_vdia_csr', 'SpMV: VDIA+CSR vs CSR')
    print_stats(data, 'spmm_vdia_csr', 'SpMM: VDIA+CSR vs CSR')
    print_stats(data, 'spmv_vdia_vbr_csr', 'SpMV: VDIA+VBR+CSR vs VBR+CSR')
    print_stats(data, 'spmm_vdia_vbr_csr', 'SpMM: VDIA+VBR+CSR vs VBR+CSR')
