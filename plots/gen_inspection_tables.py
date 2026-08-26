#!/usr/bin/env python3
"""Generate LaTeX tables of SABLE inspection/compilation times for the appendix.

Reads results/inspection_{vbr_csr,vdia_csr,vdia_vbr_csr}_spmv.json (produced by
SABLE/bench_inspection.py) and prints the LaTeX for the three tables of the
"Inspection Cost" appendix section to stdout (or to a file with -o), ready to
be pasted into appendix.tex. This script never modifies appendix.tex.

    python3 gen_inspection_tables.py            # print to stdout
    python3 gen_inspection_tables.py -o t.tex   # write to a file

Column semantics (all times in seconds):
  VDIA    = band search wall time (find_vdia_regions, min density 0.75)
  VBR     = block partitioner subprocess wall time (20 threads, 4-hour budget;
            a dagger marks matrices where the search exhausted the budget)
  CSR     = conversion of the residual matrix to CSR
  CodeGen = emission of the specialized C program and its data file
  GCC     = compilation of the generated program (--- marks failure)
  Total   = sum of the phase columns of the table (a double dagger marks totals
            that exclude a failed compilation)
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

COLUMNS = {
    'vbr_csr': ['vbr', 'csr', 'codegen', 'gcc'],
    'vdia_csr': ['vdia', 'csr', 'codegen', 'gcc'],
    'vdia_vbr_csr': ['vdia', 'vbr', 'csr', 'codegen', 'gcc'],
}
HEADERS = {'vdia': r'\textbf{VDIA}', 'vbr': r'\textbf{VBR}', 'csr': r'\textbf{CSR}',
           'codegen': r'\textbf{CodeGen}', 'gcc': r'\textbf{GCC}', 'total': r'\textbf{Total}'}


def load(set_name):
    path = os.path.join(RESULTS_DIR, f'inspection_{set_name}_spmv.json')
    with open(path) as f:
        entries = [e for e in json.load(f) if 'error' not in e]
    return sorted(entries, key=lambda e: e['nnz'])


def esc(name):
    return name.replace('_', r'\_')


def fmt_s(value):
    if value is None:
        return '---'
    if value >= 100:
        return f'{value:,.0f}'
    if value >= 1:
        return f'{value:.1f}'
    return f'{value:.2f}'


def gcc_ok(entry):
    info = entry['phases'].get('gcc', {})
    return not info.get('timed_out') and info.get('returncode') == 0


def phase_value(entry, column):
    """Seconds for one column, or None if the phase failed/is absent."""
    phases = entry['phases']
    if column == 'vdia':
        return phases['vdia']['search_seconds'] if 'vdia' in phases else None
    if column == 'vbr':
        return phases['vbr']['partitioner_seconds'] if 'vbr' in phases else None
    if column == 'csr':
        return phases['csr']['convert_seconds']
    if column == 'codegen':
        return phases['codegen']['wall_seconds']
    if column == 'gcc':
        return phases['gcc']['wall_seconds'] if gcc_ok(entry) else None
    raise KeyError(column)


def cell(entry, column):
    value = phase_value(entry, column)
    text = fmt_s(value)
    if column == 'vbr' and entry['phases'].get('vbr', {}).get('timeout'):
        text += r'$^\dagger$'
    return text


def total_cell(entry, columns):
    values = [phase_value(entry, c) for c in columns]
    total = sum(v for v in values if v is not None)
    text = fmt_s(total)
    if any(v is None for v in values):
        text += r'$^\ddagger$'
    return text


def row(entry, columns):
    cells = [esc(entry['matrix_name'])]
    cells.extend(cell(entry, c) for c in columns)
    cells.append(total_cell(entry, columns))
    return '    ' + ' & '.join(cells) + r' \\'


def tabular(entries, columns):
    lines = [r'    \begin{tabular}{l ' + 'r' * len(columns) + ' r}',
             r'    \toprule',
             '    ' + ' & '.join([r'\textbf{Matrix}'] + [HEADERS[c] for c in columns + ['total']]) + r' \\',
             r'    \midrule']
    lines.extend(row(entry, columns) for entry in entries)
    lines.append(r'    \bottomrule')
    lines.append(r'    \end{tabular}')
    return lines


def has_failed_gcc(entries):
    return any(not gcc_ok(e) for e in entries)


def vbr_csr_table():
    entries = load('vbr_csr')
    columns = COLUMNS['vbr_csr']
    half = (len(entries) + 1) // 2
    lines = [
        r'\begin{table*}[t!]',
        r'  \caption{One-time inspection and compilation cost of \sys{} for the VBR+CSR',
        r'    configuration on the 55 matrices of \autoref{table:matrices} (seconds,',
        r'    sorted by non-zeros). \textit{VBR} is the block-partitioner wall time',
        r'    (20 threads, four-hour budget; $^\dagger$ marks matrices where the search',
        r'    exhausted the budget and reported the blocks found so far), \textit{CSR}',
        r'    the conversion of the residual matrix, \textit{CodeGen} the emission of',
        r'    the specialized C program and its data file, \textit{GCC} the',
        r'    compilation of the generated program, and \textit{Total} the sum of the',
        r'    preceding columns. CodeGen and GCC are for the SpMV program with',
        r"    \sys{}'s naive kernels for every dispatch.}",
        r'  \label{table:inspection_vbr_csr}',
        r'  \centering',
        r'  \resizebox{\textwidth}{!}{%',
    ]
    lines.extend(tabular(entries[:half], columns))
    lines.append(r'    \hspace{2.5em}')
    lines.extend(tabular(entries[half:], columns))
    lines.append(r'  }')
    lines.append(r'\end{table*}')
    return lines


def single_column_table(set_name, label, caption_lines):
    entries = load(set_name)
    lines = [r'\begin{table}[t!]', r'  \caption{%']
    lines.extend(caption_lines)
    if has_failed_gcc(entries):
        failed = ', '.join(r'\textit{' + esc(e['matrix_name']) + '}' for e in entries if not gcc_ok(e))
        lines.append(rf'    On {failed}, GCC v11.4.0 aborts with an internal compiler error')
        lines.append(r'    (marked ---; $^\ddagger$ marks a total that excludes compilation).')
    lines.extend([
        r'  }',
        rf'  \label{{{label}}}',
        r'  \centering',
        r'  \resizebox{\columnwidth}{!}{%',
    ])
    lines.extend(tabular(entries, COLUMNS[set_name]))
    lines.append(r'  }')
    lines.append(r'\end{table}')
    return lines


def main():
    lines = []
    lines.extend(vbr_csr_table())
    lines.append('')
    lines.extend(single_column_table('vdia_csr', 'table:inspection_vdia_csr', [
        r'    One-time inspection and compilation cost of \sys{} for the VDIA+CSR',
        r'    configuration on the 28 matrices of \autoref{table:vdia_csr} (seconds,',
        r'    sorted by non-zeros). \textit{VDIA} is the band-search wall time',
        r'    (minimum band density 0.75); the remaining columns are as in',
        r'    \autoref{table:inspection_vbr_csr}.',
    ]))
    lines.append('')
    lines.extend(single_column_table('vdia_vbr_csr', 'table:inspection_vdia_vbr_csr', [
        r'    One-time inspection and compilation cost of \sys{} for the VDIA+VBR+CSR',
        r'    configuration on the 24 matrices of \autoref{table:vdia_vbr_csr}',
        r'    (seconds, sorted by non-zeros). The band search runs on the full matrix',
        r'    and the block partitioner on the residual left after band extraction;',
        r'    columns are as in \autoref{table:inspection_vbr_csr} and',
        r'    \autoref{table:inspection_vdia_csr}. $^\dagger$ marks matrices where the',
        r'    block search exhausted its four-hour budget.',
    ]))
    lines.append('')

    text = '\n'.join(lines).rstrip('\n') + '\n'
    output = sys.argv[2] if len(sys.argv) == 3 and sys.argv[1] == '-o' else None
    if output:
        with open(output, 'w') as f:
            f.write(text)
        print(f'Wrote {output}', file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
