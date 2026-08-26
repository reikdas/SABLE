#!/usr/bin/env python3
"""Load per-backend VDIA+CSR results for the four diagonal-band matrices
drawn from Fukaya et al. (circuit5M_dc, memchip, ohne2, CoupCons3D).

These were measured in a single session with in-run CSR baselines
(SpMV: mkl, spv8; SpMM: mkl). The naive dispatch from the earlier
manual run is omitted: it was measured in a separate session, so its
absolute times are not comparable with these. UZP was also measured
but is omitted: it was never the fastest on either side, and it has
no column in the VDIA tables, so including it would silently distort
the per-row best-speedup highlighting.

Entries are returned in the condensed per_backend schema used by the
spm{v,m}_vdia_csr_d075.json files, so callers can merge them directly.
"""

import json
import os
import re

# thread is excluded from all VDIA results: its band-extraction metadata
# is defective and its generated program produces incorrect output.
EXCLUDED_MATRICES = frozenset({'thread'})

FUKAYA_FILES = {
    'spmv': {
        'mkl': 'sable_spmv_bandmkl_mkl_circuit5M_dc_memchip_ohne2_CoupCons3D.json',
        'spv8': 'sable_spmv_bandmkl_spv8_circuit5M_dc_memchip_ohne2_CoupCons3D.json',
    },
    'spmm': {
        'mkl': 'sable_spmm_bandnaive_mkl_circuit5M_dc_memchip_ohne2_CoupCons3D.json',
    },
}

_PART_RE = re.compile(r'dispatch_1_part_(\d+)')


def _n_segments(timing):
    # The VDIA dispatch (dispatch 1) reports one part per segment.
    parts = {m.group(1) for k in timing.get('dispatch_part_times', {})
             for m in [_PART_RE.match(k)] if m}
    return len(parts)


def load_fukaya(results_dir, operation):
    """Return {matrix_name: entry} for the given operation ('spmv'/'spmm')."""
    out = {}
    for backend, fname in FUKAYA_FILES[operation].items():
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            raw = json.load(f)
        for e in raw:
            t = e['timing']['1 thread']
            baseline_us = t['csr_baseline_time_ns'] / 1000.0
            vdia_us = t['total_time_ns'] / 1000.0
            m = out.setdefault(e['matrix_name'], {
                'matrix_name': e['matrix_name'],
                'rows': e['matrix_dimensions']['rows'],
                'cols': e['matrix_dimensions']['cols'],
                'nnz': e['matrix_dimensions']['nnz'],
                'density': '0.075',
                'band_nnz_pct': e['nnz']['format_claimed_nnz_perc'],
                'n_segments': _n_segments(t),
                'per_backend': {},
            })
            m['per_backend'][backend] = {
                'vdia_us': round(vdia_us, 2),
                'baseline_us': round(baseline_us, 2),
                'speedup': round(baseline_us / vdia_us, 4),
            }
    for m in out.values():
        best_v = min(m['per_backend'].items(), key=lambda kv: kv[1]['vdia_us'])
        best_b = min(m['per_backend'].items(), key=lambda kv: kv[1]['baseline_us'])
        m['best_vs_best'] = {
            'best_vdia_us': best_v[1]['vdia_us'],
            'best_vdia_backend': best_v[0],
            'best_baseline_us': best_b[1]['baseline_us'],
            'best_baseline_backend': best_b[0],
            'speedup': round(best_b[1]['baseline_us'] / best_v[1]['vdia_us'], 4),
        }
    return out
