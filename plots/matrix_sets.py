"""The paper's named matrix sets, read from ../matrices.json.

The result files hold every matrix the benchmark was run on -- the full block
sweep is 117 matrices -- and the figures report a subset of them, so the
plotting scripts select by set rather than relying on which file an entry is in.
"""
import json
import os

_MATRICES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'matrices.json')


def load_matrix_set(name):
    """Names in one set of matrices.json: 'vbr_csr' (55), 'vdia_only' (19), 'fukaya' (4)."""
    with open(_MATRICES_JSON) as f:
        spec = json.load(f)
    return frozenset(m['name'] for m in spec[name]['matrices'])
