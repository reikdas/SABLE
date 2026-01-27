import os

import numpy

from utils.fileio import read_vbr

'''
This file contains functionality to convert VBR matrices to Matrix Market format.
'''

def find_nonneg(l):
    for _, ele in enumerate(l):
        if ele != -1:
            return ele
    assert(False)
