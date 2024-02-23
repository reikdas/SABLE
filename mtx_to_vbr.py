from scipy.io import mmread

from src.util import convert_mtx_to_vbr
from src.fileio import write_vbr_matrix

def conversion1():
    # change to the input file
    input = "bcsstk14/bcsstk14.mtx"
    # input = "bcsstk16/bcsstk16.mtx"
    coo = mmread(input)
    
    # fill in these arrays to manually fit in a variable block size grid
    # by default it fits in blocks of 2x2 if not provided
    row_widths = None
    col_widths = None
    
    vbr = convert_mtx_to_vbr(coo = coo, row_widths = row_widths, col_widths = col_widths)
    write_vbr_matrix(filename = "bcsstk14", vbr_matrix = vbr, dir_name = "mtx_to_vbr")
    
def conversion2():
    # change to the input file
    input = "bcsstk14/bcsstk14.mtx"
    # input = "bcsstk16/bcsstk16.mtx"
    coo = mmread(input)
    
    # fill in these arrays to manually fit in a variable block size grid
    # by default it fits in blocks of 2x2 if not provided
    row_widths = [4 for _ in range(0, coo.shape[0]//4)]
    col_widths = [4 for _ in range(0, coo.shape[1]//4)]
    
    vbr = convert_mtx_to_vbr(coo = coo, row_widths = row_widths, col_widths = col_widths)
    write_vbr_matrix(filename = "bcsstk14_4x4", vbr_matrix = vbr, dir_name = "mtx_to_vbr")
    
if __name__ == "__main__":
    conversion1()