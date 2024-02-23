from src.util import convert_mtx_to_vbr
from src.fileio import write_vbr_matrix

def conversion1():
    # change to the input file
    input = "bcsstk14/bcsstk14.mtx"
    
    # fill in these arrays to manually fit in a variable block size grid
    # by default it fits in blocks of 2x2 if not provided
    row_widths = []
    col_widths = []
    
    vbr = convert_mtx_to_vbr(input = input, row_widths = None, col_widths = None)
    write_vbr_matrix(filename = "bcsstk14", vbr_matrix = vbr, dir_name = "mtx_to_vbr")
    
if __name__ == "__main__":
    conversion1()