from scipy.io import mmread
import matplotlib.pyplot as plt

from src.fileio import write_vbr_matrix
from src.util import convert_dense_to_vbr, convert_mtx_to_vbr

def test_get_dense_for_mtx():
    csr = mmread("tests/example-canon.mtx")
    
    dense = csr.toarray()
    # mask = dense != 0
    # dense[mask] = 1.0
    # del_stuff([csr, mask])
    
    print(dense)
    
    print(type(dense))
    
    row_widths = [2 for _ in range(0, dense.shape[0]//2)]
    col_widths = [2 for _ in range(0, dense.shape[1]//2)]
    
    row_widths = [2, 3, 1, 3, 2]
    col_widths = [2, 3, 1, 3, 2]
    
    vbr = convert_dense_to_vbr(dense, row_widths, col_widths)
            
    print(vbr.val)
    print(vbr.indx)
    print(vbr.bindx)
    print(vbr.rpntr)
    print(vbr.cpntr)
    print(vbr.bpntrb)
    print(vbr.bpntre)
    
    assert vbr.val == [4.0, 1.0, 2.0, 5.0, 1.0, 2.0, -1.0, 0.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 0.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 25.0, 8.0, 3.0, 12.0]
    assert vbr.indx == [0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51]
    assert vbr.bindx == [0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]
    assert vbr.rpntr == [0, 2, 5, 6, 9, 11]
    assert vbr.cpntr == [0, 2, 5, 6, 9, 11]
    assert vbr.bpntrb == [0, 3, 5, 9, 11]
    assert vbr.bpntre == [3, 5, 9, 11, 13]
    
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    # show numpy.matrix a in a plot
    plt.imshow(dense)
    plt.colorbar(orientation='vertical')
    ax.set_aspect('equal')
    
    # add vertical and horizontal lines, the gap between the lines should be 2
    # for i in range(0, a.shape[0], 2):
    count = 0
    for i, w in enumerate(row_widths):
        plt.axhline(count-0.5, color='black', lw=1)
        count += w
    count = 0
    for i, w in enumerate(col_widths):
        plt.axvline(count-0.5, color='black', lw=1)
        count += w
    
    plt.show()
    # save plot to file
    plt.savefig('tests/matrices/example.png')
    
    
def test_mtx_to_vbr_no_blk_in_rowp():
    csr = mmread("tests/example.mtx")
    dense = csr.toarray()
    
    row_widths = [2 for _ in range(0, dense.shape[0]//2)]
    col_widths = [2 for _ in range(0, dense.shape[1]//2)]
    
    row_widths = [2, 3, 1, 3, 2]
    col_widths = [2, 3, 1, 3, 2]
    
    vbr = convert_dense_to_vbr(dense, row_widths, col_widths)
            
    print(vbr.val)
    print(vbr.indx)
    print(vbr.bindx)
    print(vbr.rpntr)
    print(vbr.cpntr)
    print(vbr.bpntrb)
    print(vbr.bpntre)
    
    assert vbr.val == [4.0, 1.0, 2.0, 5.0, 1.0, 2.0, -1.0, 0.0, 1.0, -1.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 0.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 25.0, 8.0, 3.0, 12.0]
    assert vbr.indx == [0, 4, 6, 10, 12, 15, 16, 19, 22, 31, 35, 39]
    assert vbr.bindx == [0, 2, 4, 0, 1, 2, 3, 2, 3, 0, 4]
    assert vbr.rpntr == [0, 2, 5, 6, 9, 11]
    assert vbr.cpntr == [0, 2, 5, 6, 9, 11]
    assert vbr.bpntrb == [0, -1, 3, 7, 9]
    assert vbr.bpntre == [3, -1, 7, 9, 11]
