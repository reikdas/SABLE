from random import sample

def test_val_array_append():
    block_size = 100
    perc_zeros = 20
    
    zeros = sample([x for x in range(block_size)], (block_size * perc_zeros) // 100)
    zeros = set(zeros)
    
    val1 = []
    for index in range(block_size):
        if index in zeros:
            val1.append(0)
        else:
            val1.append(1)
    
    val2 = [0 if index in zeros else 1 for index in range(block_size)]
    
    assert (val1 == val2)
    
def test_extend_bpntrb():
    bpntrb = []
    
    curr_row = 0
    new_row = 10
    for _ in range(curr_row, new_row):
            bpntrb.append(-1)
            
    assert (bpntrb == [-1 for _ in range(curr_row, new_row)])
