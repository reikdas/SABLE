from src.util import correct_widths, get_accumulated_values

# add a test case to test correct widths
def test_correct_widths():
    assert correct_widths([2, 3, 1, 3], 10) == [2, 3, 1, 3, 1]
    assert correct_widths([2, 3, 1, 3], 9) == [2, 3, 1, 3]
    
# add a test case to test get accumulated values
def test_accumulated_values():
    assert get_accumulated_values([]) == [0]
    assert get_accumulated_values([1, 2, 3, 4]) == [0, 1, 3, 6, 10]
    assert get_accumulated_values([1, 2, 3, 4, 5]) == [0, 1, 3, 6, 10, 15]
    
