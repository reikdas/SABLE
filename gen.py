from argparse import ArgumentParser
from enum import Enum

from src.gen_vbr_matrices import gen_vbr_matrix

class PartitionType(Enum):
    uniform = 'uniform'
    non_uniform = 'non-uniform'

# add parser method
def parse_arguments():
    '''
    Parse arguments for VBR
    
    '''
    parser = ArgumentParser(description='VBR')
    parser.add_argument("-r", "--num-rows", type=int, default=1000, required=False, help="Number of rows in the generated VBR matrix")
    parser.add_argument("-c", "--num-cols", type=int, default=1000, required=False, help="Number of columns in the generated VBR matrix")
    parser.add_argument("-p", "--partition-type", type=PartitionType, choices=list(PartitionType), default="uniform", required=False, help="Partition type for the generated VBR matrix. Default is uniform. If uniform, the number of rows is divided by row-split to create uniform partitions. If non-uniform, the number of rows is divided by row-split to create non-uniform partitions, where each partition size is randomly chosen, the sum of all partitions is guaranteed to sum to the total number of rows.")
    parser.add_argument("--row-split", type=int, default=50, required=False, help="Number of rows is divided by this number to get the number of row partitions in the generated VBR matrix")
    parser.add_argument("--col-split", type=int, default=50, required=False, help="Number of columns is divided by this number to get the number of column partitions in the generated VBR matrix")
    parser.add_argument("--percentage-of-blocks", type=int, default = 50, required=False, help="Percentage of dense blocks (out of all the blocks calculated using row-split*col-split) with non-zero values in the generated VBR matrix. Suggested to use values 20, 15, 10, 5, 1")
    parser.add_argument("--percentage-of-zeros", type=int, default = 50, required=False, help="Percentage of zeros in a dense block in the generated VBR matrix. Suggested to use values 50, 40, 30, 20, 10, 0")
    
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    num_blocks = args.row_split * args.col_split
    num_dense = (args.percentage_of_blocks*num_blocks)//100
        
    partition_type = "uniform" if args.partition_type == PartitionType.uniform else "nonuniform"
    
    gen_vbr_matrix(args.num_rows, args.num_cols, partition_type, args.row_split, args.col_split, num_dense, args.percentage_of_zeros)
    
    
    