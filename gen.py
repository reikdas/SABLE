import random
from argparse import ArgumentParser
from enum import Enum

from src.mtx_matrices_gen import convert_all_vbr_to_mtx
from src.vbr_matrices_gen import vbr_matrix_gen
from src.codegen import vbr_spmv_codegen_for_all, vbr_spmm_codegen_for_all, vbr_spmv_cuda_codegen_for_all, vbr_spmm_cuda_codegen_for_all
from src.fileio import write_dense_matrix, write_dense_vector

class PartitionType(Enum):
    uniform = 'uniform'
    non_uniform = 'non-uniform'
    
class Operation(Enum):
    vbr = 'vbr'
    vbr_to_mtx = 'vbr_to_mtx'
    vbr_to_spmv = 'vbr_to_spmv'
    vbr_to_spmm = 'vbr_to_spmm'

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
    parser.add_argument("--dense-blocks-only", action="store_true", default=False, required=False, help="If set, only dense blocks are generated in the VBR matrix. If not set, both dense and sparse blocks are generated in the VBR matrix")
    parser.add_argument("--percentage-dense", type=int, default = 50, required=False, help="Percentage of dense blocks (out of all the blocks calculated using row-split*col-split) with non-zero values in the generated VBR matrix. Suggested to use values 20, 15, 10, 5, 1")
    parser.add_argument("--percentage-sparse", type=int, required=False, default=0)
    parser.add_argument("--percentage-of-zeros", type=int, default = 50, required=False, help="Percentage of zeros in a dense block in the generated VBR matrix. Suggested to use values 50, 40, 30, 20, 10, 0")
    parser.add_argument("-o", "--operation", type=Operation, choices=list(Operation), default="vbr", required=False, help="Operation to perform. Values are `vbr`, `vbr_to_mtx`, `vbr_to_spmv`, `vbr_to_spmm`. Default is vbr. If vbr, generates a VBR matrix with given configuration. If vbr_to_mtx, converts all VBR matrices in Generated_Data to Matrix Market format and saves them in Generated_Matrix.")
    parser.add_argument("--cuda", action='store_true')

    args = parser.parse_args()
    
    dense_blocks_only = args.dense_blocks_only
    percentage_sparse = args.percentage_sparse
    
    if (dense_blocks_only and percentage_sparse != 0):
        raise ValueError("dense_blocks_only and percentage_sparse cannot be set at the same time")
    
    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    random.seed(0)
    
    if (args.operation == Operation.vbr_to_mtx):
        if (args.dense_blocks_only):
            convert_all_vbr_to_mtx(dense_blocks_only=True)
        else:
            convert_all_vbr_to_mtx(dense_blocks_only=False)
        exit(0)
    elif (args.operation == Operation.vbr_to_spmv and args.cuda and args.dense_blocks_only):
        vbr_spmv_cuda_codegen_for_all(dense_blocks_only=True)
        exit(0)
    elif (args.operation == Operation.vbr_to_spmv and args.dense_blocks_only):
        vbr_spmv_codegen_for_all(dense_blocks_only=True)
        exit(0)
    elif (args.operation == Operation.vbr_to_spmv and args.cuda and not args.dense_blocks_only):
        vbr_spmv_cuda_codegen_for_all(dense_blocks_only=False)
        exit(0)
    elif (args.operation == Operation.vbr_to_spmv and not args.dense_blocks_only):
        vbr_spmv_codegen_for_all(dense_blocks_only=False)
        exit(0)
    elif (args.operation == Operation.vbr_to_spmm and args.cuda and args.dense_blocks_only):
        vbr_spmm_cuda_codegen_for_all(dense_blocks_only=True)
        exit(0)
    elif (args.operation == Operation.vbr_to_spmm and args.dense_blocks_only):
        vbr_spmm_codegen_for_all(dense_blocks_only=True)
        exit(0)
    elif (args.operation == Operation.vbr_to_spmm and args.cuda and not args.dense_blocks_only):
        vbr_spmm_cuda_codegen_for_all(dense_blocks_only=False)
        exit(0)
    elif (args.operation == Operation.vbr_to_spmm and not args.dense_blocks_only):
        vbr_spmm_codegen_for_all(dense_blocks_only=False)
        exit(0)
    
    dense_blocks_only = args.dense_blocks_only
    num_blocks = args.row_split * args.col_split
    num_vbr_blocks = (args.percentage_dense*num_blocks)//100

    num_sparse = (args.percentage_sparse*num_vbr_blocks)//100
    num_dense = num_vbr_blocks - num_sparse

    partition_type = "uniform" if args.partition_type == PartitionType.uniform else "nonuniform"
    
    vbr_matrix_gen(args.num_rows, args.num_cols, partition_type, args.row_split, args.col_split, num_dense, args.percentage_of_zeros, num_sparse, dense_blocks_only)
    write_dense_vector(1.0, args.num_cols)
    write_dense_matrix(1.0, args.num_cols, 512)
