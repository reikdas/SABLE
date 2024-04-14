#!/usr/bin/env bash

rows=10000
cols=10000
partition_types="uniform non-uniform"
row_splits="50 100"
col_splits="50 100"
percentage_of_blocks="20 15 10 5 1"
percentage_of_zeros="99 75 50 40 30 20 10 0"

jobs=8

parallel -j $jobs python3 gen.py --num-rows {1} --num-cols {2} --partition-type {3} --row-split {4}  --col-split {5} --percentage-of-blocks {6} --percentage-of-zeros {7} \
                                ::: $rows      ::: $cols      ::: $partition_types ::: $row_splits  ::: $col_splits ::: $percentage_of_blocks ::: $percentage_of_zeros
