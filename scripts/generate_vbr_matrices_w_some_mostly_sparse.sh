#!/usr/bin/env bash

rows=10000
cols=10000
partition_types="uniform non-uniform"
percentage_dense="20"
percentage_sparse="40"
row_splits="50"
col_splits="50"
percentage_of_zeros="75 50 40 30 20 10 0"
jobs=8

parallel -j $jobs python3 gen.py --num-rows {1} --num-cols {2} --partition-type {3} --row-split {4}  --col-split {5} --percentage-dense {6} --percentage-sparse {7} --percentage-of-zeros {8} \
                                ::: $rows      ::: $cols      ::: $partition_types ::: $row_splits  ::: $col_splits ::: $percentage_dense ::: $percentage_sparse ::: $percentage_of_zeros
