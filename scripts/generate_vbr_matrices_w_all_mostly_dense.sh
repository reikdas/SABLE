#!/usr/bin/env bash

rows=10000
cols=10000
partition_types="non-uniform"
percentage_dense="20 15 10 5 1"
percentage_sparse="0"
row_splits="50 100"
col_splits="50 100"
percentage_of_zeros="99 75 50 40 30 20 10 0"
jobs=12

parallel -j $jobs python3 gen.py --dense-blocks-only  --num-rows {1} --num-cols {2} --partition-type {3} --row-split {4}  --col-split {5} --percentage-dense {6} --percentage-sparse $percentage_sparse --percentage-of-zeros {8} \
                                ::: $rows      ::: $cols      ::: $partition_types ::: $row_splits  ::: $col_splits ::: $percentage_dense ::: $percentage_sparse ::: $percentage_of_zeros
