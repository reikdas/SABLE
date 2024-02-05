#!/usr/bin/env bash

rows=5000
cols=5000
partition_types="uniform non-uniform"
row_splits="50 100"
col_splits="50 100"
percentage_of_blocks="20 15 10 5 1"
percentage_of_zeros="99 75 50 40 30 20 10 0"

jobs=8

# time execution of script
start=`date +%s.%N`
parallel -j $jobs python3 gen.py --num-rows {1} --num-cols {2} --partition-type {3} --row-split {4}  --col-split {5} --percentage-of-blocks {6} --percentage-of-zeros {7} \
                                ::: $rows      ::: $cols      ::: $partition_types ::: $row_splits  ::: $col_splits ::: $percentage_of_blocks ::: $percentage_of_zeros
end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo "Runtime was $runtime"
