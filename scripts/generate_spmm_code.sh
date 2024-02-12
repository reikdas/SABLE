#!/bin/bash

start=`date +%s.%N`
python gen.py -o vbr_to_spmm
end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo "Runtime was $runtime"