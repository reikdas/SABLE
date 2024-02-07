#!/bin/bash

start=`date +%s.%N`
python3 gen.py -o vbr_to_mtx
end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo "Runtime was $runtime"