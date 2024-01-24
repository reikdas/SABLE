#!/bin/bash

rows=(1000)
cols=(1000)
partition_types=(uniform non-uniform)
row_splits=(50 100)
col_splits=(50 100)
percentage_of_blocks=(20 15 10 5 1)
percentage_of_zeros=(50 40 30 20 10 0)
FAIL=0

# time execution of script
start=`date +%s.%N`
for row in "${rows[@]}"
do
    for col in "${cols[@]}"
    do
        for partition_type in "${partition_types[@]}"
        do
            for row_split in "${row_splits[@]}"
            do
                for col_split in "${col_splits[@]}"
                do
                    for percentage_of_block in "${percentage_of_blocks[@]}"
                    do
                        for percentage_of_zero in "${percentage_of_zeros[@]}"
                        do
                            python gen.py --num-rows $row --num-cols $col --partition-type $partition_type --row-split $row_split --col-split $col_split --percentage-of-blocks $percentage_of_block --percentage-of-zeros $percentage_of_zero &
                        done
                    done
                    # https://stackoverflow.com/questions/356100/how-to-wait-in-bash-for-several-subprocesses-to-finish-and-return-exit-code-0
                    # wait for the processes to finish
                    # the number of processes to wait for is the number of elements in the percentage of zeros array times the number of elements in the percentage of blocks array
                    for job in `jobs -p`
                    do
                        echo $job
                        wait $job || let "FAIL+=1"
                    done
                    echo $FAIL
                    if [ "$FAIL" == "0" ];
                    then
                        echo "YAY!"
                    else
                        echo "FAIL! ($FAIL)"
                    fi
                    echo "Done with $row $col $partition_type $row_split $col_split"
                done
            done
        done
    done
done
end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo "Runtime was $runtime"