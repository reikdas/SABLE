#!/bin/bash

rows=(5000)
cols=(5000)
partition_types=(uniform non-uniform)
row_splits=(50 100)
col_splits=(50 100)
percentage_of_blocks=(20 15 10 5 1)
percentage_of_zeros=(99 75 50 40 30 20 10 0)
FAIL=0
NUM_PROCESSES=30
total_processes=0

# create function to include job wait
# https://stackoverflow.com/questions/356100/how-to-wait-in-bash-for-several-subprocesses-to-finish-and-return-exit-code-0
# wait for the processes to finish
wait_for_jobs() {
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
    FAIL=0
}

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
                            let "total_processes+=1"

                            # limit the number of threads
                            if [ $total_processes -eq $NUM_PROCESSES ]
                            then
                                # wait for the processes to finish
                                wait_for_jobs
                                total_processes=0
                            fi
                        done
                    done
                done
            done
        done
    done
done
wait_for_jobs
end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo "Runtime was $runtime"