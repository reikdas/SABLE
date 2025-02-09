#!/usr/bin/env bash
set -e

AFFINITY=$(cat /proc/$$/status | grep Cpus_allowed_list | cut -f2)

C_OPTS="-O3 -mavx -mprefer-vector-width=512 -funroll-all-loops -Wno-implicit-function-declaration -ffast-math"

# Take a file name as an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file.c> <chunk_size>"
    exit 1
fi

FILE=$1
CHUNK_SIZE=$2

FILE_NODIR="${FILE##*/}"  # Remove directory path -> "g7jac020.c"
BASENAME="${FILE_NODIR%.c}"   # Remove ".c" extension -> "g7jac020"

OUT_DIR="split-and-binaries/${BASENAME}/"

SPLIT_NAME="${BASENAME}_split"

START_FILE="start_${SPLIT_NAME}"

taskset -a -c $AFFINITY python3 $(dirname "$0")/splitter.py "$FILE" "$CHUNK_SIZE"

cd "${OUT_DIR}"

for f in "${SPLIT_NAME}"*.c; do
    taskset -a -c $AFFINITY gcc -c $C_OPTS -Winline $f &
done

wait
taskset -a -c $AFFINITY gcc -c $C_OPTS $START_FILE.c
taskset -a -c $AFFINITY gcc $C_OPTS -o "./$BASENAME" "$START_FILE.o" ./${SPLIT_NAME}*.o

cd "../.."
