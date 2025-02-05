#!/usr/bin/env bash
set -e

C_OPTS="-O3 -mavx -mprefer-vector-width=512 -funroll-all-loops -Wno-implicit-function-declaration"

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

echo "$FILE"
echo "$FILE_NODIR"
echo "$BASENAME"
echo "$CHUNK_SIZE"
echo "$OUT_DIR"

SPLIT_NAME="${BASENAME}_split"

echo "$SPLIT_NAME"

START_FILE="start_${SPLIT_NAME}"
echo "$START_FILE"

python3 $(dirname "$0")/splitter.py "$FILE" "$CHUNK_SIZE"

cd "${OUT_DIR}"

for f in "${SPLIT_NAME}"*.c; do
    echo $f
    gcc -c $C_OPTS -Winline $f &
done

wait
gcc -c $C_OPTS -Wno-implicit-function-declaration $START_FILE.c
gcc $C_OPTS -o "./$BASENAME" "$START_FILE.o" ./${SPLIT_NAME}*.o

cd "../.."
