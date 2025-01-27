#!/usr/bin/env bash
# Takes $FILE and replaces K foo-calls in it with $N calls to fooi,
# fooi are functions defined in separate files fooi.c (10<=i<N+10).
# Compiles the result into a.out.
#
# Example run:
#   ./split 64 filename.c
#
# The arguments are $FILE and $N, resp.
#
# Depends on GHC being installed
#
AFFINITY=$(cat /proc/$$/status | grep Cpus_allowed_list | cut -f2)
N=$1 # number of chunks; 64 can be decreased but can't go higher than 89
N1=$(($N + 9)) # start from 10 rather than from 1
FILE=$2
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# get the string before .c and after the last /
SAVE_DIR=${FILE##*/}  # Remove directory path -> "g7jac020.c"
SAVE_DIR=${SAVE_DIR%.c}   # Remove ".c" extension -> "g7jac020"
echo "Save dir: $SAVE_DIR"
echo "script dir $SCRIPT_DIR"
mkdir -p $SCRIPT_DIR/split-and-binaries/$SAVE_DIR
echo "0. Cleanup old fooi's and restore original C file"
rm -f foo*.c foo*.h
cp $FILE $SCRIPT_DIR/split-and-binaries/$SAVE_DIR/$SAVE_DIR.c 2>/dev/null || echo "WARN: No original.
NOTE: If you create a backup copy of '$FILE' called '$FILE.orig' before running the script,
we will use it to restore the original on every run."
FILE=$SCRIPT_DIR/split-and-binaries/$SAVE_DIR/$SAVE_DIR.c
echo '1. Extract the original foo-calls in `body.ins` and put foo$i-calls instead'
taskset -a -c $AFFINITY runhaskell "$SCRIPT_DIR/replace_foo.hs" $N $FILE split-and-binaries/$SAVE_DIR
echo "2. Split 'body.ins' into N"
split -n r/$N $SCRIPT_DIR/split-and-binaries/$SAVE_DIR/body.ins $SCRIPT_DIR/split-and-binaries/$SAVE_DIR/foo --numeric-suffixes=10 --additional-suffix=".c"
echo "3. Patch the foo\$i.c: add funciton header and footer"
for ((i=10; i<=$N1; i++)); do
#    sed -i "1i void foo$i(float *x, float *y, float *val) {" "foo$i.c"
    sed -i "1i #include \"foo.h\"\nvoid foo$i(float *x, float *y, float *val) {" "$SCRIPT_DIR/split-and-binaries/$SAVE_DIR/foo$i.c"
    echo "}" >> "$SCRIPT_DIR/split-and-binaries/$SAVE_DIR/foo$i.c"
done
echo "4. Create foo.h with the SpMV kernel (function foo)"
echo "
inline
void foo(float *y, const float* x, const float* val, int i_start, int i_end, int j_start, int j_end, int val_offset) {
    for (int j = j_start; j < j_end; j++) {
  for (int i = i_start; i < i_end; i++) {
      y[i] += ((&val[val_offset])[(((j-j_start)*(i_end-i_start)) + (i-i_start))] * x[j]);
    }
  }
}
" > $SCRIPT_DIR/split-and-binaries/$SAVE_DIR/foo.h
echo "5. Compiling and linking the result"
for f in $SCRIPT_DIR/split-and-binaries/$SAVE_DIR/foo*.c; do
    taskset -a -c $AFFINITY gcc -c -O3 -mavx -mprefer-vector-width=512 -funroll-all-loops -o ${f%.c}.o $f &
done
wait
taskset -a -c $AFFINITY gcc -c -O3 -mavx -mprefer-vector-width=512 -funroll-all-loops -Wno-implicit-function-declaration -o ${FILE%.c}.o $FILE
taskset -a -c $AFFINITY gcc -O3 -mavx -mprefer-vector-width=512 -funroll-all-loops -o $SCRIPT_DIR/split-and-binaries/$SAVE_DIR/$SAVE_DIR ${FILE%.c}.o $SCRIPT_DIR/split-and-binaries/$SAVE_DIR/foo*.o