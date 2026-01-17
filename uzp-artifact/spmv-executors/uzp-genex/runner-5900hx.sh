#!/bin/sh

BIN_DIR=`dirname $0`;
SPFDATA_DIR="$BIN_DIR/../../data";
CPU_SET="2,4,6,8,10,12,14"

#### Set OpenMP environment for miniserver/5900HX
## Parallel environment setting. Here, 16 h/w threads.
export OMP_SCHEDULE=static
export OMP_DYNAMIC=FALSE
export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15";
export KMP_SCHEDULE=static,balanced
export KMP_AFFINITY="proclist=[0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15]"
export OMP_NUM_THREADS=8;
export KMP_NUM_THREADS=8;
####

nbthreads="$OMP_NUM_THREADS";
curtime=`date "+%Y.%m.%d-%H.%M.%S"`;
CSV_FILE="$BIN_DIR/expe-spmm-$curtime.csv";

echo "Build SpMM program:"
make_cmd=`cd "$BIN_DIR" && make; cd -`;
echo "$make_cmd";

echo "[INFO] Storing data in $CSV_FILE";
rm -f "$CSV_FILE";
echo "#name NNZ INC unincp format dims nj TJ dimp nbshapes float bestgfs nbthreads bestgfs_omp" >> "$CSV_FILE";


error_list="";
NEWLINE='
';
for i in `find "$SPFDATA_DIR" -name '*d[1,2]*.spfdata' | sort`; do
	#echo "========= TEST $i =========";
	name=`echo "$i" | xargs basename`;
	res=`taskset -c "$CPU_SET" ./spf_matmul "$i" -float -stats`;
	if [ $? -ne 0 ]; then
		error_list="$error_list$NEWLINE$name";
		continue;
	fi;
	resomp=`./spf_matmul_omp "$i" -float -stats`;
	NNZ=`echo "$res" | grep "NNZ" | cut -d '=' -f 2 | tr -d ' '`;
	INC=`echo "$res" | grep "INC" | cut -d '=' -f 2 | tr -d ' '`;
	unincp=`echo "scale=2;100*($NNZ-$INC)/$NNZ" | bc`;
	format=`echo "$res" | grep "format" | cut -d '=' -f 2 | cut -d ' ' -f 2 | tr -d ' '`;
	dims=`echo "$res" | grep "dims" | cut -d '=' -f 2 | cut -d ' ' -f 2 | tr -d ' '`;
	nbshapes=`echo "$res" | grep "Base sh" | cut -d '=' -f 2 | cut -d ' ' -f 2 | tr -d ' '`;
	dimp=`echo "$res" | grep "Max dim" | cut -d '=' -f 2 | cut -d ' ' -f 2 | tr -d ' '`;
	nj=`echo "$res" | grep "C(dense)" | cut -d '=' -f 2 | cut -d 'x' -f 2 | cut -d ',' -f 1 | tr -d ' '`;
	TJ=`echo "$resomp" | grep "Adjusted TJ" | cut -d '=' -f 2 | cut -d ',' -f 1 | tr -d ' '`;
	if [ -z "$TJ" ]; then TJ="$nj"; fi;
	bestgfs=`echo "$res" | sort -n -r | head -n 1`;
	bestgfs_omp=`echo "$resomp" | sort -n -r | head -n 1`;
	echo "$name $NNZ $INC $unincp $format $dims $nj $TJ $dimp $nbshapes float $bestgfs $nbthreads $bestgfs_omp";
	echo "$name $NNZ $INC $unincp $format $dims $nj $TJ $dimp $nbshapes float $bestgfs $nbthreads $bestgfs_omp" >> "$CSV_FILE";
done;

echo "[INFO] All data stored in $CSV_FILE";
if ! [ -z "$error_list" ]; then
	echo "[ERROR] These matrices failed:";
	echo "$error_list";
	exit 1;
fi;
exit 0;

