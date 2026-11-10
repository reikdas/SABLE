#!/bin/bash
# Time three_format.py's comparison: for every (operation, matrix) pair with a
# find-submatrices/three_format/results_vbr050_vdia<op>/<matrix>.yaml, VBR+CSR, VDIA+CSR, and
# VBR+VDIA+CSR (blocks first), each in its own bench_suitesparse.py process
# with its own CSR-only baseline, replaying three_format.py's extractions. The
# variant order rotates from matrix to matrix. Results merge into
# results/three_format/; every output is checked against SciPy. Timed binaries
# run on core SABLE_FIRST_CORE (default 0); SABLE_LOCAL_SUITESPARSE_DIR points
# at a local copy of the matrices.
#   three_format.sh SPMV_KERNELS SPMM_KERNELS [bench_suitesparse.py options]
#   e.g. three_format.sh naive,mkl,spv8,uzp naive,mkl
#        three_format.sh none spreg --max-run-seconds 3600 --run-timeout 21600
set -u
SPMV_KERNELS=${1:?usage: three_format.sh SPMV_KERNELS SPMM_KERNELS [options]}
SPMM_KERNELS=${2:?usage: three_format.sh SPMV_KERNELS SPMM_KERNELS [options]}
shift 2
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
EXTRACTIONS=$ROOT/find-submatrices/three_format
VARIANTS=(vbr vdia vbr_vdia)
export PYTHONUNBUFFERED=1

i=0
for op in spmv spmm; do
    case $op in
        spmv) CSR=$SPMV_KERNELS ;;
        spmm) CSR=$SPMM_KERNELS ;;
    esac
    [ "$CSR" = none ] && continue
    for y in "$EXTRACTIONS/results_vbr050_vdia$op"/*.yaml; do
        [ -e "$y" ] || continue
        m=$(basename "$y" .yaml)
        for k in 0 1 2; do
            v=${VARIANTS[$(( (i + k) % 3 ))]}
            case $v in
                vbr)      FORMATS=(--no-vdia --vbr-kernels blockmixed) ;;
                vdia)     FORMATS=(--vbr-kernels none) ;;
                vbr_vdia) FORMATS=(--order vbr,vdia --vbr-kernels blockmixed) ;;
            esac
            echo "=== $op $m: $v ($CSR) $(date '+%F %T')"
            "$PY" "$ROOT/bench_suitesparse.py" --operation "$op" --csr-kernels "$CSR" "${FORMATS[@]}" \
                --extraction-dir "$EXTRACTIONS" --output-dir "$ROOT/results/three_format" \
                "$@" "$m" < /dev/null
        done
        i=$((i + 1))
    done
done
echo "=== done $(date '+%F %T')"
