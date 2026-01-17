#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UZP_ARTIFACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 GROUP MATRIX_NAME [float|double] [hot|cold] [1th|2th|8th]"
    exit 1
fi

GROUP=$1
MATRIX_NAME=$2
FLOAT_DATA_TYPE=$3
CACHE_MODE=$4
EXECUTION_THREADS=$5

# Colors
BYellow='\033[1;33m'      # Yellow
Color_Off='\033[0m'       # Text Reset
BackRed='\033[0;41m'      # Back Red
SCRIPT_TAG="${BYellow}[SCRIPT]${Color_Off}"
ERROR_TAG="${BackRed}[ERROR]${Color_Off}"

# Validate FLOAT_DATA_TYPE
if [[ "$FLOAT_DATA_TYPE" != "float" && "$FLOAT_DATA_TYPE" != "double" ]]; then
    echo -e "${ERROR_TAG} Invalid FLOAT_DATA_TYPE: $FLOAT_DATA_TYPE. Must be 'float' or 'double'."
    exit 1
else
    echo -e "${SCRIPT_TAG} FLOAT_DATA_TYPE: $FLOAT_DATA_TYPE"
fi

# Validate CACHE_MODE
if [[ "$CACHE_MODE" != "hot" && "$CACHE_MODE" != "cold" ]]; then
    echo -e "${ERROR_TAG} Invalid CACHE_MODE: $CACHE_MODE. Must be 'hot' or 'cold'."
    exit 1
else
    echo -e "${SCRIPT_TAG} CACHE_MODE: $CACHE_MODE"
fi

# Validate EXECUTION_THREADS
if [[ "${EXECUTION_THREADS}" != "1th" && "${EXECUTION_THREADS}" != "2th" && "${EXECUTION_THREADS}" != "8th" ]];
then
    echo -e "${ERROR_TAG} Invalid EXECUTION_THREADS: ${EXECUTION_THREADS}. Must be '1th', '2th', or '8th'."
    exit 1
else
    echo -e "${SCRIPT_TAG} EXECUTION_THREADS: ${EXECUTION_THREADS}"
fi

TEMP_DIRECTORY="/tmp"
MATRIX_FILE="${TEMP_DIRECTORY}/${MATRIX_NAME}.tar.gz"

# Check if the matrix file already exists in the temporary directory
if [ ! -f "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.mtx" ]; then
    echo -e "${SCRIPT_TAG} Matrix file \"$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.mtx\" not found. Downloading..."
    # Download the matrix file using wget
    wget -O "$MATRIX_FILE" "https://suitesparse-collection-website.herokuapp.com/MM/${GROUP}/${MATRIX_NAME}.tar.gz"

    echo -e "${SCRIPT_TAG} Extracting file \"$MATRIX_FILE\". Contents:"
    # Extract the matrix file
    tar xzfv "$MATRIX_FILE" -C "$TEMP_DIRECTORY"
else
    echo -e "${SCRIPT_TAG} Matrix file already downloaded in $TEMP_DIRECTORY"
fi

# Run the pldi25-singularity/scripts/z_polyhedrator_manager.py
if [ ! -f "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.1d.uzp" ]; then
    python3 "$UZP_ARTIFACT_DIR/lib/scripts/z_polyhedrator_manager.py" "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.mtx" "$TEMP_DIRECTORY/${MATRIX_NAME}";
fi

# Tune the UZP file
if [ -f "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.1d.uzp" ]; then
    if [ ! -f "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.tuned.uzp" ]; then
            # Build if necessary
            echo -e "${SCRIPT_TAG} Building spf_aggregator..."
	    make -C "$UZP_ARTIFACT_DIR/uzp-tuners" spf_aggregator

            echo -e "${SCRIPT_TAG} Running spf_aggregator..."
            "$UZP_ARTIFACT_DIR/uzp-tuners/spf_aggregator" "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.1d.uzp" "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.tuned.uzp"
    fi
else
    echo -e "${ERROR_TAG} File \"$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.1d.uzp\" not found."
    exit 1
fi

# Generate headers for SPFv3
python3 uzp-genex/automatic_spf_loop_generator.py "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.tuned.uzp" uzp-genex/generated_shapes.h

echo -e "${SCRIPT_TAG} Building executable..."

if [ "$FLOAT_DATA_TYPE" == "float" ]; then
    CMD_DT="-float"
else
    CMD_DT=""
fi

if [ "${CACHE_MODE}" == "hot" ]; then
    KERNEL_REPS=100;
else
    KERNEL_REPS=1;
fi

if [ "${EXECUTION_THREADS}" == "1th" ];
then
    CMD="spf_matvec"
    OMP_NUM_THREADS="1"
    PIN_CORES="10"
elif [ "${EXECUTION_THREADS}" == "2th" ];
then
    CMD="spf_matvec_omp"
    OMP_NUM_THREADS="2"
    PIN_CORES="10,12"
else
    CMD="spf_matvec_omp"
    OMP_NUM_THREADS="8"
    PIN_CORES="10,12,14,1,2,4,6,8"
fi

# Recompile and launch
make -C uzp-genex -B ${CMD} OUTPUT_TYPE=GFLOPS KERNEL_REPS=${KERNEL_REPS}

# Check if hugectl is available
if command -v hugectl &> /dev/null; then
    HUGECTL_CMD="hugectl --heap"
else
    HUGECTL_CMD=""
fi

RUNCOUNT=10
i=0; while [[ $i -lt RUNCOUNT ]]; do
    if [ -n "$HUGECTL_CMD" ]; then
        OMP_NUM_THREADS=${OMP_NUM_THREADS} numactl -C${PIN_CORES} $HUGECTL_CMD ./uzp-genex/${CMD} $CMD_DT "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.tuned.uzp"
    else
        OMP_NUM_THREADS=${OMP_NUM_THREADS} numactl -C${PIN_CORES} ./uzp-genex/${CMD} $CMD_DT "$TEMP_DIRECTORY/${MATRIX_NAME}/${MATRIX_NAME}.tuned.uzp"
    fi
    i=$((i+1));
done
