#!/usr/bin/env bash
#
# uzp_prepare.sh - Prepare UZP files from a sparse .mtx file
#
# Borrowed from uzp_spmv.sh. Runs z_polyhedrator to generate .1d.uzp
# and spf_aggregator to create .tuned.uzp for a given Matrix Market file.
#
# Usage: uzp_prepare.sh <mtx_file> <output_dir>
#
#   mtx_file   - Path to a Matrix Market (.mtx) file containing the sparse part
#   output_dir - Directory where .1d.uzp and .tuned.uzp will be written
#
# Output files (named after the matrix basename without .mtx extension):
#   <output_dir>/<name>.1d.uzp
#   <output_dir>/<name>.tuned.uzp

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UZP_ARTIFACT_DIR="$SCRIPT_DIR/uzp-artifact"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <mtx_file> <output_dir>"
    exit 1
fi

MTX_FILE="$1"
OUTPUT_DIR="$2"

if [ ! -f "$MTX_FILE" ]; then
    echo "Error: MTX file not found: $MTX_FILE"
    exit 1
fi

MATRIX_NAME=$(basename "$MTX_FILE" .mtx)

mkdir -p "$OUTPUT_DIR"

# Step 1: Run z_polyhedrator to generate .1d.uzp
UZP_BASE="$OUTPUT_DIR/$MATRIX_NAME"
if [ ! -f "${UZP_BASE}.1d.uzp" ]; then
    python3 "$UZP_ARTIFACT_DIR/lib/scripts/z_polyhedrator_manager.py" \
        "$MTX_FILE" "$OUTPUT_DIR"
fi

if [ ! -f "${UZP_BASE}.1d.uzp" ]; then
    echo "Error: z_polyhedrator failed to generate ${UZP_BASE}.1d.uzp"
    exit 1
fi

# Step 2: Build spf_aggregator if needed, then tune
if [ ! -f "${UZP_BASE}.tuned.uzp" ]; then
    make -C "$UZP_ARTIFACT_DIR/uzp-tuners" \
        spf_aggregator \
        GCC="${CC:-gcc}" \
        GCC_OMP="${CC:-gcc} -fopenmp"
    "$UZP_ARTIFACT_DIR/uzp-tuners/spf_aggregator" \
        "${UZP_BASE}.1d.uzp" "${UZP_BASE}.tuned.uzp"
fi

if [ ! -f "${UZP_BASE}.tuned.uzp" ]; then
    echo "Error: spf_aggregator failed to generate ${UZP_BASE}.tuned.uzp"
    exit 1
fi
