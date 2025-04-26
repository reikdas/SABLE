#!/bin/bash
set -euo pipefail

# List of matrices
matrices=(
"eris1176"
"std1_Jac3"
"lp_wood1p"
"jendrec1"
"lowThrust_5"
"hangGlider_4"
"brainpc2"
"hangGlider_3"
"lowThrust_7"
"lowThrust_11"
"lowThrust_3"
"lowThrust_6"
"lowThrust_12"
"hangGlider_5"
"Journals"
"bloweybl"
"heart1"
"TSOPF_FS_b9_c6"
"Sieber"
"case9"
"c-30"
"c-32"
"freeFlyingRobot_10"
"freeFlyingRobot_11"
"freeFlyingRobot_12"
"lowThrust_10"
"lowThrust_13"
"lowThrust_4"
"lowThrust_8"
"lowThrust_9"
"lp_fit2p"
"nd12k"
"std1_Jac2"
"vsp_c-30_data_data"
)

# Create Suitesparse directory if it doesn't exist
mkdir -p Suitesparse

# Download each matrix
for matrix in "${matrices[@]}"; do
    echo "Downloading $matrix..."
    python3 -m ssgetpy -n "$matrix" -o Suitesparse/
done

echo "All downloads completed successfully."
