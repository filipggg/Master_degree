#!/bin/bash

# ugly...
in_file=$(mktemp)
csv_file=$(mktemp)
out_file=$(mktemp)
cat > $in_file

MAIN_CODE_DIR=./code/Master_degree/src
PYTHONPATH=${MAIN_CODE_DIR}:code:code/image_size:code/vision/references/detection $MAIN_CODE_DIR/predict.py $in_file $csv_file $out_file

cat $out_file
