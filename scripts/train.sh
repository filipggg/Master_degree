#!/bin/bash

mkdir -p models/model_output
mkdir -p models/saved_models

MAIN_CODE_DIR=./code/Master_degree/src

PYTHONPATH=${MAIN_CODE_DIR}:code:code/image_size:code/vision/references/detection $MAIN_CODE_DIR/main.py
PYTHONPATH=${MAIN_CODE_DIR} $MAIN_CODE_DIR/dump_gonito_yaml.py

for t in dev-0 test-A
do
    ./predict.sh < $t/in.tsv > $t/out.tsv
done
