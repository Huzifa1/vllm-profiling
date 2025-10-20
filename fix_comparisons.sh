#!/bin/bash

EXP_DIR=${1:-model_size}
EXP_BRANCH=${2:-node7}

DIR_PATH="experiments/$EXP_BRANCH/examples/$EXP_DIR"

for i in 1 2 3 4 5; do
    python3 compare_logs.py $DIR_PATH/iterations/$i/output_model_* $DIR_PATH/iterations/$i/comparison_results.json;
done

python3 compare_logs.py $DIR_PATH/uncached/output_model_* $DIR_PATH/uncached/comparison_results.json;

python3 avg_results.py $DIR_PATH/iterations