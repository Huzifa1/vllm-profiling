#!/bin/bash

EXP_DIR=batch_size
EXP_BRANCH=node5
DIR_PATH="experiments/$EXP_BRANCH/examples/$EXP_DIR"

run_test() {
    python3 run_test.py $DIR_PATH/env.json
}

mv_outputs() {
    mv $DIR_PATH/output* "$1"
    mv $DIR_PATH/comparison_results.json "$1"
}

# run_test
# mv_outputs "$DIR_PATH/uncached"

for i in 1 2 3 4 5; do
    run_test
    mv_outputs "$DIR_PATH/iterations/$i"
done

python avg_results.py $DIR_PATH/iterations
mv $DIR_PATH/iterations/avg_comparison_results.json $DIR_PATH/