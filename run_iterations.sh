#!/bin/bash

EXP_DIR=${1:-model_size}
EXP_BRANCH=${2:-node7}
IS_FIRST_TIME=${3:-false}

DIR_PATH="experiments/$EXP_BRANCH/examples/$EXP_DIR"

run_test() {
    python3 run_test.py $DIR_PATH/env.json
}

mv_outputs() {
    mv $DIR_PATH/output* "$1"
    mv $DIR_PATH/comparison_results.json "$1"
}

# is it first time?
if [ "$IS_FIRST_TIME" = true ] ; then
    run_test
    mv_outputs "$DIR_PATH/uncached"
fi

for i in 1 2 3 4 5; do
    run_test
    mv_outputs "$DIR_PATH/iterations/$i";
done

python avg_results.py $DIR_PATH/iterations