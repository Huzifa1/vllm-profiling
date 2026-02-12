export VLLM_LOGGING_CONFIG_PATH=$(realpath ../vllm_logging_config.json)

FIGURE=${1:-1}

if [ "$FIGURE" = "1" ]; then
    echo "hi"
elif [ "$FIGURE" = "2" ] || [ "$FIGURE" = "3-8" ] || [ "$FIGURE" = "9" ] || [ "$FIGURE" = "13" ] || [ "$FIGURE" = "14" ]; then
    FIGURE_DIR="figure$FIGURE"
    DIR_PATH="figures/$FIGURE_DIR"

    run_test() {
        python3 run_test.py $DIR_PATH/env.json
    }

    mv_outputs() {
        mv $DIR_PATH/output* "$1"
        mv $DIR_PATH/comparison_results.json "$1"
    }

    cd ..

    # Warmup run
    run_test
    mkdir -p "$DIR_PATH/uncached"
    mv_outputs "$DIR_PATH/uncached"

    for i in $(seq 1 5); do
        mkdir -p "$DIR_PATH/iterations/$i"
        run_test
        mv_outputs "$DIR_PATH/iterations/$i";
    done

    python3 avg_results.py $DIR_PATH/iterations

    cd -

    # Plot figure
    python3 figure$FIGURE/plot.py

elif [ "$FIGURE" = "10" ];then
    echo "hi"
elif [ "$FIGURE" = "11" ];then
    echo "hi"
elif [ "$FIGURE" = "12" ];then
    echo "hi"
elif [ "$FIGURE" = "16" ];then
    echo "hi"
else
    echo "Invalid figure number: $FIGURE. Please provide one of the following: '1', '2', '3-8', '9', '10', '11', '12', '13', '14', '16'."
    exit 1
fi