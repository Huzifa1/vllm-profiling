# Getting Started

1) Clone the repo:
```bash
git clone https://github.com/Huzifa1/vllm-profiling.git
cd vllm-profiling
```

2) Install vllm:
```bash
pip install vllm==0.10.1.1
```

3) Apply custom vllm changes (addition of profiling logs):
```bash
python3 apply_vllm_changes.py
```

4) Create `env.json` file. Currently there is an example in `examples/model_size/env.json`. The file should follow the following format:
```json
{
    "env": {
        "YOUR_CUSTOM_ENV_VAR": "CUSTOM_VALUE"
    },
    "default_params": {
        "gpu-memory-utilization": 0.9,
        "max-model-len": [4096, 8192]
    },
    "configs": [
        {"model": "meta-llama/Llama-2-7b-hf"},
        {"model": "Qwen/Qwen1.5-4B", "max-num-seqs": 32}
    ]
}
```

When running vllm, all environment variables in `env` will be set, and all params in `default_params` will be passed to vllm for each config.

If the `default_params` value is an array, vllm will run each config for every value. If there are multiple array values, vllm will run for all possible combinations.

In the previous example, the following commands will be run:
```bash
python3 engine.py --model "meta-llama/Llama-2-7b-hf" --gpu-memory-utilization 0.9 --max-model-len 4096
python3 engine.py --model "meta-llama/Llama-2-7b-hf" --gpu-memory-utilization 0.9 --max-model-len 8192
python3 engine.py --model "Qwen/Qwen1.5-4B" --max-num-seqs 32 --gpu-memory-utilization 0.9 --max-model-len 4096
python3 engine.py --model "Qwen/Qwen1.5-4B" --max-num-seqs 32 --gpu-memory-utilization 0.9 --max-model-len 8192
```

5) Run tests:

```bash
python3 run_test.py path/to/env.json
```

This will produce output logs in the same dir as `env.json` in addition to `comparison_results.json` file

5) Visualize comparison results:

Finally, you can use the `visualize.ipynb` jupyter notebook to visualize the results. All you have to do is change the path to `comparison_results.json` in the second cell.
