# Getting Started

1) Clone the repo:
```bash
git clone https://github.com/Huzifa1/vllm-profiling.git
cd vllm-profiling
```

2) Install vllm:
```bash
pip install vllm==0.9.2
```

3) Apply custom vllm changes (addition of profiling logs):
```bash
python3 apply_vllm_changes.py
```

4) Create `env.json` file. Currently there is an example in `examples/model_size/env.json`. The file show follow the following format:
```json
{
    "env": {
        "YOUR_CUSTOM_ENV_VAR": "CUSTOM_VALUE"
    },
    "params": {
        "vllm-flag1": "vllm-flag1-value",
        "vllm-flag2": ["vllm-flag2-value1", "vllm-flag2-value2"]
    }
}
```

When running vllm, all environment variables in `env` will be set, and all flags in `params` will be passed to vllm.

If the flag value is an array, vllm will run for every value. If there are multiple array values, vllm will run for all possible combinations

5) Run tests:

```bash
python3 run_test.py path/to/env.json
```

This will produce output logs in the same dir as `env.json` in addition to `comparison_results.json` file

5) Visualize comparison results:

Finally, you can use the `visualize.ipynb` jupyter notebook to visualize the results. All you have to do is change the path to `comparison_results.json` in the second cell.
