import json
import os
import subprocess
import shlex
import sys
from itertools import product

from utils import clear_cache


def is_array(val):
    return isinstance(val, list)

def build_command(base_params, combination, config_dir):
    args = []
    output_suffix = []

    for key, value in base_params.items():
        flag = f"--{key}"
        if key in combination:
            value = combination[key]
            val_str = str(value)
            # If value is a path, use only the basename
            if '/' in val_str:
                val_str = os.path.basename(val_str.rstrip("/"))
                
            output_suffix.append(f"{key}_{val_str.replace(' ', '_')}")
        if value is None:
            args.append(flag)
        elif isinstance(value, str):
            args.append(flag)
            args.extend(shlex.split(value))
        else:
            args.extend([flag, str(value)])

    command = ["python3", "engine.py"] + args
    if len(output_suffix) == 0:
        output_file = "output.txt"
    else:
        output_file = f"output_{'_'.join(output_suffix)}.txt"
    output_file = os.path.join(config_dir, output_file)
    full_cmd = f"{' '.join(shlex.quote(arg) for arg in command)} | tee {shlex.quote(output_file)}"
    return full_cmd

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    config_dir = os.path.dirname(os.path.abspath(config_file))
    with open(config_file, "r") as f:
        config = json.load(f)

    # Export env vars
    for k, v in config.get("env", {}).items():
        os.environ[k] = str(v)

    params = config.get("params", {})
    static_params = {k: v for k, v in params.items() if not is_array(v)}
    tuned_params = {k: v for k, v in params.items() if is_array(v)}

    if not tuned_params:
        # No arrays: run once
        cmd = build_command(static_params, {}, config_dir)
        clear_cache()
        print("Running:", cmd)
        subprocess.run(cmd, shell=True, check=True)
        return

    # Cartesian product of all array-valued params
    tuned_keys = list(tuned_params.keys())
    tuned_values = list(product(*[tuned_params[k] for k in tuned_keys]))

    for values in tuned_values:
        combination = dict(zip(tuned_keys, values))
        cmd = build_command({**static_params, **combination}, combination, config_dir)
        clear_cache()
        print("Running:", cmd)
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()
