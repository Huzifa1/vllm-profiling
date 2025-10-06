import re
import sys

def extract_time(line, pattern):
    """Extract time from a line using a regex pattern."""
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    return 0

def main(file_name):
    profiling_results = {
        "load_weights": None,
        "model_init": None,
        "model_loading": None,
        "dynamo_transform_time": None,
        "graph_compile_general_shape": None,
        "graph_compile_cached": None,
        "torch.compile": None,
        "kv_cache_profiling": None,
        "graph_capturing": None,
        "init_engine": None,
        "tokenizer_init": None,
        "total_time": 0,
        "actual_total_time": 0,
    }
    
    # init_engine includes: kv_cache profiling, kv_cache init, graph capturing
    # model_loading includes: load_weights, model_init, pre- and post-processing (not mentioned in the log)
    # dynamo_transform_time: time taken by TorchDynamo to transform the code to an intermediate representation
    # torch.compile: It includes graph_compile_general_shape and dynamo time
    # kv_cache_profiling: time taken to profile the KV cache. It includes torch.compile + graph_compile_cached
    
    with open(file_name, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "Loading weights took" in line:
            s = extract_time(line, r"Loading weights took ([\d.]+) seconds")
            profiling_results["load_weights"] = s

        elif "Model loading took" in line:
            s = extract_time(line, r"GiB and ([\d.]+) seconds")
            profiling_results["model_loading"] = s

        elif "Graph capturing finished" in line:
            s = extract_time(line, r"finished in ([\d.]+) secs")
            profiling_results["graph_capturing"] = s

        elif "init engine (profile, create kv cache, warmup model)" in line:
            s = extract_time(line, r"took ([\d.]+) seconds")
            profiling_results["init_engine"] = s

        elif "Dynamo bytecode transform time" in line:
            s = extract_time(line, r"time: ([\d.]+) s")
            profiling_results["dynamo_transform_time"] = s

        elif "Compiling a graph for" in line:
            s = extract_time(line, r"takes ([\d.]+) s")
            profiling_results["graph_compile_general_shape"] = s
            
        elif "Directly load the compiled" in line:
            s = extract_time(line, r"took ([\d.]+) s")
            profiling_results["graph_compile_cached"] = s

        elif "torch.compile takes" in line:
            s = extract_time(line, r"takes ([\d.]+) s in total")
            profiling_results["torch.compile"] = s
            
        elif "Memory profiling takes" in line:
            s = extract_time(line, r"takes ([\d.]+) seconds")
            profiling_results["kv_cache_profiling"] = s

        elif "Init Tokenizer took" in line:
            s = extract_time(line, r"took ([\d.]+) seconds")
            profiling_results["tokenizer_init"] = s

        elif "Initializing Model took" in line:
            s = extract_time(line, r"took ([\d.]+) seconds")
            profiling_results["model_init"] = s

        elif "initialize_engine took" in line:
            s = extract_time(line, r"initialize_engine took ([\d.]+) seconds")
            profiling_results["actual_total_time"] = s
            
    
    # Calculate total time
    total_time_keys = ["model_loading", "init_engine", "tokenizer_init"]
    total_time = 0
    for key in total_time_keys:
        if profiling_results[key] is not None:
            total_time += profiling_results[key]
    profiling_results["total_time"] = total_time        
    
    # In some cases (bitsandbytes qunantization), load_weights is not logged
    # Deduce it from model_loading and model_init when possible
    if profiling_results["load_weights"] is None:
        if profiling_results["model_loading"] is not None and profiling_results["model_init"] is not None:
            profiling_results["load_weights"] = profiling_results["model_loading"] - profiling_results["model_init"]
            
    return profiling_results  
                
                
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <vllm_log_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    profiling_results = main(file_path)
    
    print("Profiling Results:")
    for key, value in profiling_results.items():
        if value is None:
            print(f"{key}: None")  
        else:
            print(f"{key}: {value:.2f} seconds")  
            