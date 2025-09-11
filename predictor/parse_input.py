import json
import argparse
from pathlib import Path
import re

def parse_input(avg_comparison_results_path, models_config_path, output_path):
    with open(avg_comparison_results_path, "r") as f:
        results = json.load(f)
        
    with open(models_config_path, "r") as f:
        models_config = json.load(f)
     
    # Prepare the labels   
    labels = []
    for l in results["labels"]:
        # Model name will be always like this *_model_MODELNAME*
        match = re.search(r"_model_([a-zA-Z0-9.-]+(?:-[0-9.]+[bk])?(?:-[a-zA-Z0-9]+)*)(?=(_|\.))", l)
        if match:
            model_name = match.group(1)
            labels.append(model_name)
            
    # Prepare the data
    data = results["data"]
    
    # Update kv_cache_profiling
    for i in range(len(data["kv_cache_profiling"])):
        data["kv_cache_profiling"][i] -= data["torch.compile"][i] + data["graph_compile_cached"][i]
        
    # Remove unecessary keys
    del data["model_loading"]
    del data["torch.compile"]
    del data["init_engine"]
    del data["actual_total_time"]
    
        
        
    output = {
        "labels": labels,
        "data": data,
        "models_config": models_config
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
        
    print(f"{output_path} has been created successfully!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse input JSON files.")
    parser.add_argument("--avg_comparison_results", help="Path to avg_comparison_results.json", type=Path, required=True)
    parser.add_argument("--models_config", help="Path to models_config.json", type=Path, required=True)
    parser.add_argument("--output_path", help="Path to write out predictor_input.json", type=Path, required=True)
    args = parser.parse_args()

    parse_input(args.avg_comparison_results, args.models_config, args.output_path)