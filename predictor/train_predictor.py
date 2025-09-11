import json
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import argparse
from pathlib import Path
import pickle
import re

def create_predictor_wrt_key(key, metric, data, models_output_dir):
    """Create predictor for key based on metric"""
    
    # Extract configs and timings
    configs = data["models_config"]
    timings = data["data"]
    labels = data["labels"]
    
    key_values = np.array([configs[label][key] for label in labels]).reshape(-1, 1)
    times = np.array(timings[metric])

    model = LinearRegression()
    model.fit(key_values, times)
    
    joblib.dump(model, f"{models_output_dir}/{metric}.pkl")

def create_predictor_graph_capturing_time(data, models_output_dir):
    """Create predictor for graph capturing given model size (in B) and batch size."""
    
    # Extract configs and timings
    configs = data["models_config"]
    timings = data["data"]
    labels = data["labels"]
    
    size_batch_product = np.array([
        configs[label]["size"] * 67 for label in labels
    ]).reshape(-1, 1)

    times_graph = np.array(timings["graph_capturing"])

    model_graph = LinearRegression()
    model_graph.fit(size_batch_product, times_graph)
    
    joblib.dump(model_graph, f"{models_output_dir}/graph_capturing.pkl")
    
def create_constant_predictor(key, data, models_output_dir):
    """Create constant predictor"""
    # Currently, this is just for model_init
    
    key_times = data["data"][key]
    avg = sum(key_times) / len(key_times)
    
    with open(f"{models_output_dir}/constant_{key}.pkl", "wb") as f:
        pickle.dump(avg, f)
    
def create_predictors(predictor_input, models_output_dir):
    
    key_metric_comb = [
        ("size", "load_weights"),
        ("layers", "dynamo_transform_time"),
        ("layers", "graph_compile_cached"),
        ("size", "kv_cache_profiling"),
        ("tokenizer_size", "tokenizer_init")
    ]
    for key_metric in key_metric_comb:
        create_predictor_wrt_key(key_metric[0], key_metric[1], predictor_input, models_output_dir)
    create_predictor_graph_capturing_time(predictor_input, models_output_dir)
    create_constant_predictor("model_init", predictor_input, models_output_dir)

def parse_input(avg_comparison_results_path, models_config_path):
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
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the predictors")
    parser.add_argument("--avg_comparison_results", help="Path to avg_comparison_results.json", type=Path, required=True)
    parser.add_argument("--models_config", help="Path to models_config.json", type=Path, required=True)
    parser.add_argument("--models_output_dir", help="Path to dump output models", type=Path, required=True)
    args = parser.parse_args()
    
    predictor_input = parse_input(args.avg_comparison_results, args.models_config)
    create_predictors(predictor_input, args.models_output_dir)    
    
    print("Predictor models have been created successfully!")