import json
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import argparse
from pathlib import Path
import pickle

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
    
def create_predictors(predictor_input_path, models_output_dir):
    with open(predictor_input_path, "r") as f:
        data = json.load(f)
    
    key_metric_comb = [
        ("size", "load_weights"),
        ("layers", "dynamo_transform_time"),
        ("layers", "graph_compile_cached"),
        ("size", "kv_cache_profiling"),
        ("tokenizer_size", "tokenizer_init")
    ]
    for key_metric in key_metric_comb:
        create_predictor_wrt_key(key_metric[0], key_metric[1], data, models_output_dir)
    create_predictor_graph_capturing_time(data, models_output_dir)
    create_constant_predictor("model_init", data, models_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse input JSON files.")
    parser.add_argument("--predictor_input_path", help="Path to predictor_input.json", type=Path, required=True)
    parser.add_argument("--models_output_dir", help="Path to dump output models", type=Path, required=True)
    args = parser.parse_args()
    
    create_predictors(args.predictor_input_path, args.models_output_dir)    
    
    print("Predictor models have been created successfully!")