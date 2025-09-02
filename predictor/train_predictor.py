import json
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Load data
with open("predictor_input.json", "r") as f:
    data = json.load(f)

# Extract configs and timings
configs = data["models_config"]
timings = data["data"]
labels = data["labels"]

models_output_dir = "./models"

def predict_metric_wrt_key(key, metric):
    """predict key based on metric"""
    
    key_values = np.array([configs[label][key] for label in labels]).reshape(-1, 1)
    times = np.array(timings[metric])

    model = LinearRegression()
    model.fit(key_values, times)
    
    joblib.dump(model, f"{models_output_dir}/{metric}.pkl")

def predict_graph_capturing_time():
    """Predict time for graph capturing given model size (in B) and batch size."""
    
    size_batch_product = np.array([
        configs[label]["size"] * 67 for label in labels
    ]).reshape(-1, 1)

    times_graph = np.array(timings["graph_capturing"])

    model_graph = LinearRegression()
    model_graph.fit(size_batch_product, times_graph)
    
    joblib.dump(model_graph, f"{models_output_dir}/graph_capturing.pkl")

if __name__ == "__main__":

    predict_metric_wrt_key("size", "load_weights")
    predict_metric_wrt_key("layers", "dynamo_transform_time")
    predict_metric_wrt_key("layers", "graph_compile_cached")
    predict_metric_wrt_key("size", "kv_cache_profiling")
    predict_metric_wrt_key("tokenizer_size", "tokenizer_init")
    predict_graph_capturing_time()
    
    print("Predictor models have been created successfully!")