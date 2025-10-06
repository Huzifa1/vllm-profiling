import joblib
import numpy as np
import json
import argparse
from pathlib import Path
import pickle

predictors = [
    {"model": "load_weights", "features": ["size"]},
    {"model": "dynamo_transform_time", "features": ["layers"]},
    {"model": "graph_compile_cached", "features": ["layers"]},
    {"model": "kv_cache_profiling", "features": ["size"]},
    {"model": "graph_capturing", "features": ["size", "batch_size"]},
    {"model": "tokenizer_init", "features": ["tokenizer_size"]},
]

def predict_total_time(size, layers, batch_size, tokenizer_size, models_path):
    total_time = 0
    for pred in predictors:
        # Load trained predictor
        model = joblib.load(f"{models_path}/{pred['model']}.pkl")
        
        target_metric = 1
        for f in pred["features"]:
            if f == "size":
                target_metric *= size
            elif f == "layers":
                target_metric *= layers
            elif f == "batch_size":
                target_metric *= batch_size
            elif f == "tokenizer_size":
                target_metric *= tokenizer_size
        
        current_pred = model.predict(np.array([[target_metric]]))[0]
        total_time += current_pred

    # Read constant time for model_init
    with open(f"{models_path}/constant_model_init.pkl", "rb") as f:
        model_init_time = pickle.load(f)
        
    return total_time + model_init_time

def predict(models_path, test_data_path):
    results = {
        "train": [],
        "validation": []
    }
    
    with open(test_data_path, "r") as f:
        test_data_dict = json.load(f)
        
    for split in ["train", "validation"]:
        print(f"\n\n{split}")
        test_data = test_data_dict[split]
        diff_sum = 0
        for t in test_data:
            pred = predict_total_time(t["size"], t["layers"], t["batch_size"], t["tokenizer_size"], models_path)
            diff = abs(pred - t["time"])
            diff_sum += diff
            
            results[split].append({
                "label": t["label"],
                "predicted": pred,
                "truth": t["time"],
                "diff": diff
            })
            print(f"{t['label']} | Predicted: {pred:.3f}s | Truth: {t['time']} | Diff: {diff:.2f}")

        print(f"Diff: {diff_sum:.2f} | Avg: {(diff_sum / len(test_data)):.2f}")
    
    return results
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictor")
    parser.add_argument("--models_path", help="Path to models dir", type=Path, required=True)
    parser.add_argument("--test_data_path", help="Path to test_data.json", type=Path, required=True)
    args = parser.parse_args()

    predict(args.models_path, args.test_data_path)