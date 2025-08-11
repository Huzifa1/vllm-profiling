import joblib
import numpy as np
import json

predictors = [
    {"model": "load_weights", "features": ["size"]},
    {"model": "dynamo_transform_time", "features": ["layers"]},
    {"model": "graph_compile_cached", "features": ["layers"]},
    {"model": "kv_cache_profiling", "features": ["size"]},
    {"model": "graph_capturing", "features": ["size", "batch_size"]},
    {"model": "tokenizer_init", "features": ["tokenizer_size"]},
]

with open("./predictor_input.json", "r") as f:
    res = json.load(f)

def predict_total_time(size, layers, batch_size, tokenizer_size, i):
    total_time = 0
    for pred in predictors:
        # Load trained predictor
        model = joblib.load(f"./models/{pred['model']}.pkl")
        
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

    # Add 0.2s for model_init
    return total_time + 0.2



with open("test_data.json", "r") as f:
    test_data_dict = json.load(f)
    
for split in ["train", "validation"]:
    print(f"\n\n{split}")
    test_data = test_data_dict[split]
    diff_sum = 0
    for i, t in enumerate(test_data):
        pred = predict_total_time(t["size"], t["layers"], t["batch_size"], t["tokenizer_size"], i)
        diff = abs(pred - t["time"])
        diff_sum += diff
        print(f"Predicted ({t['label']}): {pred:.3f}s | Truth: {t['time']} | Diff: {diff}")

    print(f"Diff: {diff_sum} | Avg: {diff_sum / len(test_data)}")