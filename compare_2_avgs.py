import sys
import json
from tabulate import tabulate

def extract_model_name(label):
    return label.split("model_")[1].split("_")[0].split(".txt")[0]

def read_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def compare_step(results1, results2, step):
    table_data = []
    sum_speedup = 0
    
    assert len(results1["labels"]) == len(results2["labels"])
    # Assuming the labels match
    for i, label in enumerate(results1["labels"]):
        model_name = extract_model_name(label)
    
        value1 = results1["data"][step][i]
        value2 = results2["data"][step][i]
        speedup = value2 / value1
        sum_speedup += speedup
        
        table_data.append([model_name, value1, value2, str(round(speedup, 2)) + "x"])
    
    print(f"\nComparing {step}")
    # print(tabulate(table_data, headers=["Model Name", "Value1", "Value2", "SpeedUp"], tablefmt="grid"))
    print(f"Avg Speedup: {(sum_speedup / len(results1['labels'])):.2f}x")
    # print("\n\n")

def compare_files(file_path1, file_path2):
    results1 = read_json_file(file_path1)
    results2 = read_json_file(file_path2)
    
    for step in ["load_weights", "model_init", "dynamo_transform_time", "graph_compile_cached", "graph_capturing", "tokenizer_init", "total_time"]:
        compare_step(results1, results2, step)
        
        
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <avg_comparison_results1.json> <avg_comparison_results2.json>")
        sys.exit(1)

    avg_comparison_results1_path = sys.argv[1]    
    avg_comparison_results2_path = sys.argv[2]
    
    
    compare_files(avg_comparison_results1_path, avg_comparison_results2_path)
    