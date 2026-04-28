import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
from pathlib import Path

results_dir = "results"
combined_json = {}

for json_file in Path(results_dir).rglob("*.json"):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        combined_json[str(json_file)] = data
        
        print(f"Processed file: {json_file}")
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

if not combined_json:
    print("No .json files found in the specified directory or subdirectories.")
else:
    json_output_path = "results_sian_review_experiments_hyperparam_sweep_2008.json"
    with open(json_output_path, 'w') as f:
        json.dump(combined_json, f, indent=2)
    print(f"Combined JSON saved to {json_output_path}")
    print(f"Number of records saved: {len(combined_json)}")