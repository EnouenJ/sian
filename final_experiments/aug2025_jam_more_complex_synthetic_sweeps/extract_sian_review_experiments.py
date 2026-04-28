import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import pandas as pd
from pathlib import Path

results_dir = "results"

all_results = []

for json_file in Path(results_dir).rglob("*.json"):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        result = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                result[key] = str(value)
            else:
                result[key] = value
                
        result['file_path'] = str(json_file)
        
        all_results.append(result)
        print(f"Processed file: {json_file}")
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

if not all_results:
    print("No .json files found in the specified directory or subdirectories.")
else:
    df = pd.DataFrame(all_results)
    
    output_path = "results_sian_review_experiments_hyperparam_sweep_2008.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Number of records saved: {len(df)}")