import json
from pathlib import Path
from model.pipeline import train_model

PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_FILE = OUTPUTS_DIR / "training_results.json"

DATASETS = {
    "cleveland": INPUTS_DIR / "Heart_disease_cleveland_new.csv",
    "cleveland_cleaned": INPUTS_DIR / "heart_disease_cleveland_cleaned.csv",
    "hungarian": INPUTS_DIR / "heart_disease_hungarian_cleaned.csv",
    "switzerland": INPUTS_DIR / "heart_disease_switzerland_cleaned.csv",
    "long_beach": INPUTS_DIR / "heart_disease_long_beach_cleaned.csv",
    "combined": INPUTS_DIR / "heart_disease_cleaned_combined.csv",
}

METHODS = ["knn", "naive_bayes", "svm"]

def train_all():
    all_results = {}
    
    for ds_name, ds_path in DATASETS.items():
        print(f"Processing dataset: {ds_name}...")
        all_results[ds_name] = {}
        
        for method in METHODS:
            print(f"  Training {method}...")
            try:
                # Use a specific output name to keep them unique
                output_name = f"{ds_name}_{method}"
                result = train_model(
                    ds_path,
                    method=method,
                    output_name=output_name,
                    n_neighbors=5 # Default for KNN
                )
                all_results[ds_name][method] = result
            except Exception as e:
                print(f"    Error training {method} on {ds_name}: {e}")
                all_results[ds_name][method] = {"error": str(e)}

    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"All models trained. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    train_all()
