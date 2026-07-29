import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.pipeline import train_model
INPUTS_DIR = PROJECT_ROOT / 'inputs'

DATASETS = {
    'heart_disease_cleveland_cleaned': INPUTS_DIR / 'heart_disease_cleveland_cleaned.csv',
}

METHODS = ['knn', 'naive_bayes', 'svm']


def train_all():
    for ds_name, ds_path in DATASETS.items():
        print(f"Processing dataset: {ds_name}...")
        
        for method in METHODS:
            print(f"  Training {method}...")
            try:
                result = train_model(
                    ds_path,
                    method=method,
                    dataset_key=ds_name,
                    n_neighbors=5 # Default for KNN
                )
                print(f"    Model saved to {result['model_path']}")
            except Exception as e:
                print(f"    Error training {method} on {ds_name}: {e}")

    print("All models trained and registered in outputs/training_results.json")


if __name__ == "__main__":
    train_all()
