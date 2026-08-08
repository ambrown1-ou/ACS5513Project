import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model import FEATURE_FIELDS, train_model
from config import paths as project_paths

DATASETS = {
    'heart_disease_cleveland_cleaned': project_paths.BUNDLED_DATASETS_DIR / 'heart_disease_cleveland_cleaned.csv',
}

METHODS = ("naive_bayes", "knn", "svm")


def train_all():
    for ds_name, ds_path in DATASETS.items():
        print(f"Processing dataset: {ds_name}...")
        
        for method in METHODS:
            print(f"  Training {method}...")
            try:
                kwargs = {}
                if method == "knn":
                    kwargs["n_neighbors"] = 5
                result = train_model(
                    ds_path,
                    method=method,
                    cv_folds=5,
                    random_state=42,
                    feature_fields=FEATURE_FIELDS,
                    missing_strategy="impute",
                    dataset_key=ds_name,
                    **kwargs,
                )
                print(f"    Model saved to {result['model_path']}")
            except Exception as e:
                print(f"    Error training {method} on {ds_name}: {e}")

    print(f"All models trained and registered in {project_paths.REGISTRY_PATH}")


if __name__ == "__main__":
    train_all()
