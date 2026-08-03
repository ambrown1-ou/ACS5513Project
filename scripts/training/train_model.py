import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model import FEATURE_FIELDS, SUPPORTED_METHODS, train_model
from config import paths as project_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the heart disease prediction model.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default=project_paths.BUNDLED_DATASETS_DIR / "heart_disease_cleveland_cleaned.csv",
        help="CSV file containing the feature columns and target",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument("--method", type=str, default="knn", choices=list(SUPPORTED_METHODS))
    parser.add_argument(
        "--feature-fields",
        type=str,
        default=",".join(FEATURE_FIELDS),
        help="Comma-separated canonical feature fields to train on",
    )
    parser.add_argument("--missing-strategy", type=str, default="drop", choices=["drop", "impute", "native"])
    args = parser.parse_args()

    kwargs = {}
    if args.method == "knn":
        kwargs["n_neighbors"] = args.neighbors
    feature_fields = [field.strip() for field in args.feature_fields.split(",") if field.strip()]

    result = train_model(
        args.data_path,
        method=args.method,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        feature_fields=feature_fields,
        missing_strategy=args.missing_strategy,
        dataset_key=Path(args.data_path).stem,
        **kwargs,
    )
    print(f"Model saved to {result['model_path']}")
    print(f"Method used: {result['method']}")
    print(f"Training rows: {result['rows']}")
    print(f"Cross-validation folds: {result['cv_folds']}")
    print(f"Cross-validation accuracy: {result['accuracy']:.1%} +/- {result['accuracy_std']:.1%}")
    print(f"Cross-validation precision: {result['precision']:.1%} +/- {result['precision_std']:.1%}")
    print(f"Cross-validation recall: {result['recall']:.1%} +/- {result['recall_std']:.1%}")
    print(f"Cross-validation F1 score: {result['f1']:.1%} +/- {result['f1_std']:.1%}")
