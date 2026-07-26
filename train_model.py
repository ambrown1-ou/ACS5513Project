import argparse

from model import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the heart disease prediction model.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="inputs/Heart_disease_cleveland_new.csv",
        help="CSV file containing the feature columns and target",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument("--method", type=str, default="knn", choices=["knn", "naive_bayes", "svm"])
    args = parser.parse_args()

    result = train_model(
        args.data_path,
        method=args.method,
        test_size=args.test_size,
        random_state=args.random_state,
        n_neighbors=args.neighbors,
    )
    print(f"Model saved to {result['model_path']}")
    print(f"Method used: {result['method']}")
    print(f"Training rows: {result['rows']}")
    print(f"Validation accuracy: {result['accuracy']:.1%}")
    print(f"Validation precision: {result['precision']:.1%}")
    print(f"Validation recall: {result['recall']:.1%}")
    print(f"Validation F1 score: {result['f1']:.1%}")
