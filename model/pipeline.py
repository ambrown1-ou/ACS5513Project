from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "heart_disease_model.joblib"
FEATURE_FIELDS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_FIELD = "target"


def load_training_data(data_path):
    """Load and validate a CSV containing the model features and target."""
    data = pd.read_csv(data_path)
    
    # Handle different target column names
    possible_targets = ["target", "heart_disease_binary", "num"]
    actual_target = next((col for col in possible_targets if col in data.columns), None)
    
    if not actual_target:
        raise ValueError(f"Missing target column (tried {possible_targets})")
    
    # If the column is 'num', it might have values 0-4. Map to binary if so.
    if actual_target == "num":
        data["target"] = (data["num"] > 0).astype(int)
    else:
        data["target"] = data[actual_target].astype(int)

    required_fields = FEATURE_FIELDS + ["target"]
    missing_fields = [field for field in FEATURE_FIELDS if field not in data.columns]
    if missing_fields:
        raise ValueError(f"Missing required columns: {', '.join(missing_fields)}")

    data = data[required_fields].replace("?", pd.NA).dropna()
    if data.empty:
        raise ValueError("The uploaded dataset has no complete rows to train on.")

    return data.apply(pd.to_numeric, errors="raise")


def train_model(data_path, method="random_forest", test_size=0.2, random_state=42, output_name=None, **kwargs):
    """Train, evaluate, and save a supervised learning model."""
    data = load_training_data(data_path)
    features = data[FEATURE_FIELDS]
    target = data["target"].astype(int)

    if target.nunique() < 2:
        raise ValueError("The target column must contain both 0 and 1 values.")

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    if method == "knn":
        model = KNeighborsClassifier(
            n_neighbors=kwargs.get("n_neighbors", 5),
        )
    elif method == "naive_bayes":
        model = GaussianNB()
    elif method == "svm":
        model = SVC(
            probability=True,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown training method: {method}")

    model.fit(train_features, train_target)
    
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_target, predictions)
    precision = precision_score(test_target, predictions)
    recall = recall_score(test_target, predictions)
    f1 = f1_score(test_target, predictions)

    if output_name:
        file_name = f"heart_disease_{output_name}.joblib"
    else:
        file_name = f"heart_disease_{method}.joblib"
    
    save_path = MODEL_PATH.parent / file_name
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "rows": len(data),
        "model_path": str(save_path),
        "method": method,
        "name": output_name or method
    }


def load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("No trained model exists yet. Train a model first.")
    return joblib.load(model_path)


def predict(model, values):
    """Return a class prediction and probability for one patient."""
    features = pd.DataFrame(
        [[values[field] for field in FEATURE_FIELDS]], columns=FEATURE_FIELDS
    )
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][prediction])
    return prediction, probability