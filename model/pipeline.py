import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "heart_disease_model.joblib"
REGISTRY_PATH = PROJECT_ROOT / "outputs" / "training_results.json"
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
SUPPORTED_METHODS = ("knn", "naive_bayes", "svm")
METHOD_LABELS = {
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}
FEATURE_RULES = {
    "age": {"minimum": 0, "maximum": 120},
    "sex": {"allowed": {0, 1}},
    "cp": {"allowed": {1, 2, 3, 4}},
    "trestbps": {"minimum": 0, "maximum": 300},
    "chol": {"minimum": 0, "maximum": 1000},
    "fbs": {"allowed": {0, 1}},
    "restecg": {"allowed": {0, 1, 2}},
    "thalach": {"minimum": 0, "maximum": 250},
    "exang": {"allowed": {0, 1}},
    "oldpeak": {"minimum": 0, "maximum": 10},
    "slope": {"allowed": {1, 2, 3}},
    "ca": {"allowed": {0, 1, 2, 3}},
    "thal": {"allowed": {3, 6, 7}},
}
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


# Loads a CSV, normalizes its target column, and enforces the feature domains.
def load_training_data(data_path):
    """Load and validate a CSV containing the model features and target."""
    data = pd.read_csv(data_path)
    
    # Handle different target column names
    possible_targets = ["target", "heart_disease_binary", "num"]
    actual_target = next((col for col in possible_targets if col in data.columns), None)
    
    if actual_target is None:
        raise ValueError(f"Missing target column (tried {possible_targets})")

    missing_fields = [field for field in FEATURE_FIELDS if field not in data.columns]
    if missing_fields:
        raise ValueError(f"Missing required columns: {', '.join(missing_fields)}")

    required_fields = FEATURE_FIELDS + [actual_target]
    data = data[required_fields].replace("?", pd.NA).dropna()
    if data.empty:
        raise ValueError("The uploaded dataset has no complete rows to train on.")

    for field in FEATURE_FIELDS:
        try:
            data[field] = pd.to_numeric(data[field], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Feature column '{field}' must contain numeric values.") from error

    try:
        target_values = pd.to_numeric(data[actual_target], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Target column '{actual_target}' must contain numeric values.") from error

    if actual_target == "num":
        data[TARGET_FIELD] = (target_values > 0).astype(int)
    else:
        if not target_values.isin({0, 1}).all():
            raise ValueError("The target column must contain only 0 and 1 values.")
        data[TARGET_FIELD] = target_values.astype(int)

    data = data[FEATURE_FIELDS + [TARGET_FIELD]]
    _validate_feature_frame(data[FEATURE_FIELDS])
    return data


# Validates a feature frame against the shared clinical input domains.
def _validate_feature_frame(data):
    for field, rule in FEATURE_RULES.items():
        values = pd.to_numeric(data[field], errors="coerce")
        finite = values.notna() & values.map(math.isfinite)
        if not finite.all():
            raise ValueError(f"Feature column '{field}' contains missing or non-finite values.")

        allowed = rule.get("allowed")
        if allowed is not None and not values.isin(allowed).all():
            raise ValueError(f"Feature column '{field}' contains values outside its allowed domain.")

        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        if minimum is not None and (values < minimum).any():
            raise ValueError(f"Feature column '{field}' contains values below {minimum}.")
        if maximum is not None and (values > maximum).any():
            raise ValueError(f"Feature column '{field}' contains values above {maximum}.")


# Validates one prediction payload and returns normalized numeric feature values.
def validate_feature_values(values):
    if not isinstance(values, Mapping):
        raise ValueError("Prediction values must be an object containing the clinical features.")

    missing_fields = [field for field in FEATURE_FIELDS if field not in values]
    if missing_fields:
        raise ValueError(f"Missing fields: {', '.join(missing_fields)}")

    unknown_fields = sorted(set(values) - set(FEATURE_FIELDS))
    if unknown_fields:
        raise ValueError(f"Unknown fields: {', '.join(unknown_fields)}")

    normalized = {}
    for field in FEATURE_FIELDS:
        try:
            normalized[field] = float(values[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Field '{field}' must be numeric.") from error

    _validate_feature_frame(pd.DataFrame([normalized], columns=FEATURE_FIELDS))
    return normalized


# Builds the estimator used for a supported training method.
def _build_estimator(method, random_state, n_neighbors):
    if method == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=n_neighbors),
        )
    if method == "naive_bayes":
        return GaussianNB()
    if method == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=random_state),
        )
    raise ValueError(f"Unknown training method: {method}")


# Converts a dataset or output label into a stable filename component.
def _slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def _model_display_name(dataset_key, method):
    dataset_label = str(dataset_key).replace("_", " ").replace("-", " ").title()
    method_label = METHOD_LABELS.get(method, str(method).replace("_", " ").title())
    return f"{dataset_label} - {method_label}"


# Creates a unique model ID for a training run.
def _build_model_id(dataset_key, method, output_name=None):
    prefix = _slug(output_name or f"{dataset_key}_{method}") or f"heart_disease_{method}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


# Returns an artifact path relative to the project root for the registry.
def _relative_artifact_path(path):
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


# Reads the model registry and converts the previous results format when needed.
def _read_training_registry():
    if not REGISTRY_PATH.exists():
        return {"active_model_id": None, "runs": []}

    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as registry_file:
            raw_registry = json.load(registry_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Training results could not be read.") from error

    if isinstance(raw_registry, dict) and isinstance(raw_registry.get("runs"), list):
        return raw_registry

    legacy_runs = []
    if isinstance(raw_registry, dict):
        for dataset_key, methods in raw_registry.items():
            if not isinstance(methods, dict):
                continue
            for method, metrics in methods.items():
                if not isinstance(metrics, dict):
                    continue
                model_path = metrics.get("model_path")
                model_id = Path(model_path).stem if model_path else _slug(f"{dataset_key}_{method}")
                legacy_runs.append({
                    **metrics,
                    "model_id": model_id,
                    "dataset_key": dataset_key,
                    "method": method,
                    "created_at": "",
                })

    active_model_id = next(
        (run["model_id"] for run in reversed(legacy_runs) if run.get("model_path")),
        None,
    )
    return {
        "active_model_id": active_model_id,
        "runs": legacy_runs,
        "_legacy": True,
    }


# Loads registered model runs for application views and API status checks.
def load_training_registry():
    registry = _read_training_registry()
    registry.pop("_legacy", None)
    return registry


# Atomically writes the model registry so interrupted runs do not corrupt results.
def _write_training_registry(registry):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = REGISTRY_PATH.with_name(f".{REGISTRY_PATH.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry, registry_file, indent=4)
            registry_file.write("\n")
        os.replace(temporary_path, REGISTRY_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# Registers one trained model and makes it the default API model.
def register_training_result(result, dataset_key=None):
    registry = _read_training_registry()
    registry.pop("_legacy", None)

    record = dict(result)
    record["dataset_key"] = dataset_key or result["dataset_key"]
    registry["runs"].append(record)
    registry["active_model_id"] = record["model_id"]
    _write_training_registry(registry)
    return record


# Returns successful registered models whose artifacts are still available.
def list_registered_models():
    models = []
    for record in reversed(load_training_registry().get("runs", [])):
        if record.get("error") or not record.get("model_path"):
            continue
        try:
            artifact_path = _resolve_model_path(record["model_path"])
        except (OSError, ValueError):
            continue
        if artifact_path.exists():
            models.append(record)
    return models


# Resolves a model artifact while keeping it inside the outputs directory.
def _resolve_model_path(model_path):
    output_root = MODEL_PATH.parent.resolve()
    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError as error:
        raise ValueError("Model artifacts must be stored in the outputs directory.") from error
    if candidate.suffix.lower() != ".joblib":
        raise ValueError("Model artifacts must use the .joblib format.")
    return candidate


# Finds a registered model by its opaque model ID.
def _find_model_record(model_id):
    for record in load_training_registry().get("runs", []):
        if record.get("model_id") == model_id:
            return record
    raise FileNotFoundError("The selected model is no longer registered.")


# Trains with stratified cross-validation, fits on all rows, and registers the artifact.
def train_model(
    data_path,
    method="knn",
    test_size=None,
    random_state=42,
    output_name=None,
    cv_folds=5,
    dataset_key=None,
    register=True,
    **kwargs,
):
    """Train, evaluate, and save a supervised learning model."""
    del test_size
    method = str(method).strip().lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown training method: {method}")
    if not isinstance(cv_folds, int) or isinstance(cv_folds, bool) or not 2 <= cv_folds <= 10:
        raise ValueError("Validation folds must be an integer from 2 to 10.")

    data = load_training_data(data_path)
    features = data[FEATURE_FIELDS]
    target = data[TARGET_FIELD]
    class_counts = target.value_counts()
    if target.nunique() < 2:
        raise ValueError("The target column must contain both 0 and 1 values.")
    if int(class_counts.min()) < cv_folds:
        raise ValueError("Each target class must contain at least as many rows as validation folds.")

    n_neighbors = kwargs.get("n_neighbors", 5)
    try:
        n_neighbors = int(n_neighbors)
    except (TypeError, ValueError) as error:
        raise ValueError("K neighbors must be an integer.") from error
    minimum_training_rows = len(data) - math.ceil(len(data) / cv_folds)
    if method == "knn" and not 1 <= n_neighbors <= minimum_training_rows:
        raise ValueError(f"K neighbors must be between 1 and {minimum_training_rows} for this dataset.")

    estimator = _build_estimator(method, random_state, n_neighbors)
    cross_validator = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    scores = cross_validate(
        estimator,
        features,
        target,
        cv=cross_validator,
        scoring=SCORING,
        error_score="raise",
    )

    result = {
        "dataset_key": dataset_key or Path(data_path).stem,
        "method": method,
        "rows": len(data),
        "cv_folds": cv_folds,
        "random_state": random_state,
    }
    for metric in METRIC_NAMES:
        metric_values = scores[f"test_{metric}"]
        result[metric] = float(metric_values.mean())
        result[f"{metric}_std"] = float(metric_values.std())

    estimator.fit(features, target)
    model_id = _build_model_id(result["dataset_key"], method, output_name)
    save_path = MODEL_PATH.parent / f"heart_disease_{model_id}.joblib"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, save_path)
    display_name = _model_display_name(result["dataset_key"], method)
    result.update({
        "model_id": model_id,
        "name": display_name,
        "display_name": display_name,
        "model_path": _relative_artifact_path(save_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    if register:
        register_training_result(result, dataset_key=result["dataset_key"])
    return result


# Loads a registered model or the legacy default artifact for compatibility.
def load_model(model_path=None, model_id=None):
    if model_path is not None and model_id is not None:
        raise ValueError("Choose a model ID or a model path, not both.")

    if model_id is not None:
        record = _find_model_record(model_id)
        model_path = record.get("model_path")
    elif model_path is None:
        registry = load_training_registry()
        active_model_id = registry.get("active_model_id")
        if active_model_id:
            try:
                model_path = _find_model_record(active_model_id).get("model_path")
            except FileNotFoundError:
                model_path = MODEL_PATH
        else:
            model_path = MODEL_PATH

    resolved_path = _resolve_model_path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError("No trained model exists yet. Train a model first.")
    return joblib.load(resolved_path)


# Reports whether the configured default model can be loaded.
def model_exists():
    try:
        load_model()
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


# Validates one patient, runs the model, and returns its predicted-class probability.
def predict(model, values):
    normalized_values = validate_feature_values(values)
    features = pd.DataFrame(
        [[normalized_values[field] for field in FEATURE_FIELDS]], columns=FEATURE_FIELDS
    )
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = [int(value) for value in model.classes_]
    probability = float(probabilities[classes.index(prediction)])
    return prediction, probability
