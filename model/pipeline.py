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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _normalize_optional_depth(value):
    if value in (None, 0, "0"):
        return None
    return value


def _build_knn_estimator(random_state, params):
    del random_state
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=params["n_neighbors"]),
    )


def _build_naive_bayes_estimator(random_state, params):
    del random_state, params
    return GaussianNB()


def _build_svm_estimator(random_state, params):
    del params
    return make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=random_state),
    )


def _build_decision_tree_estimator(random_state, params):
    return DecisionTreeClassifier(
        max_depth=_normalize_optional_depth(params["max_depth"]),
        min_samples_leaf=params["min_samples_leaf"],
        random_state=random_state,
    )


def _build_random_forest_estimator(random_state, params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=_normalize_optional_depth(params["max_depth"]),
        min_samples_leaf=params["min_samples_leaf"],
        random_state=random_state,
    )


def _build_hist_gradient_boosting_estimator(random_state, params):
    return HistGradientBoostingClassifier(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_depth=_normalize_optional_depth(params["max_depth"]),
        random_state=random_state,
    )


def _build_voting_estimator(random_state, params):
    del params
    return VotingClassifier(
        estimators=[
            ("knn", _build_estimator("knn", random_state, {"n_neighbors": 5})),
            ("naive_bayes", _build_estimator("naive_bayes", random_state, {})),
            ("svm", _build_estimator("svm", random_state, {})),
        ],
        voting="soft",
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import paths as project_paths

MODEL_ARTIFACTS_DIR = project_paths.MODEL_ARTIFACTS_DIR
MODEL_PATH = MODEL_ARTIFACTS_DIR / "heart_disease_model.joblib"
REGISTRY_PATH = project_paths.REGISTRY_PATH
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
MISSING_STRATEGIES = {
    "drop": {
        "label": "Drop incomplete rows",
        "description": "Matches the current training behavior by removing rows with missing features.",
    },
    "impute": {
        "label": "Median imputation",
        "description": "Imputes missing feature values inside the training pipeline before fitting the model.",
    },
    "native": {
        "label": "Native missing-value support",
        "description": "Uses estimators that can handle NaN values directly.",
    },
}
TARGET_FIELD = "target"
METHOD_SPECS = {
    "knn": {
        "label": "K-Nearest Neighbors",
        "needs_scaling": True,
        "supports_nan": False,
        "params": (
            {
                "name": "n_neighbors",
                "label": "K neighbors",
                "type": "int",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
                "step": 1,
                "help": "Number of neighboring rows to include in the vote.",
            },
        ),
        "build": _build_knn_estimator,
    },
    "naive_bayes": {
        "label": "Naive Bayes",
        "needs_scaling": False,
        "supports_nan": False,
        "params": (),
        "build": _build_naive_bayes_estimator,
    },
    "svm": {
        "label": "Support Vector Machine",
        "needs_scaling": True,
        "supports_nan": False,
        "params": (),
        "build": _build_svm_estimator,
    },
    "decision_tree": {
        "label": "Decision Tree",
        "needs_scaling": False,
        "supports_nan": False,
        "params": (
            {
                "name": "max_depth",
                "label": "Maximum depth",
                "type": "int",
                "default": 0,
                "minimum": 0,
                "maximum": 30,
                "step": 1,
                "help": "Use 0 for an unlimited tree depth.",
            },
            {
                "name": "min_samples_leaf",
                "label": "Minimum samples per leaf",
                "type": "int",
                "default": 1,
                "minimum": 1,
                "maximum": 50,
                "step": 1,
                "help": "Leaves must contain at least this many rows.",
            },
        ),
        "build": _build_decision_tree_estimator,
    },
    "random_forest": {
        "label": "Random Forest",
        "needs_scaling": False,
        "supports_nan": False,
        "params": (
            {
                "name": "n_estimators",
                "label": "Number of trees",
                "type": "int",
                "default": 200,
                "minimum": 10,
                "maximum": 500,
                "step": 1,
                "help": "Controls how many trees are averaged together.",
            },
            {
                "name": "max_depth",
                "label": "Maximum depth",
                "type": "int",
                "default": 0,
                "minimum": 0,
                "maximum": 30,
                "step": 1,
                "help": "Use 0 for an unlimited tree depth.",
            },
            {
                "name": "min_samples_leaf",
                "label": "Minimum samples per leaf",
                "type": "int",
                "default": 1,
                "minimum": 1,
                "maximum": 50,
                "step": 1,
                "help": "Leaves must contain at least this many rows.",
            },
        ),
        "build": _build_random_forest_estimator,
    },
    "hist_gradient_boosting": {
        "label": "Histogram Gradient Boosting",
        "needs_scaling": False,
        "supports_nan": True,
        "params": (
            {
                "name": "max_iter",
                "label": "Maximum iterations",
                "type": "int",
                "default": 200,
                "minimum": 10,
                "maximum": 500,
                "step": 1,
                "help": "Controls how many boosting rounds are fitted.",
            },
            {
                "name": "learning_rate",
                "label": "Learning rate",
                "type": "float",
                "default": 0.1,
                "minimum": 0.01,
                "maximum": 1.0,
                "step": 0.01,
                "help": "Smaller values slow learning but can improve generalization.",
            },
            {
                "name": "max_depth",
                "label": "Maximum depth",
                "type": "int",
                "default": 0,
                "minimum": 0,
                "maximum": 30,
                "step": 1,
                "help": "Use 0 for an unlimited tree depth.",
            },
        ),
        "build": _build_hist_gradient_boosting_estimator,
    },
    "voting": {
        "label": "Soft Voting Ensemble",
        "needs_scaling": False,
        "supports_nan": False,
        "params": (),
        "build": _build_voting_estimator,
    },
}
SUPPORTED_METHODS = tuple(METHOD_SPECS)
METHOD_LABELS = {method: spec["label"] for method, spec in METHOD_SPECS.items()}
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


def method_catalog():
    catalog = []
    for method in SUPPORTED_METHODS:
        spec = METHOD_SPECS[method]
        catalog.append(
            {
                "method": method,
                "value": method,
                "label": spec["label"],
                "needs_scaling": spec["needs_scaling"],
                "supports_nan": spec["supports_nan"],
                "params": [
                    {
                        **parameter,
                        **(
                            {"choices": list(parameter["choices"])}
                            if "choices" in parameter
                            else {}
                        ),
                    }
                    for parameter in spec["params"]
                ],
            }
        )
    return catalog


def missing_strategy_catalog():
    return [
        {
            "value": missing_strategy,
            "label": spec["label"],
            "description": spec["description"],
        }
        for missing_strategy, spec in MISSING_STRATEGIES.items()
    ]


def _resolve_method_params(method, values):
    spec = METHOD_SPECS[method]
    parameters = spec["params"]
    allowed_names = {parameter["name"] for parameter in parameters}
    unexpected_fields = sorted(set(values) - allowed_names)
    if unexpected_fields:
        raise ValueError(f"Unknown parameters for {method}: {', '.join(unexpected_fields)}")

    resolved = {}
    for parameter in parameters:
        name = parameter["name"]
        raw_value = values.get(name, parameter.get("default"))
        if raw_value in (None, ""):
            raw_value = parameter.get("default")

        parameter_type = parameter.get("type", "str")
        if parameter_type == "int":
            try:
                normalized_value = int(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{parameter['label']} must be an integer.") from error
        elif parameter_type == "float":
            try:
                normalized_value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{parameter['label']} must be a number.") from error
        elif parameter_type == "choice":
            choices = tuple(parameter.get("choices") or ())
            choice_lookup = {str(choice): choice for choice in choices}
            if raw_value in choices:
                normalized_value = raw_value
            elif str(raw_value) in choice_lookup:
                normalized_value = choice_lookup[str(raw_value)]
            else:
                raise ValueError(
                    f"{parameter['label']} must be one of: {', '.join(map(str, choices))}."
                )
        else:
            normalized_value = raw_value

        minimum = parameter.get("minimum")
        maximum = parameter.get("maximum")
        if minimum is not None and normalized_value < minimum:
            raise ValueError(f"{parameter['label']} must be at least {minimum}.")
        if maximum is not None and normalized_value > maximum:
            raise ValueError(f"{parameter['label']} must be at most {maximum}.")

        resolved[name] = normalized_value

    return resolved


def _normalize_feature_fields(feature_fields=None):
    fields = list(FEATURE_FIELDS if feature_fields is None else feature_fields)
    if not fields:
        raise ValueError("Select at least one feature field.")
    if any(not isinstance(field, str) or not field.strip() for field in fields):
        raise ValueError("Feature fields must be non-empty names.")
    if len(set(fields)) != len(fields):
        raise ValueError("Feature fields must not contain duplicates.")
    unknown_fields = [field for field in fields if field not in FEATURE_RULES]
    if unknown_fields:
        raise ValueError(f"Unknown feature fields: {', '.join(unknown_fields)}")
    return fields


# Loads a CSV, normalizes its target column, and enforces the feature domains.
def load_training_data(
    data_path,
    feature_fields=None,
    missing_strategy="drop",
    preserve_source=False,
    reject_missing=False,
):
    """Load and validate a CSV containing the model features and target."""
    feature_fields = _normalize_feature_fields(feature_fields)
    missing_strategy = str(missing_strategy).strip().lower()
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(f"Unknown missing-data strategy: {missing_strategy}")

    # Resolve path and fall back to bundled resources if the explicit path is missing.
    data_path = Path(data_path)
    if not data_path.exists():
        bundled = project_paths.BUNDLED_DATASETS_DIR / data_path.name
        if bundled.exists():
            data_path = bundled
        else:
            raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    if preserve_source and "source_row" not in data.columns:
        data["source_row"] = range(1, len(data) + 1)
    
    # Handle different target column names
    possible_targets = ["target", "heart_disease_binary", "num"]
    actual_target = next((col for col in possible_targets if col in data.columns), None)
    
    if actual_target is None:
        raise ValueError(f"Missing target column (tried {possible_targets})")

    missing_fields = [field for field in feature_fields if field not in data.columns]
    if missing_fields:
        raise ValueError(f"Missing required columns: {', '.join(missing_fields)}")

    required_fields = feature_fields + [actual_target]
    selected_fields = list(required_fields)
    if preserve_source and "source" in data.columns:
        selected_fields.append("source")
    if preserve_source and "source_row" in data.columns:
        selected_fields.append("source_row")

    data = data[selected_fields].replace("?", pd.NA)
    if reject_missing:
        missing_feature_fields = [field for field in feature_fields if data[field].isna().any()]
        if missing_feature_fields:
            raise ValueError(
                "KNN cannot train with missing feature values: "
                f"{', '.join(missing_feature_fields)}."
            )
    if missing_strategy == "drop":
        data = data.dropna()
    else:
        drop_subset = [actual_target]
        if preserve_source and "source" in data.columns:
            drop_subset.append("source")
        if preserve_source and "source_row" in data.columns:
            drop_subset.append("source_row")
        data = data.dropna(subset=drop_subset)
    if data.empty:
        raise ValueError("The uploaded dataset has no complete rows to train on.")

    for field in feature_fields:
        # Keep strict behavior for the "drop" strategy: require numeric values.
        if missing_strategy == "drop":
            try:
                data[field] = pd.to_numeric(data[field], errors="raise")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Feature column '{field}' must contain numeric values.") from error
        else:
            # For "impute" and "native" strategies, coerce non-numeric values to NaN
            # so downstream imputation or native-NaN-support estimators can handle them.
            data[field] = pd.to_numeric(data[field], errors="coerce")
            # When using median imputation, also treat domain-violating numeric values
            # as missing so the imputer can fill them rather than having validation fail.
            if missing_strategy == "impute":
                rule = FEATURE_RULES.get(field, {})
                allowed = rule.get("allowed")
                if allowed is not None:
                    mask_invalid_allowed = ~data[field].isin(allowed) & data[field].notna()
                    if mask_invalid_allowed.any():
                        data.loc[mask_invalid_allowed, field] = pd.NA
                minimum = rule.get("minimum")
                if minimum is not None:
                    mask_below = (data[field] < minimum) & data[field].notna()
                    if mask_below.any():
                        data.loc[mask_below, field] = pd.NA
                maximum = rule.get("maximum")
                if maximum is not None:
                    mask_above = (data[field] > maximum) & data[field].notna()
                    if mask_above.any():
                        data.loc[mask_above, field] = pd.NA

    if reject_missing:
        missing_feature_fields = [field for field in feature_fields if data[field].isna().any()]
        if missing_feature_fields:
            raise ValueError(
                "KNN cannot train with missing feature values: "
                f"{', '.join(missing_feature_fields)}."
            )

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

    output_fields = feature_fields + [TARGET_FIELD]
    if preserve_source and "source" in data.columns:
        output_fields.append("source")
    if preserve_source and "source_row" in data.columns:
        output_fields.append("source_row")
    data = data[output_fields]
    _validate_feature_frame(data[feature_fields], allow_missing=missing_strategy != "drop")
    return data


def prepare_training_data(data_path, output_path, feature_fields=None, missing_strategy="drop"):
    """Materialize a validated dataset configuration for later model training."""
    feature_fields = _normalize_feature_fields(feature_fields)
    missing_strategy = str(missing_strategy).strip().lower()
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(f"Unknown missing-data strategy: {missing_strategy}")

    data = load_training_data(
        data_path,
        feature_fields=feature_fields,
        missing_strategy=missing_strategy,
        preserve_source=True,
    )
    class_counts = data[TARGET_FIELD].value_counts()
    if data.empty or data[TARGET_FIELD].nunique() < 2:
        remaining_classes = {int(key): int(value) for key, value in class_counts.to_dict().items()}
        raise ValueError(
            f"The selected fields with '{missing_strategy}' leave an unusable target "
            f"after preparation ({remaining_classes}). Choose another preparation strategy."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = feature_fields + [TARGET_FIELD]
    if "source" in data.columns:
        output_fields.append("source")
    if "source_row" in data.columns:
        output_fields.append("source_row")
    data[output_fields].to_csv(output_path, index=False)
    return {
        "path": output_path,
        "feature_fields": feature_fields,
        "missing_strategy": missing_strategy,
        "rows": int(len(data)),
        "training_row_ids": [int(value) for value in data["source_row"].tolist()] if "source_row" in data.columns else [],
        "class_counts": {int(key): int(value) for key, value in class_counts.to_dict().items()},
    }


# Validates a feature frame against the shared clinical input domains.
def _validate_feature_frame(data, allow_missing=False):
    for field in data.columns:
        rule = FEATURE_RULES.get(field)
        if rule is None:
            raise ValueError(f"Unknown feature column '{field}'.")

        values = pd.to_numeric(data[field], errors="coerce")
        if not allow_missing and values.isna().any():
            raise ValueError(f"Feature column '{field}' contains missing or non-finite values.")

        non_missing = values.dropna()
        finite = non_missing.map(math.isfinite)
        if not finite.all():
            raise ValueError(f"Feature column '{field}' contains missing or non-finite values.")

        allowed = rule.get("allowed")
        if allowed is not None and not non_missing.isin(allowed).all():
            raise ValueError(f"Feature column '{field}' contains values outside its allowed domain.")

        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        if minimum is not None and (non_missing < minimum).any():
            raise ValueError(f"Feature column '{field}' contains values below {minimum}.")
        if maximum is not None and (non_missing > maximum).any():
            raise ValueError(f"Feature column '{field}' contains values above {maximum}.")


# Validates one prediction payload and returns normalized numeric feature values.
def validate_feature_values(values, feature_fields=None):
    feature_fields = list(feature_fields or FEATURE_FIELDS)
    if not isinstance(values, Mapping):
        raise ValueError("Prediction values must be an object containing the clinical features.")

    missing_fields = [field for field in feature_fields if field not in values]
    if missing_fields:
        raise ValueError(f"Missing fields: {', '.join(missing_fields)}")

    unknown_fields = sorted(set(values) - set(feature_fields))
    if unknown_fields:
        raise ValueError(f"Unknown fields: {', '.join(unknown_fields)}")

    normalized = {}
    for field in feature_fields:
        try:
            normalized[field] = float(values[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Field '{field}' must be numeric.") from error

    _validate_feature_frame(pd.DataFrame([normalized], columns=feature_fields))
    return normalized


# Builds the estimator used for a supported training method.
def _build_estimator(method, random_state, params):
    try:
        build_estimator = METHOD_SPECS[method]["build"]
    except KeyError as error:
        raise ValueError(f"Unknown training method: {method}") from error
    return build_estimator(random_state, params)


def _apply_missing_data_strategy(estimator, missing_strategy):
    if missing_strategy == "drop" or missing_strategy == "native":
        return estimator
    if missing_strategy == "impute":
        return make_pipeline(SimpleImputer(strategy="median"), estimator)
    raise ValueError(f"Unknown missing-data strategy: {missing_strategy}")


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


def delete_registered_model(model_id):
    """Remove a registered model run and its artifact, if present."""
    registry = _read_training_registry()
    registry.pop("_legacy", None)
    runs = registry.get("runs", [])
    target = next((run for run in runs if run.get("model_id") == model_id), None)
    if target is None:
        raise FileNotFoundError("The selected model is no longer registered.")

    if target.get("model_path"):
        artifact_path = _resolve_model_path(target["model_path"])
        if artifact_path.exists():
            artifact_path.unlink()

    remaining_runs = [run for run in runs if run.get("model_id") != model_id]
    registry["runs"] = remaining_runs
    if registry.get("active_model_id") == model_id:
        registry["active_model_id"] = None
        for run in reversed(remaining_runs):
            if run.get("error") or not run.get("model_path"):
                continue
            try:
                candidate_path = _resolve_model_path(run["model_path"])
            except (OSError, ValueError):
                continue
            if candidate_path.exists():
                registry["active_model_id"] = run.get("model_id")
                break

    _write_training_registry(registry)
    return target


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
    artifact_root = MODEL_ARTIFACTS_DIR.resolve()
    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("Model artifacts must be stored in the configured model artifacts directory.") from error
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
    feature_fields=None,
    missing_strategy="drop",
    dataset_key=None,
    register=True,
    **kwargs,
):
    """Train, evaluate, and save a supervised learning model."""
    del test_size
    method = str(method).strip().lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown training method: {method}")
    feature_fields = _normalize_feature_fields(feature_fields)
    missing_strategy = str(missing_strategy).strip().lower()
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(f"Unknown missing-data strategy: {missing_strategy}")
    if missing_strategy == "native" and not METHOD_SPECS[method]["supports_nan"]:
        raise ValueError("Native missing-value support is only available for methods that support NaN values.")
    if not isinstance(cv_folds, int) or isinstance(cv_folds, bool) or not 2 <= cv_folds <= 10:
        raise ValueError("Validation folds must be an integer from 2 to 10.")

    data = load_training_data(
        data_path,
        feature_fields=feature_fields,
        missing_strategy=missing_strategy,
        preserve_source=True,
    )
    features = data[feature_fields]
    target = data[TARGET_FIELD]
    groups = data["source"] if "source" in data.columns else None
    group_count = int(groups.nunique(dropna=True)) if groups is not None else 0
    class_counts = target.value_counts()
    if target.nunique() < 2:
        remaining_classes = {int(key): int(value) for key, value in class_counts.to_dict().items()}
        raise ValueError(
            f"The selected fields with '{missing_strategy}' leave only one target class "
            f"after row filtering ({remaining_classes}). Choose imputation or another field selection."
        )
    if int(class_counts.min()) < cv_folds:
        raise ValueError("Each target class must contain at least as many rows as validation folds.")

    method_params = _resolve_method_params(method, kwargs)
    minimum_training_rows = len(data) - math.ceil(len(data) / cv_folds)
    n_neighbors = method_params.get("n_neighbors", 5)
    if method == "knn" and not 1 <= n_neighbors <= minimum_training_rows:
        raise ValueError(f"K neighbors must be between 1 and {minimum_training_rows} for this dataset.")

    estimator = _build_estimator(method, random_state, method_params)
    estimator = _apply_missing_data_strategy(estimator, missing_strategy)
    if groups is not None and group_count >= 2 and group_count >= cv_folds:
        cross_validator = StratifiedGroupKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
        cv_strategy = "stratified_group"
        cv_note = ""
        scores = cross_validate(
            estimator,
            features,
            target,
            groups=groups,
            cv=cross_validator,
            scoring=SCORING,
            error_score="raise",
        )
    else:
        cross_validator = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
        cv_strategy = "stratified"
        if groups is None:
            cv_note = "No source column was available, so stratified cross-validation was used."
        elif group_count < 2:
            cv_note = "Only one source value was available, so stratified cross-validation was used."
        else:
            cv_note = f"Only {group_count} source groups were available for {cv_folds} folds, so stratified cross-validation was used."
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
        "feature_fields": feature_fields,
        "missing_strategy": missing_strategy,
        "group_count": group_count,
        "cv_strategy": cv_strategy,
        "cv_note": cv_note,
        "rows": len(data),
        "training_row_count": len(data),
        "training_row_ids": [int(value) for value in data["source_row"].tolist()] if "source_row" in data.columns else [],
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
def predict(model, values, feature_fields=None):
    feature_fields = list(feature_fields or FEATURE_FIELDS)
    normalized_values = validate_feature_values(
        {field: values[field] for field in feature_fields},
        feature_fields=feature_fields,
    )
    features = pd.DataFrame(
        [[normalized_values[field] for field in feature_fields]], columns=feature_fields
    )
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = [int(value) for value in model.classes_]
    probability = float(probabilities[classes.index(prediction)])
    return prediction, probability


def _unwrap_final_estimator(estimator):
    current = estimator
    while hasattr(current, "steps") and getattr(current, "steps", None):
        current = current.steps[-1][1]
    return current


def extract_feature_importances(model, feature_fields):
    estimator = _unwrap_final_estimator(model)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None

    values = [float(value) for value in importances]
    if len(values) != len(feature_fields):
        return None
    return dict(zip(feature_fields, values))


def inspect_training_data(data_path, cv_folds=5, feature_fields=None):
    """Inspect a CSV dataset and return a JSON-serializable summary useful for UI validation.

    The inspection re-uses the same semantics as :func:`load_training_data` for target
    detection and missing-value handling but does not raise for feature-level issues;
    instead it reports counts so the UI can provide guidance before training.
    """
    # Validate folds similar to train_model
    if not isinstance(cv_folds, int) or isinstance(cv_folds, bool) or not 2 <= cv_folds <= 10:
        raise ValueError("Validation folds must be an integer from 2 to 10.")
    feature_fields = _normalize_feature_fields(feature_fields)

    data_path = Path(data_path)
    dataset_key = data_path.stem
    filename = data_path.name

    try:
        raw = pd.read_csv(data_path)
    except Exception as error:
        raise FileNotFoundError("The dataset could not be read.") from error

    possible_targets = ["target", "heart_disease_binary", "num"]
    actual_target = next((col for col in possible_targets if col in raw.columns), None)
    if actual_target is None:
        raise ValueError(f"Missing target column (tried {possible_targets})")

    total_rows = int(len(raw))
    df = raw.replace("?", pd.NA)

    # Target summary
    target_present = int(df[actual_target].notna().sum())
    target_missing = int(total_rows - target_present)

    target_numeric = pd.to_numeric(df[actual_target], errors="coerce")
    invalid_target_non_numeric = int((~df[actual_target].isna() & target_numeric.isna()).sum())

    if actual_target == "num":
        normalized = (target_numeric > 0).dropna().astype(int)
        class_counts = {int(k): int(v) for k, v in normalized.value_counts().to_dict().items()}
        normalized_target = (target_numeric > 0).where(target_numeric.notna())
        invalid_target_non_binary = 0
    else:
        # Count non-binary numeric values as invalid for binary targets
        non_binary_mask = ~target_numeric.isna() & ~target_numeric.isin({0, 1})
        invalid_target_non_binary = int(non_binary_mask.sum())
        class_counts = {int(k): int(v) for k, v in target_numeric.dropna().astype(int).value_counts().to_dict().items()} if not target_numeric.dropna().empty else {}
        normalized_target = target_numeric.where(target_numeric.isin({0, 1}))

    # Per-field diagnostics
    fields = []
    for field in FEATURE_FIELDS:
        present = field in df.columns
        if not present:
            fields.append({"name": field, "present": False, "missing_count": None, "invalid_count": None, "valid_count": None})
            continue

        total = int(len(df))
        missing_count = int(df[field].isna().sum())
        numeric = pd.to_numeric(df[field], errors="coerce")
        non_numeric_invalid = int((~df[field].isna() & numeric.isna()).sum())

        non_missing_values = numeric.dropna()
        non_finite_count = int((~non_missing_values.map(math.isfinite)).sum()) if not non_missing_values.empty else 0

        rule = FEATURE_RULES.get(field, {})
        domain_invalid = 0
        allowed = rule.get("allowed")
        if allowed is not None and not non_missing_values.empty:
            domain_invalid += int((~non_missing_values.isin(allowed)).sum())
        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        if minimum is not None and not non_missing_values.empty:
            domain_invalid += int((non_missing_values < minimum).sum())
        if maximum is not None and not non_missing_values.empty:
            domain_invalid += int((non_missing_values > maximum).sum())

        invalid_total = non_numeric_invalid + non_finite_count + domain_invalid
        valid_count = int(total - missing_count - invalid_total)

        fields.append({
            "name": field,
            "present": True,
            "missing_count": int(missing_count),
            "invalid_count": int(invalid_total),
            "valid_count": int(valid_count),
        })

    required = list(feature_fields)
    missing_required = [field for field in required if field not in df.columns]
    if missing_required:
        selection_summary = {
            "feature_fields": required,
            "target_field": actual_target,
            "available": False,
            "missing_fields": missing_required,
            "rows_by_strategy": None,
            "class_counts_by_strategy": None,
            "folds_supported_by_strategy": None,
        }
    else:
        valid_target_mask = normalized_target.notna()
        drop_mask = df[required].notna().all(axis=1) & valid_target_mask
        drop_rows = int(drop_mask.sum())
        keep_if_target_present = int(valid_target_mask.sum())
        strategy_masks = {
            "drop": drop_mask,
            "impute": valid_target_mask,
            "native": valid_target_mask,
        }
        class_counts_by_strategy = {}
        folds_supported_by_strategy = {}
        for strategy, strategy_mask in strategy_masks.items():
            strategy_counts = {
                int(key): int(value)
                for key, value in normalized_target[strategy_mask].value_counts().to_dict().items()
            }
            class_counts_by_strategy[strategy] = strategy_counts
            folds_supported_by_strategy[strategy] = bool(
                len(strategy_counts) >= 2 and min(strategy_counts.values()) >= cv_folds
            )

        selection_summary = {
            "feature_fields": required,
            "target_field": actual_target,
            "available": True,
            "missing_fields": [],
            "rows_by_strategy": {
                "drop": drop_rows,
                "impute": keep_if_target_present,
                "native": keep_if_target_present,
            },
            "class_counts_by_strategy": class_counts_by_strategy,
            "folds_supported_by_strategy": folds_supported_by_strategy,
        }

    # Cross-validation suitability info
    groups_present = "source" in df.columns
    group_count = int(df["source"].nunique(dropna=True)) if groups_present else 0

    # choose strategy using the same rules as train_model
    if groups_present and group_count >= 2 and group_count >= cv_folds:
        cv_strategy = "stratified_group"
        cv_note = ""
    else:
        cv_strategy = "stratified"
        if not groups_present:
            cv_note = "No source column was available, so stratified cross-validation was used."
        elif group_count < 2:
            cv_note = "Only one source value was available, so stratified cross-validation was used."
        else:
            cv_note = f"Only {group_count} source groups were available for {cv_folds} folds, so stratified cross-validation was used."

    # Determine whether class counts are sufficient for requested folds
    class_count_min = None
    if class_counts:
        class_count_min = min(class_counts.values())
    folds_supported = bool(class_count_min is not None and int(class_count_min) >= cv_folds)

    issues = []
    if invalid_target_non_numeric:
        issues.append(f"{invalid_target_non_numeric} target values could not be parsed as numbers.")
    if invalid_target_non_binary:
        issues.append(f"{invalid_target_non_binary} target values are not binary 0/1.")
    # report any feature-level problems
    for f in fields:
        if f.get("present") and f.get("invalid_count") and f.get("invalid_count") > 0:
            issues.append(f"{f['invalid_count']} invalid values in field '{f['name']}'.")

    return {
        "dataset_key": dataset_key,
        "filename": filename,
        "total_rows": total_rows,
        "target": {
            "column": actual_target,
            "present": int(target_present),
            "missing_count": int(target_missing),
            "invalid_non_numeric": int(invalid_target_non_numeric),
            "invalid_non_binary": int(invalid_target_non_binary),
            "class_counts": class_counts,
        },
        "fields": fields,
        "selection": selection_summary,
        "cv": {
            "requested_folds": cv_folds,
            "group_count": group_count,
            "cv_strategy": cv_strategy,
            "cv_note": cv_note,
            "folds_supported": bool(folds_supported),
            "class_counts": class_counts,
        },
        "issues": issues,
    }
