import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from model import (
    delete_registered_model,
    extract_feature_importances,
    inspect_training_data,
    list_registered_models,
    load_model,
    load_training_registry,
    missing_strategy_catalog,
    method_catalog,
    model_exists,
    prepare_training_data,
    predict,
    train_model,
    validate_feature_values,
    FEATURE_FIELDS,
)
from model.dataset_mapping import analyze_dataset, apply_mapping, identity_mapping, review_mapping
from model.schema import SCHEMA_ID, get_schema, schema_catalog
from model.visualization import (
    create_correlation_matrix_plot,
    create_correlation_plot,
    create_distribution_plot,
    create_feature_importance_plot,
    create_strongest_correlations_plot,
)
from config import paths as project_paths


api = Blueprint("api", __name__, url_prefix="/api")

# Configuration (mirrors routes.py setup)
BUNDLED_DATASETS_DIR = project_paths.BUNDLED_DATASETS_DIR
INPUTS_DIR = project_paths.UPLOADED_DATASETS_DIR
MAPPED_DATASETS_DIR = project_paths.MAPPED_DATASETS_DIR
PREPARED_DATASETS_DIR = project_paths.PREPARED_DATASETS_DIR
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFO_PATH = project_paths.DATASET_METADATA_PATH
VALIDATION_MODES = {"NORMAL", "NO_TEST"}
INTAKE_STATUSES = {"mapping", "review", "ready", "trusted", "legacy"}
BUNDLED_DATASET_KEY = "heart_disease_cleveland_cleaned"
SUPPORTED_APP_METHODS = ("naive_bayes", "knn", "svm")


# ============================================================================
# Helper Functions (Moved from routes.py)
# ============================================================================

def _read_metadata_file(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
    except (OSError, ValueError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _read_dataset_metadata():
    """Read bundled defaults and overlay runtime dataset configuration."""
    bundled_info_path = BUNDLED_DATASETS_DIR / "datasets_info.json"
    metadata = {}
    if bundled_info_path != INFO_PATH:
        metadata.update(_read_metadata_file(bundled_info_path))
    metadata.update(_read_metadata_file(INFO_PATH))
    return metadata


def _read_csv_metadata(path):
    with Path(path).open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.reader(file_handle)
        source_columns = [str(column) for column in next(reader, [])]
        source_row_ids = []
        for row in reader:
            if not row:
                continue
            source_row_ids.append(len(source_row_ids) + 1)
    return source_columns, source_row_ids


def _write_dataset_metadata(metadata):
    INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INFO_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=4)


def _resolve_metadata_path(path_value):
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _dataset_preparation(info):
    preparation = info.get("preparation") if isinstance(info, dict) else None
    if not isinstance(preparation, dict) or not preparation.get("path"):
        return {}
    resolved = _resolve_metadata_path(preparation["path"])
    return {**preparation, "path": resolved, "exists": resolved.exists()}


def _dataset_intake(info, raw_path):
    info = info if isinstance(info, dict) else {}
    validation_mode = str(info.get("validation_mode", "NORMAL")).strip().upper()
    explicit_status = info.get("intake_status")
    if explicit_status in INTAKE_STATUSES:
        intake_status = explicit_status
    elif validation_mode == "NO_TEST":
        intake_status = "trusted"
    else:
        # Existing bundled and pre-mapping datasets keep working through an
        # identity interpretation until they are explicitly remapped.
        intake_status = "legacy"

    canonical_value = info.get("canonical_path")
    canonical_path = _resolve_metadata_path(canonical_value) if canonical_value else None
    if validation_mode == "NO_TEST" and canonical_path is None:
        canonical_path = Path(raw_path).resolve()
    if intake_status == "legacy" and canonical_path is None:
        canonical_path = Path(raw_path).resolve()

    return {
        "schema_id": info.get("schema_id", SCHEMA_ID),
        "schema_version": info.get("schema_version", get_schema()["version"]),
        "validation_mode": validation_mode,
        "intake_status": intake_status,
        "mapping_status": info.get("mapping_status", "legacy_identity" if intake_status == "legacy" else "unmapped"),
        "final_row_count": info.get("final_row_count"),
        "source_columns": list(info.get("source_columns") or []),
        "source_row_ids": list(info.get("source_row_ids") or []),
        "selected_columns": list(info.get("selected_columns") or []),
        "feature_fields": list(info.get("feature_fields") or []),
        "target_field": info.get("target_field"),
        "accepted_row_ids": list(info.get("accepted_row_ids") or []),
        "dropped_row_ids": list(info.get("dropped_row_ids") or []),
        "canonical_path": canonical_path,
        "mapped_path": _resolve_metadata_path(info["mapped_path"]) if info.get("mapped_path") else None,
        "mapping": dict(info.get("mapping") or {}),
        "conversion_ids": dict(info.get("conversion_ids") or {}),
        "validation_report": dict(info.get("validation_report") or {}),
        "review_decisions": dict(info.get("review_decisions") or {}),
    }


def _dataset_training_path(dataset):
    if dataset.get("intake_status") not in {"ready", "trusted", "legacy"}:
        return None
    canonical_path = dataset.get("canonical_path")
    if canonical_path and Path(canonical_path).exists():
        return Path(canonical_path)
    if not dataset.get("deletable"):
        raw_path = dataset.get("path")
        if raw_path and Path(raw_path).exists():
            return Path(raw_path)
    if dataset.get("validation_mode") == "NO_TEST" or dataset.get("intake_status") == "legacy":
        raw_path = dataset.get("path")
        if raw_path and Path(raw_path).exists():
            return Path(raw_path)
    return None


def _dataset_feature_fields(dataset):
    feature_fields = list(dataset.get("feature_fields") or [])
    if feature_fields:
        return feature_fields

    path = dataset.get("canonical_path") or dataset.get("path")
    if not path or not Path(path).exists():
        return list(FEATURE_FIELDS)
    try:
        source_columns, _ = _read_csv_metadata(path)
    except (OSError, csv.Error):
        return list(FEATURE_FIELDS)
    selected_fields = [field for field in FEATURE_FIELDS if field in source_columns]
    return selected_fields or list(FEATURE_FIELDS)


def _prepared_output_path(dataset_key, missing_strategy):
    safe_key = secure_filename(str(dataset_key)) or "dataset"
    safe_strategy = secure_filename(str(missing_strategy)) or "strategy"
    PREPARED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    candidate = (PREPARED_DATASETS_DIR / f"{safe_key}__prepared__{safe_strategy}.csv").resolve()
    try:
        candidate.relative_to(PREPARED_DATASETS_DIR.resolve())
    except ValueError as error:
        raise ValueError("The prepared dataset path is not valid.") from error
    return candidate


def _mapped_output_path(dataset_key, stage="mapped"):
    safe_key = secure_filename(str(dataset_key)) or "dataset"
    safe_stage = secure_filename(str(stage)) or "mapped"
    MAPPED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    candidate = (MAPPED_DATASETS_DIR / f"{safe_key}__{safe_stage}.csv").resolve()
    try:
        candidate.relative_to(MAPPED_DATASETS_DIR.resolve())
    except ValueError as error:
        raise ValueError("The mapped dataset path is not valid.") from error
    return candidate


def _metadata_path_value(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _safe_upload_path(filename):
    safe_name = secure_filename(filename or "")
    if not safe_name or Path(safe_name).suffix.lower() != ".csv":
        raise ValueError("Upload a CSV file with a valid filename.")

    input_root = INPUTS_DIR.resolve()
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    candidate = (input_root / safe_name).resolve()
    try:
        candidate.relative_to(input_root)
    except ValueError as error:
        raise ValueError("The uploaded filename is not valid.") from error

    if candidate.exists():
        candidate = input_root / f"{candidate.stem}_{uuid4().hex[:8]}{candidate.suffix}"
    return candidate


def get_datasets():
    """Loads dataset metadata and builds the list of available CSV inputs."""
    metadata = _read_dataset_metadata()
    datasets = {}

    # Scan bundled datasets first
    if BUNDLED_DATASETS_DIR.exists():
        for csv_file in BUNDLED_DATASETS_DIR.glob("*.csv"):
            key = csv_file.stem
            info = metadata.get(key, {
                "label": key.replace("_", " ").title(),
                "source": "Bundled",
                "description": "Bundled dataset included with the repository.",
                "link": "#",
            })
            datasets[key] = {
                "label": info.get("label", key),
                "path": csv_file,
                "source": info.get("source", "Bundled"),
                "description": info.get("description", ""),
                "link": info.get("link", "#"),
                "deletable": False,
            }

    # Then scan user-uploaded inputs
    if INPUTS_DIR.exists():
        for csv_file in INPUTS_DIR.glob("*.csv"):
            key = csv_file.stem
            info = metadata.get(key, {
                "label": key.replace("_", " ").title(),
                "source": "User Upload",
                "description": "User-supplied dataset.",
                "link": "#",
            })
            datasets[key] = {
                "label": info.get("label", key),
                "path": csv_file,
                "source": info.get("source", "User Upload"),
                "description": info.get("description", ""),
                "link": info.get("link", "#"),
                "deletable": True,
            }

    for key, dataset in datasets.items():
        intake = _dataset_intake(metadata.get(key, {}), dataset["path"])
        dataset.update(intake)
        if not dataset.get("source_columns") or not dataset.get("source_row_ids"):
            try:
                source_columns, source_row_ids = _read_csv_metadata(dataset["path"])
                dataset["source_columns"] = dataset.get("source_columns") or source_columns
                dataset["source_row_ids"] = dataset.get("source_row_ids") or source_row_ids
            except (OSError, csv.Error):
                pass
        if dataset.get("intake_status") in {"legacy", "trusted"}:
            if not dataset.get("feature_fields"):
                dataset["feature_fields"] = [
                    field for field in FEATURE_FIELDS if field in dataset.get("source_columns", [])
                ]
            if not dataset.get("target_field"):
                dataset["target_field"] = next(
                    (
                        field
                        for field in ("target", "heart_disease_binary", "num")
                        if field in dataset.get("source_columns", [])
                    ),
                    None,
                )
            if not dataset.get("selected_columns"):
                dataset["selected_columns"] = list(dataset["feature_fields"])
                if dataset.get("target_field"):
                    dataset["selected_columns"].append(dataset["target_field"])
            if not dataset.get("accepted_row_ids"):
                dataset["accepted_row_ids"] = list(dataset.get("source_row_ids", []))
            if dataset.get("final_row_count") is None:
                dataset["final_row_count"] = len(dataset["accepted_row_ids"])
        preparation = _dataset_preparation(metadata.get(key, {}))
        dataset["preparation"] = preparation
        dataset["prepared_path"] = preparation.get("path")
        dataset["training_ready"] = bool(preparation.get("exists"))
        dataset["training_available"] = bool(_dataset_training_path(dataset))

    return datasets


def _serialize_dataset(dataset_key, dataset):
    """Return dataset metadata in a JSON-safe shape for API clients."""
    serialized = {
        "dataset_key": dataset_key,
        "label": dataset.get("label", dataset_key),
        "source": dataset.get("source", ""),
        "description": dataset.get("description", ""),
        "link": dataset.get("link", "#"),
        "deletable": bool(dataset.get("deletable")),
        "schema_id": dataset.get("schema_id", SCHEMA_ID),
        "schema_version": dataset.get("schema_version", get_schema()["version"]),
        "validation_mode": dataset.get("validation_mode", "NORMAL"),
        "intake_status": dataset.get("intake_status", "legacy"),
        "mapping_status": dataset.get("mapping_status", "legacy_identity"),
        "final_row_count": dataset.get("final_row_count"),
        "source_columns": list(dataset.get("source_columns") or []),
        "source_row_ids": list(dataset.get("source_row_ids") or []),
        "selected_columns": list(dataset.get("selected_columns") or []),
        "feature_fields": list(dataset.get("feature_fields") or []),
        "target_field": dataset.get("target_field"),
        "accepted_row_ids": list(dataset.get("accepted_row_ids") or []),
        "dropped_row_ids": list(dataset.get("dropped_row_ids") or []),
        "training_available": bool(dataset.get("training_available")),
        "training_ready": bool(dataset.get("training_ready")),
        "canonical_path": _metadata_path_value(dataset["canonical_path"]) if dataset.get("canonical_path") else None,
        "intake": {
            "schema_id": dataset.get("schema_id", SCHEMA_ID),
            "schema_version": dataset.get("schema_version", get_schema()["version"]),
            "validation_mode": dataset.get("validation_mode", "NORMAL"),
            "intake_status": dataset.get("intake_status", "legacy"),
            "mapping_status": dataset.get("mapping_status", "legacy_identity"),
            "final_row_count": dataset.get("final_row_count"),
            "source_columns": list(dataset.get("source_columns") or []),
            "source_row_ids": list(dataset.get("source_row_ids") or []),
            "selected_columns": list(dataset.get("selected_columns") or []),
            "feature_fields": list(dataset.get("feature_fields") or []),
            "target_field": dataset.get("target_field"),
            "accepted_row_ids": list(dataset.get("accepted_row_ids") or []),
            "dropped_row_ids": list(dataset.get("dropped_row_ids") or []),
            "mapping": dict(dataset.get("mapping") or {}),
            "conversion_ids": dict(dataset.get("conversion_ids") or {}),
            "validation_report": dict(dataset.get("validation_report") or {}),
            "review_decisions": dict(dataset.get("review_decisions") or {}),
        },
        "preparation": dict(dataset.get("preparation") or {}),
    }
    preparation = serialized["preparation"]
    if preparation.get("path"):
        preparation["path"] = _metadata_path_value(preparation["path"])
    serialized["prepared_path"] = _metadata_path_value(dataset["prepared_path"]) if dataset.get("prepared_path") else None
    return serialized


def _serialize_model(model, datasets):
    """Add browser-friendly display metadata without hiding model inputs."""
    dataset_key = model.get("dataset_key", "")
    dataset = datasets.get(dataset_key, {})
    dataset_label = dataset.get("label") or str(dataset_key).replace("_", " ").title()
    method = str(model.get("method", "")).strip().lower()
    method_info = next((item for item in method_catalog() if item["value"] == method), None)
    method_label = method_info["label"] if method_info else method.replace("_", " ").title()
    created_at = model.get("created_at")
    display_name = f"{dataset_label} - {method_label}"
    if created_at:
        display_name = f"{display_name} ({created_at})"
    return {**model, "display_name": display_name}


def supported_method_catalog():
    """Return only the estimators exposed by the bundled Cleveland workflow."""
    catalog = {
        method["value"]: method
        for method in method_catalog()
        if method["value"] in SUPPORTED_APP_METHODS
    }
    return [catalog[method] for method in SUPPORTED_APP_METHODS if method in catalog]


def list_bundled_models():
    """Return the newest available artifact for each bundled workflow method."""
    selected = {}
    for record in list_registered_models():
        if record.get("dataset_key") != BUNDLED_DATASET_KEY:
            continue
        method = str(record.get("method", "")).strip().lower()
        if method not in SUPPORTED_APP_METHODS or method in selected:
            continue
        selected[method] = record

    method_order = {method: index for index, method in enumerate(SUPPORTED_APP_METHODS)}
    return sorted(
        selected.values(),
        key=lambda record: method_order.get(record.get("method"), len(method_order)),
    )


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@api.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_exists": model_exists(),
    })


# ============================================================================
# Dataset Endpoints
# ============================================================================

@api.get("/datasets")
def list_datasets():
    """
    List all available datasets (bundled and uploaded).

    Returns:
        - 200: Array of dataset objects with metadata
    """
    try:
        datasets = get_datasets()
        return jsonify([_serialize_dataset(key, dataset) for key, dataset in datasets.items()])
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@api.get("/schemas")
@api.get("/metadata/schemas")
def get_schemas():
    """Return schemas and field metadata used by the browser intake workflow."""
    return jsonify(schema_catalog())


@api.get("/intake")
@api.get("/metadata/intake")
def get_intake_metadata():
    """Return upload modes and review decision choices for browser-generated controls."""
    return jsonify({
        "validation_modes": [
            {
                "value": "NORMAL",
                "label": "Test",
                "description": "Map source columns, review the validation report, and approve the dataset before training.",
            },
            {
                "value": "NO_TEST",
                "label": "No Test",
                "description": "Skip mapping and schema diagnostics; the uploaded file is trusted for training.",
            },
        ],
        "review_decisions": [
            {
                "name": "missing_rows",
                "label": "Missing rows",
                "description": "Choose whether rows with missing mapped values remain in the reviewed dataset.",
                "options": [
                    {"value": "drop", "label": "Drop affected rows"},
                    {"value": "keep", "label": "Keep affected rows for training strategy"},
                ],
            },
            {
                "name": "missing_columns",
                "label": "Missing schema columns",
                "description": "Choose whether an incomplete schema blocks approval.",
                "options": [
                    {"value": "reject", "label": "Block approval"},
                    {"value": "allow", "label": "Allow and review available fields"},
                ],
            },
            {
                "name": "out_of_range",
                "label": "Out-of-range values",
                "description": "Choose whether invalid domain or range values are removed or retained.",
                "options": [
                    {"value": "reject", "label": "Block approval"},
                    {"value": "drop", "label": "Drop affected rows"},
                    {"value": "keep", "label": "Keep values for training validation"},
                ],
            },
            {
                "name": "conversion_errors",
                "label": "Conversion errors",
                "description": "Values that cannot be converted to the mapped field type must be removed or rejected.",
                "options": [
                    {"value": "reject", "label": "Block approval"},
                    {"value": "drop", "label": "Drop affected rows"},
                ],
            },
            {
                "name": "unmapped_columns",
                "label": "Unmapped source columns",
                "description": "Choose whether extra source columns remain as metadata in the reviewed file.",
                "options": [
                    {"value": "keep", "label": "Keep metadata columns"},
                    {"value": "drop", "label": "Remove unmapped columns"},
                ],
            },
        ],
        "field_issue_actions": [
            {"value": "replace_null", "label": "Replace with NULL (NaN)"},
            {"value": "impute", "label": "Impute"},
            {"value": "drop_rows", "label": "Drop affected rows"},
            {"value": "drop_column", "label": "Drop column"},
        ],
    })


@api.post("/datasets/upload")
def upload_dataset():
    """
    Upload a new CSV dataset.

    Parameters (form data):
        - data_file (required): CSV file to upload
        - label (optional): Human-readable label for the dataset
        - source (optional): Source description
        - description (optional): Dataset description
        - schema_id (optional): Dataset schema, default ``cleveland_v1``
        - validation_mode (optional): ``NORMAL`` or explicit ``NO_TEST``
    Returns:
        - 201: Created dataset with metadata
        - 400: Invalid file or parameters
    """
    uploaded_file = request.files.get("data_file")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "No CSV file provided. Use 'data_file' parameter."}), 400

    schema_id = request.form.get("schema_id", SCHEMA_ID).strip() or SCHEMA_ID
    validation_mode = request.form.get("validation_mode", "NORMAL").strip().upper() or "NORMAL"
    if validation_mode not in VALIDATION_MODES:
        return jsonify({
            "error": f"Unknown validation mode '{validation_mode}'. Use NORMAL or NO_TEST.",
            "code": "INVALID_VALIDATION_MODE",
        }), 400
    try:
        schema = get_schema(schema_id)
    except ValueError as error:
        return jsonify({"error": str(error), "code": "UNKNOWN_SCHEMA"}), 400
    try:
        file_path = _safe_upload_path(uploaded_file.filename)
        uploaded_file.save(file_path)
        source_columns, source_row_ids = _read_csv_metadata(file_path)
    except (OSError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    filename = file_path.name
    key = Path(filename).stem
    feature_fields = [field for field in FEATURE_FIELDS if field in source_columns]
    target_field = next(
        (field for field in ("target", "heart_disease_binary", "num") if field in source_columns),
        None,
    )
    selected_columns = list(feature_fields)
    if target_field:
        selected_columns.append(target_field)
    accepted_row_ids = list(source_row_ids) if validation_mode == "NO_TEST" else []

    metadata = _read_dataset_metadata()
    metadata[key] = {
        "label": request.form.get("label", key),
        "source": request.form.get("source", "User Upload"),
        "description": request.form.get("description", ""),
        "link": "#",
        "schema_id": schema_id,
        "schema_version": schema["version"],
        "validation_mode": validation_mode,
        "intake_status": "trusted" if validation_mode == "NO_TEST" else "mapping",
        "mapping_status": "trusted" if validation_mode == "NO_TEST" else "unmapped",
        "final_row_count": len(accepted_row_ids) if validation_mode == "NO_TEST" else None,
        "source_columns": source_columns,
        "source_row_ids": source_row_ids,
        "selected_columns": selected_columns if validation_mode == "NO_TEST" else [],
        "feature_fields": feature_fields if validation_mode == "NO_TEST" else [],
        "target_field": target_field if validation_mode == "NO_TEST" else None,
        "accepted_row_ids": accepted_row_ids,
        "dropped_row_ids": [],
        "canonical_path": _metadata_path_value(file_path) if validation_mode == "NO_TEST" else None,
        "mapping": {},
        "conversion_ids": {},
        "validation_report": {},
        "review_decisions": {},
    }
    _write_dataset_metadata(metadata)

    dataset = get_datasets().get(key, {})

    return jsonify({
        "dataset_key": key,
        "filename": filename,
        "label": metadata[key]["label"],
        "schema_id": schema_id,
        "validation_mode": validation_mode,
        "intake_status": metadata[key]["intake_status"],
        "mapping_status": metadata[key]["mapping_status"],
        "final_row_count": metadata[key]["final_row_count"],
        "source_columns": source_columns,
        "source_row_ids": source_row_ids,
        "selected_columns": metadata[key]["selected_columns"],
        "feature_fields": metadata[key]["feature_fields"],
        "accepted_row_ids": accepted_row_ids,
        "training_available": bool(dataset.get("training_available")),
        "dataset": _serialize_dataset(key, dataset) if dataset else None,
        "message": (
            "Dataset uploaded as trusted input. It is available for training."
            if validation_mode == "NO_TEST"
            else "Dataset uploaded. Complete field mapping and review before training."
        ),
    }), 201


@api.get("/datasets/<dataset_key>/field-analysis")
def analyze_dataset_fields(dataset_key):
    """Analyze source columns and return schema mapping suggestions."""
    datasets = get_datasets()
    dataset = datasets.get(dataset_key)
    if not dataset:
        return jsonify({"error": "Dataset not found.", "code": "DATASET_NOT_FOUND"}), 404
    if dataset.get("validation_mode") == "NO_TEST":
        return jsonify({
            "error": "NO_TEST datasets do not use field analysis.",
            "code": "NO_TEST_MAPPING_SKIPPED",
        }), 409

    try:
        analysis = analyze_dataset(
            dataset["path"],
            schema_id=dataset.get("schema_id", SCHEMA_ID),
        )
    except (OSError, ValueError) as error:
        return jsonify({"error": str(error), "code": "FIELD_ANALYSIS_FAILED"}), 400

    return jsonify({
        "dataset_key": dataset_key,
        "intake_status": dataset.get("intake_status"),
        "mapping_status": dataset.get("mapping_status"),
        **analysis,
    })


@api.post("/datasets/<dataset_key>/field-mapping")
def map_dataset_fields(dataset_key):
    """Apply a browser-confirmed field mapping and create a canonical artifact."""
    datasets = get_datasets()
    dataset = datasets.get(dataset_key)
    if not dataset:
        return jsonify({"error": "Dataset not found.", "code": "DATASET_NOT_FOUND"}), 404
    if dataset.get("validation_mode") == "NO_TEST":
        return jsonify({
            "error": "NO_TEST datasets do not use field mapping.",
            "code": "NO_TEST_MAPPING_SKIPPED",
        }), 409

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    schema_id = payload.get("schema_id") or dataset.get("schema_id", SCHEMA_ID)
    if schema_id != dataset.get("schema_id", SCHEMA_ID):
        return jsonify({"error": "The mapping schema does not match the uploaded dataset.", "code": "SCHEMA_MISMATCH"}), 400
    mapping = payload.get("mapping", payload.get("entries"))

    try:
        mapped_path = _mapped_output_path(dataset_key, "mapped")
        mapping_result = apply_mapping(
            dataset["path"],
            mapped_path,
            mapping,
            schema_id=schema_id,
        )
        metadata = _read_dataset_metadata()
        dataset_metadata = metadata.setdefault(dataset_key, {})
        dataset_metadata.update({
            "schema_id": schema_id,
            "schema_version": mapping_result["schema_version"],
            "validation_mode": "NORMAL",
            "intake_status": "review",
            "mapping_status": "mapped",
            "final_row_count": None,
            "source_columns": mapping_result["source_columns"],
            "source_row_ids": mapping_result["source_row_ids"],
            "selected_columns": mapping_result["mapping"]["selected_columns"],
            "feature_fields": mapping_result["mapping"]["feature_fields"],
            "target_field": mapping_result["mapping"]["target_field"],
            "accepted_row_ids": [],
            "dropped_row_ids": [],
            "mapped_path": _metadata_path_value(mapped_path),
            "canonical_path": _metadata_path_value(mapped_path),
            "mapping": mapping_result["mapping"],
            "conversion_ids": mapping_result["conversion_ids"],
            "validation_report": mapping_result["report"],
            "review_decisions": {},
        })
        _write_dataset_metadata(metadata)
        refreshed = get_datasets().get(dataset_key, {})
        return jsonify({
            "dataset_key": dataset_key,
            "intake_status": "review",
            "mapping_status": "mapped",
            "mapping": mapping_result["mapping"],
            "report": mapping_result["report"],
            "dataset": _serialize_dataset(dataset_key, refreshed),
            "message": "Field mapping applied. Review the validation report before approval.",
        })
    except (OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error), "code": "FIELD_MAPPING_FAILED"}), 400


@api.post("/datasets/<dataset_key>/review")
def review_dataset(dataset_key):
    """Commit explicit validation-report decisions and make a dataset training-available."""
    datasets = get_datasets()
    dataset = datasets.get(dataset_key)
    if not dataset:
        return jsonify({"error": "Dataset not found.", "code": "DATASET_NOT_FOUND"}), 404
    if dataset.get("validation_mode") == "NO_TEST":
        return jsonify({
            "error": "NO_TEST datasets do not require a mapping review.",
            "code": "NO_TEST_REVIEW_SKIPPED",
        }), 409
    if dataset.get("mapping_status") != "mapped":
        return jsonify({
            "error": "Apply a field mapping before reviewing the validation report.",
            "code": "MAPPING_REQUIRED",
        }), 409

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    field_decisions = payload.get("field_decisions")
    decisions = payload.get("decisions", payload) if field_decisions is None else {}
    canonical_path = dataset.get("canonical_path")
    mapping_result = {
        "source_columns": dataset.get("source_columns", []),
        "mapping": dataset.get("mapping", {}),
        "report": dataset.get("validation_report", {}),
    }
    if not canonical_path or not Path(canonical_path).exists():
        return jsonify({"error": "The mapped canonical dataset is not available.", "code": "MAPPED_DATASET_NOT_FOUND"}), 409

    try:
        if not mapping_result["mapping"].get("target_field"):
            raise ValueError("Map exactly one target field before approving the dataset.")
        if not mapping_result["mapping"].get("feature_fields"):
            raise ValueError("Map at least one feature field before approving the dataset.")
        reviewed_path = _mapped_output_path(dataset_key, "reviewed")
        review_result = review_mapping(
            canonical_path,
            reviewed_path,
            mapping_result,
            decisions,
            field_decisions=field_decisions,
        )
        if review_result["rows_after"] <= 0:
            raise ValueError("Review decisions leave no rows available for training.")
        metadata = _read_dataset_metadata()
        dataset_metadata = metadata.setdefault(dataset_key, {})
        dataset_metadata.update({
            "intake_status": "ready",
            "mapping_status": "reviewed",
            "final_row_count": review_result["rows_after"],
            "selected_columns": review_result["selected_columns"],
            "feature_fields": review_result["feature_fields"],
            "target_field": review_result["target_field"],
            "accepted_row_ids": review_result["selected_row_ids"],
            "dropped_row_ids": review_result["dropped_row_ids"],
            "canonical_path": _metadata_path_value(reviewed_path),
            "reviewed_path": _metadata_path_value(reviewed_path),
            "review_decisions": review_result["decisions"],
            "validation_report": {
                **review_result["report"],
                "final_row_count": review_result["rows_after"],
                "review": {
                    "rows_before": review_result["rows_before"],
                    "total_rows_before_review": review_result["total_rows_before_review"],
                    "rows_after": review_result["rows_after"],
                    "final_row_count": review_result["rows_after"],
                    "dropped_rows": review_result["dropped_rows"],
                    "decisions": review_result["decisions"],
                    "selected_columns": review_result["selected_columns"],
                    "feature_fields": review_result["feature_fields"],
                    "target_field": review_result["target_field"],
                    "source_row_ids": review_result["source_row_ids"],
                    "selected_row_ids": review_result["selected_row_ids"],
                    "dropped_row_ids": review_result["dropped_row_ids"],
                },
            },
        })
        _write_dataset_metadata(metadata)
        refreshed = get_datasets().get(dataset_key, {})
        review_response = {key: value for key, value in review_result.items() if key != "path"}
        review_response["final_row_count"] = review_result["rows_after"]
        return jsonify({
            "dataset_key": dataset_key,
            "intake_status": "ready",
            "mapping_status": "reviewed",
            "review": review_response,
            "dataset": _serialize_dataset(dataset_key, refreshed),
            "message": "Review approved. The dataset is now available for training.",
        })
    except (OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error), "code": "DATASET_REVIEW_FAILED"}), 400


@api.get("/datasets/<dataset_key>")
def get_dataset(dataset_key):
    """
    Get metadata for a specific dataset.

    Parameters:
        - dataset_key (path): The dataset identifier

    Returns:
        - 200: Dataset object with metadata
        - 404: Dataset not found
    """
    datasets = get_datasets()
    if dataset_key not in datasets:
        return jsonify({"error": "Dataset not found."}), 404
    return jsonify(_serialize_dataset(dataset_key, datasets[dataset_key]))


@api.delete("/datasets/<dataset_key>")
def delete_dataset(dataset_key):
    """
    Delete a user-uploaded dataset and its preparation.

    Parameters:
        - dataset_key (path): The dataset identifier

    Returns:
        - 200: Deletion confirmation
        - 403: Cannot delete bundled datasets
        - 404: Dataset not found
        - 500: Deletion failed
    """
    datasets = get_datasets()
    dataset = datasets.get(dataset_key)

    if not dataset:
        return jsonify({"error": "Dataset not found."}), 404

    if not dataset.get("deletable"):
        return jsonify({"error": "Bundled datasets cannot be deleted."}), 403

    try:
        input_root = INPUTS_DIR.resolve()
        dataset_path = Path(dataset["path"]).resolve()
        dataset_path.relative_to(input_root)
    except (OSError, ValueError):
        return jsonify({"error": "Cannot delete this dataset."}), 403

    try:
        dataset_path.unlink()

        # Remove canonical mapping/review artifacts without touching the original upload.
        for artifact in (dataset.get("mapped_path"), dataset.get("canonical_path")):
            if not artifact:
                continue
            try:
                artifact_path = Path(artifact).resolve()
                artifact_path.relative_to(MAPPED_DATASETS_DIR.resolve())
                artifact_path.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass

        # Delete prepared version if it exists
        prepared_path = dataset.get("prepared_path")
        if prepared_path:
            try:
                prepared_path = Path(prepared_path).resolve()
                prepared_path.relative_to(PREPARED_DATASETS_DIR.resolve())
                prepared_path.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass

        # Remove from metadata if not bundled
        bundled_path = BUNDLED_DATASETS_DIR / f"{dataset_key}.csv"
        if not bundled_path.exists():
            metadata = _read_dataset_metadata()
            if dataset_key in metadata:
                metadata.pop(dataset_key)
                _write_dataset_metadata(metadata)

        return jsonify({
            "message": f"Dataset '{dataset['label']}' deleted successfully.",
            "dataset_key": dataset_key,
        })
    except (OSError, ValueError) as error:
        return jsonify({"error": f"Could not delete the dataset: {error}"}), 500


@api.post("/datasets/<dataset_key>/prepare")
def prepare_dataset(dataset_key):
    """
    Prepare a dataset for training by handling missing values in its approved fields.

    Parameters:
        - dataset_key (path): The dataset identifier
        - missing_strategy (query): Strategy for missing values (e.g., 'drop', 'impute', 'native')

    Returns:
        - 200: Preparation result with metadata
        - 400: Invalid parameters or preparation failed
        - 404: Dataset not found
    """
    datasets = get_datasets()
    dataset = datasets.get(dataset_key)

    if not dataset:
        return jsonify({"error": "Dataset not found."}), 404

    if dataset.get("validation_mode") != "NO_TEST" and dataset.get("intake_status") not in {"ready", "legacy"}:
        return jsonify({
            "error": "Complete field mapping and review before preparing this dataset.",
            "code": "INTAKE_REVIEW_REQUIRED",
            "intake_status": dataset.get("intake_status"),
            "mapping_status": dataset.get("mapping_status"),
        }), 409

    payload = request.get_json(silent=True) or {}
    missing_strategy = request.args.get("missing_strategy") or request.form.get("missing_strategy") or payload.get("missing_strategy", "drop")
    feature_fields = _dataset_feature_fields(dataset)

    try:
        dataset_path = _dataset_training_path(dataset)
        if dataset_path is None:
            raise ValueError("The reviewed dataset is not available for preparation.")
        output_path = _prepared_output_path(dataset_key, missing_strategy)
        preparation_result = prepare_training_data(
            dataset_path,
            output_path,
            feature_fields=feature_fields,
            missing_strategy=missing_strategy,
        )

        metadata = _read_dataset_metadata()
        dataset_metadata = metadata.setdefault(dataset_key, {})
        dataset_metadata["preparation"] = {
            "path": _metadata_path_value(preparation_result["path"]),
            "missing_strategy": preparation_result["missing_strategy"],
            "feature_fields": preparation_result["feature_fields"],
            "rows": preparation_result["rows"],
            "training_row_ids": preparation_result["training_row_ids"],
            "class_counts": preparation_result["class_counts"],
            "validation_mode": dataset.get("validation_mode", "NORMAL"),
            "intake_status": dataset.get("intake_status"),
        }
        _write_dataset_metadata(metadata)

        return jsonify({
            "dataset_key": dataset_key,
            "preparation": dataset_metadata["preparation"],
            "message": f"Dataset prepared with {len(feature_fields)} selected fields and {missing_strategy} handling.",
        })
    except (OSError, ValueError, TypeError) as error:
        return jsonify({"error": f"Could not prepare the dataset: {error}"}), 400


@api.get("/datasets/<dataset_key>/inspect")
def inspect_dataset(dataset_key):
    """
    Inspect a dataset and get statistical summary.

    Parameters:
        - dataset_key (path): The dataset identifier
        - cv_folds (query, default=5): Number of cross-validation folds

    Returns:
        - 200: Dataset inspection summary
        - 400: Invalid parameters or dataset invalid
        - 404: Dataset not found
    """
    datasets = get_datasets()
    if dataset_key not in datasets:
        return jsonify({"error": "Dataset not found.", "code": "DATASET_NOT_FOUND"}), 404

    dataset = datasets[dataset_key]
    if dataset.get("validation_mode") != "NO_TEST" and dataset.get("intake_status") not in {"ready", "legacy"}:
        return jsonify({
            "dataset_key": dataset_key,
            "intake_status": dataset.get("intake_status"),
            "mapping_status": dataset.get("mapping_status"),
            "validation_report": dataset.get("validation_report", {}),
            "error": "Complete field mapping and review before inspecting training readiness.",
            "code": "INTAKE_REVIEW_REQUIRED",
        }), 409

    try:
        cv_folds = int(request.args.get("cv_folds", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid cv_folds value."}), 400

    dataset_path = _dataset_training_path(dataset)
    if dataset_path is None:
        return jsonify({"error": "The reviewed dataset is not available.", "code": "CANONICAL_DATASET_NOT_FOUND"}), 409
    try:
        summary = inspect_training_data(
            dataset_path,
            cv_folds=cv_folds,
            feature_fields=_dataset_feature_fields(dataset),
        )
    except ValueError as error:
        return jsonify({"error": str(error), "code": "DATASET_INVALID"}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error), "code": "DATASET_NOT_FOUND"}), 404
    except Exception as error:
        return jsonify({"error": str(error), "code": "INSPECT_FAILED"}), 500

    return jsonify({
        **summary,
        "dataset_key": dataset_key,
        "intake_status": dataset.get("intake_status"),
        "mapping_status": dataset.get("mapping_status"),
        "validation_mode": dataset.get("validation_mode"),
        "validation_report": dataset.get("validation_report", {}),
    })


# Legacy endpoint for backwards compatibility
@api.get("/inspect")
def api_inspect():
    """Legacy endpoint - redirect to /datasets/{key}/inspect"""
    dataset_key = request.args.get("dataset")
    if not dataset_key:
        return jsonify({"error": "Missing 'dataset' query parameter."}), 400

    # Redirect to the new endpoint
    return inspect_dataset(dataset_key)


# ============================================================================
# Model & Training Endpoints
# ============================================================================

@api.get("/models")
def list_models():
    """List the available bundled Cleveland model artifacts."""
    try:
        datasets = get_datasets()
        return jsonify([_serialize_model(model, datasets) for model in list_bundled_models()])
    except Exception as error:
        return jsonify({"error": str(error)}), 500


def _normalize_requested_method(value):
    method = str(value or "knn").strip().lower()
    return "naive_bayes" if method == "bayes" else method


def _train_model_from_payload(payload, bundled_only=False, include_visualizations=True):
    dataset_key = payload.get("dataset") or BUNDLED_DATASET_KEY
    if bundled_only and dataset_key != BUNDLED_DATASET_KEY:
        raise ValueError("The application training workflow uses the bundled Cleveland dataset.")

    datasets = get_datasets()
    dataset = datasets.get(dataset_key)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_key}' not found.")
    if not dataset.get("training_available"):
        raise ValueError("Complete field mapping and review before training.")

    is_bundled = dataset_key == BUNDLED_DATASET_KEY
    method = _normalize_requested_method(payload.get("method", "knn"))
    if is_bundled and method not in SUPPORTED_APP_METHODS:
        raise ValueError("Choose Bayes, KNN, or SVM for the bundled Cleveland workflow.")

    if is_bundled:
        cv_folds = 5
        random_state = int(payload.get("random_state", 42))
        feature_fields = list(FEATURE_FIELDS)
        missing_strategy = "impute"
    else:
        cv_folds = int(payload.get("cv_folds", 5))
        random_state = int(payload.get("random_state", 42))
        feature_fields = _dataset_feature_fields(dataset)
        missing_strategy = payload.get("missing_strategy") or "impute"

    dataset_path = _dataset_training_path(dataset)
    if dataset_path is None:
        raise ValueError("The approved dataset is not available for training.")

    kwargs = {}
    selected_method = next(
        (item for item in method_catalog() if item["value"] == method),
        None,
    )
    if selected_method:
        for parameter in selected_method.get("params", []):
            if parameter["name"] in payload:
                kwargs[parameter["name"]] = payload[parameter["name"]]
    if is_bundled and method == "knn":
        kwargs["n_neighbors"] = 5

    result = train_model(
        dataset_path,
        method=method,
        random_state=random_state,
        cv_folds=cv_folds,
        feature_fields=feature_fields,
        missing_strategy=missing_strategy,
        dataset_key=dataset_key,
        **kwargs,
    )
    result.update({
        "dataset_selected_columns": list(dataset.get("selected_columns") or feature_fields),
        "dataset_source_row_ids": list(dataset.get("source_row_ids") or []),
        "dataset_accepted_row_ids": list(dataset.get("accepted_row_ids") or []),
    })

    metrics_plot = None
    feature_importance_plot = None
    if include_visualizations:
        try:
            from model.visualization import create_training_metrics_plot
            metrics_plot = create_training_metrics_plot(
                result,
                title=f"{result['method'].replace('_', ' ').title()} Training Metrics",
            )
        except Exception:
            pass

        try:
            trained_model = load_model(model_path=result["model_path"])
            feature_importances = extract_feature_importances(
                trained_model,
                result.get("feature_fields", FEATURE_FIELDS)
            )
            if feature_importances:
                from model.visualization import create_feature_importance_plot
                feature_importance_plot = create_feature_importance_plot(
                    feature_importances,
                    title=f"{result['method'].replace('_', ' ').title()} Feature Importance",
                )
        except Exception:
            pass

    return result, metrics_plot, feature_importance_plot


@api.post("/models/train")
def train_model_api():
    """Train one model, using the fixed bundled configuration for Cleveland."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        result, metrics_plot, feature_importance_plot = _train_model_from_payload(payload)
        return jsonify({
            **result,
            "metrics_plot": metrics_plot,
            "feature_importance_plot": feature_importance_plot,
        }), 201
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except (FileNotFoundError, OSError) as error:
        return jsonify({"error": str(error)}), 503
    except Exception as error:
        return jsonify({"error": f"Training failed: {error}"}), 500


@api.post("/models/train-all")
def train_all_models_api():
    """Train and register Bayes, KNN, and SVM on the bundled Cleveland data."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        base_payload = {
            "dataset": payload.get("dataset", BUNDLED_DATASET_KEY),
            "random_state": payload.get("random_state", 42),
        }
        results = []
        for method in SUPPORTED_APP_METHODS:
            result, _, _ = _train_model_from_payload(
                {**base_payload, "method": method},
                bundled_only=True,
                include_visualizations=False,
            )
            results.append(result)
        return jsonify({
            "dataset_key": BUNDLED_DATASET_KEY,
            "feature_fields": list(FEATURE_FIELDS),
            "missing_strategy": "impute",
            "cv_folds": 5,
            "models": results,
        }), 201
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except (FileNotFoundError, OSError) as error:
        return jsonify({"error": str(error)}), 503
    except Exception as error:
        return jsonify({"error": f"Training failed: {error}"}), 500


@api.delete("/models/<model_id>")
def delete_model_api(model_id):
    """
    Delete a trained model.

    Parameters:
        - model_id (path): The model identifier

    Returns:
        - 200: Deletion confirmation
        - 404: Model not found
        - 500: Deletion failed
    """
    try:
        deleted_model = delete_registered_model(model_id)
        return jsonify({
            "message": f"Model '{model_id}' deleted successfully.",
            "model_id": model_id,
            "deleted_model": deleted_model,
        })
    except FileNotFoundError:
        return jsonify({"error": "Model not found."}), 404
    except (OSError, ValueError) as error:
        return jsonify({"error": f"Could not delete the model: {error}"}), 500


# ============================================================================
# Prediction Endpoint
# ============================================================================

@api.post("/predict")
def api_predict():
    """
    Make a prediction using the default or specified model.

    Parameters (JSON):
        - age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        - model_id (optional): Specific model to use; defaults to latest

    Returns:
        - 200: Prediction with probability
        - 400: Invalid input or model not found
        - 503: Model load failed
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        model_id = payload.get("model_id")
        selected_fields = FEATURE_FIELDS
        registered_models = list_bundled_models()
        if model_id:
            selected_model = next(
                (record for record in registered_models if record.get("model_id") == model_id),
                None,
            )
        else:
            active_model_id = load_training_registry().get("active_model_id")
            selected_model = next(
                (record for record in registered_models if record.get("model_id") == active_model_id),
                None,
            )
        if selected_model and selected_model.get("feature_fields"):
            selected_fields = selected_model["feature_fields"]
        feature_payload = {field: value for field, value in payload.items() if field != "model_id"}
        values = validate_feature_values(feature_payload, feature_fields=selected_fields)
        model = load_model(model_id=model_id) if model_id else load_model()
        prediction_value, probability = predict(model, values, feature_fields=selected_fields)
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except (FileNotFoundError, OSError) as error:
        return jsonify({"error": str(error)}), 503

    return jsonify({"prediction": prediction_value, "probability": probability})


# ============================================================================
# Metadata & Catalog Endpoints
# ============================================================================

@api.get("/metadata/data-dictionary")
def get_data_dictionary():
    """
    Get the data dictionary describing all available fields.

    Returns:
        - 200: Array of field definitions
    """
    return jsonify(_data_dictionary())


def _data_dictionary():
    dictionary = []
    for field in get_schema()["fields"]:
        if field.get("allowed") is not None:
            domain = ", ".join(str(value) for value in field["allowed"])
        else:
            minimum = field.get("minimum", "")
            maximum = field.get("maximum", "")
            domain = f"Numeric ({minimum}-{maximum})"
        dictionary.append({**field, "domain": domain})
    return dictionary


@api.get("/metadata/field-definitions")
def get_field_definitions():
    """
    Get field definitions with options for categorical fields.

    Returns:
        - 200: Dictionary mapping field names to definitions and options
    """
    prediction_field_options = {
        "sex": [
            {"value": 0, "label": "0 = Female"},
            {"value": 1, "label": "1 = Male"},
        ],
        "cp": [
            {"value": 1, "label": "1 = Typical angina"},
            {"value": 2, "label": "2 = Atypical angina"},
            {"value": 3, "label": "3 = Non-anginal pain"},
            {"value": 4, "label": "4 = Asymptomatic"},
        ],
        "fbs": [
            {"value": 0, "label": "0 = False"},
            {"value": 1, "label": "1 = True"},
        ],
        "restecg": [
            {"value": 0, "label": "0 = Normal"},
            {"value": 1, "label": "1 = ST-T wave abnormality"},
            {"value": 2, "label": "2 = Left ventricular hypertrophy"},
        ],
        "exang": [
            {"value": 0, "label": "0 = No"},
            {"value": 1, "label": "1 = Yes"},
        ],
        "slope": [
            {"value": 1, "label": "1 = Upsloping"},
            {"value": 2, "label": "2 = Flat"},
            {"value": 3, "label": "3 = Downsloping"},
        ],
        "ca": [
            {"value": 0, "label": "0 = 0 vessels"},
            {"value": 1, "label": "1 = 1 vessel"},
            {"value": 2, "label": "2 = 2 vessels"},
            {"value": 3, "label": "3 = 3 vessels"},
        ],
        "thal": [
            {"value": 3, "label": "3 = Normal"},
            {"value": 6, "label": "6 = Fixed defect"},
            {"value": 7, "label": "7 = Reversible defect"},
        ],
    }

    data_dictionary = [
        {"field": "age", "label": "Age", "units": "years", "domain": "Numeric (0-120)", "description": "Age of the patient."},
        {"field": "sex", "label": "Sex", "units": "binary", "domain": "0=Female, 1=Male", "description": "Biological sex of the patient."},
        {"field": "cp", "label": "Chest Pain Type", "units": "coded", "domain": "1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic", "description": "Type of chest pain experienced."},
        {"field": "trestbps", "label": "Resting BP", "units": "mm Hg", "domain": "Numeric (0-300)", "description": "Resting blood pressure on admission."},
        {"field": "chol", "label": "Serum Cholestrol", "units": "mg/dl", "domain": "Numeric (0-1000)", "description": "Serum cholesterol measurements."},
        {"field": "fbs", "label": "Fasting Blood Sugar", "units": "binary", "domain": ">120 mg/dl (1=True, 0=False)", "description": "Fasting blood sugar level."},
        {"field": "restecg", "label": "Resting ECG", "units": "coded", "domain": "0=Normal, 1=ST-T wave, 2=LV hypertrophy", "description": "Resting electrocardiographic results."},
        {"field": "thalach", "label": "Max Heart Rate", "units": "bpm", "domain": "Numeric (0-250)", "description": "Maximum heart rate achieved."},
        {"field": "exang", "label": "Exercise Angina", "units": "binary", "domain": "1=Yes, 0=No", "description": "Exercise induced angina."},
        {"field": "oldpeak", "label": "ST Depression", "units": "mm", "domain": "Numeric (0-10)", "description": "ST depression induced by exercise relative to rest."},
        {"field": "slope", "label": "ST Slope", "units": "coded", "domain": "1=Upsloping, 2=Flat, 3=Downsloping", "description": "The slope of the peak exercise ST segment."},
        {"field": "ca", "label": "Colored Vessels", "units": "count", "domain": "0-3 vessels", "description": "Number of major vessels colored by fluoroscopy."},
        {"field": "thal", "label": "Thalassemia", "units": "coded", "domain": "3=Normal, 6=Fixed, 7=Reversable", "description": "A blood disorder called thalassemia."},
        {"field": "target", "label": "Diagnosis", "units": "binary", "domain": "0=<50% diameter, 1=>50% diameter", "description": "Heart disease diagnosis status."}
    ]

    field_specs = {
        "age": {"type": "number", "minimum": 0, "maximum": 120, "step": 1},
        "sex": {"type": "select"},
        "cp": {"type": "select"},
        "trestbps": {"type": "number", "minimum": 0, "maximum": 300, "step": 1},
        "chol": {"type": "number", "minimum": 0, "maximum": 1000, "step": 1},
        "fbs": {"type": "select"},
        "restecg": {"type": "select"},
        "thalach": {"type": "number", "minimum": 0, "maximum": 250, "step": 1},
        "exang": {"type": "select"},
        "oldpeak": {"type": "number", "minimum": 0, "maximum": 10, "step": 0.1},
        "slope": {"type": "select"},
        "ca": {"type": "select"},
        "thal": {"type": "select"},
    }

    field_definitions = {
        item["field"]: {
            "label": item["label"],
            "description": item["description"],
            "domain": item["domain"],
            "units": item["units"],
            "required": True,
            **field_specs.get(item["field"], {"type": "number", "step": "any"}),
            "options": prediction_field_options.get(item["field"]),
        }
        for item in data_dictionary
        if item["field"] != "target"
    }
    return jsonify(field_definitions)


@api.get("/metadata/methods")
def get_methods():
    """
    Get available training methods.

    Returns:
        - 200: Array of method objects with parameters
    """
    return jsonify(supported_method_catalog())


@api.get("/metadata/missing-strategies")
def get_missing_strategies():
    """
    Get available strategies for handling missing values.

    Returns:
        - 200: Array of missing value strategy options
    """
    return jsonify(missing_strategy_catalog())
