# OU ACS-5513: Heart Disease Clinical Dashboard

This project is a Flask dashboard for exploring, preparing, training, and evaluating machine-learning models on the bundled cleaned Cleveland heart-disease dataset.

The documented dataset is:

- Dataset key: `heart_disease_cleveland_cleaned`
- Schema: `cleveland_v1`
- Schema version: `1`
- Source: Cleveland Clinic Foundation data distributed through the UCI Machine Learning Repository
- File: `resources/datasets/heart_disease_cleveland_cleaned.csv`
- Target: `target`, where `0` is no diagnosed heart disease and `1` is diagnosed heart disease

The application uses the same schema and validation rules for its browser forms, API responses, training pipeline, and prediction inputs.

## Features

- Explore feature distributions, correlations, and custom plots for the Cleveland dataset.
- Inspect the canonical schema and data dictionary.
- Map any desired canonical columns during dataset intake, exclude unmapped schema fields, and approve field-level validation actions before training.
- Train the bundled Cleveland benchmark with Bayes, KNN, and SVM.
- Compare accuracy, precision, recall, and F1 score across fixed five-fold stratified validation.
- Review feature importance when the selected estimator provides it.
- Use a trained model for an individual patient prediction.

## Application Structure

- `app/routes.py` renders the browser pages.
- `app/api.py` provides the JSON API under `/api`.
- `app/static/` contains the browser clients and shared styles.
- `app/templates/` contains the server-rendered page shells.
- `model/schema.py` defines the Cleveland schema and field rules.
- `model/dataset_mapping.py` analyzes, maps, validates, and reviews field data.
- `model/pipeline.py` prepares data, trains models, evaluates metrics, and makes predictions.
- `model/visualization.py` creates the dashboard plots.
- `resources/datasets/` contains the bundled Cleveland CSV and its metadata.
- `storage/` contains all runtime uploads, generated configuration, mapped and prepared datasets, graphics, model artifacts, registry data, and reports.

## Repository Tree

The tree below lists source and project files. Local secrets, caches, and generated model or CSV artifacts are intentionally omitted.

```text
.
|-- .env.example
|-- .gitignore
|-- .python-version
|-- Procfile
|-- README.md
|-- requirements.txt
|-- run.py
|-- app/
|   |-- __init__.py
|   |-- api.py
|   |-- remote_fileshare.py
|   |-- routes.py
|   |-- static/
|   |   |-- data_dictionary.js
|   |   |-- field_mapper.js
|   |   |-- form_builder.js
|   |   |-- landing.css
|   |   |-- landing.js
|   |   `-- styles.css
|   `-- templates/
|       |-- base.html
|       |-- data.html
|       |-- data_plot_result.html
|       |-- index.html
|       |-- predict.html
|       |-- prediction_result.html
|       |-- results.html
|       |-- train.html
|       `-- train_result.html
|-- config/
|   |-- __init__.py
|   `-- paths.py
|-- model/
|   |-- __init__.py
|   |-- dataset_mapping.py
|   |-- pipeline.py
|   |-- schema.py
|   `-- visualization.py
|-- resources/
|   `-- datasets/
|       |-- datasets_info.json
|       `-- heart_disease_cleveland_cleaned.csv
|-- scripts/
|   |-- analysis/
|   |   |-- dataset_analysis.py
|   |   |-- model_comparison.py
|   |   `-- run_analysis.py
|   `-- training/
|       |-- train_all.py
|       `-- train_model.py
|-- storage/
|   |-- config/
|   |-- datasets/
|   |   |-- uploads/
|   |   |-- mapped/
|   |   `-- prepared/
|   |-- graphics/
|   |-- models/
|   |-- registry/
|   `-- reports/
`-- tests/
    |-- test_api_inspect.py
    |-- test_api_mapping.py
    |-- test_pipeline.py
    |-- test_remote_fileshare.py
    |-- test_routes.py
    `-- test_schema_mapping.py
```

`storage/` contains runtime uploads as well as the deployment model bundle. The Cleveland CSV and its default metadata remain read-only under `resources/datasets/`; the generated `storage/models/*.joblib` files and `storage/registry/training_results.json` are kept visible to version control for inclusion with the deployment.

## Local Setup

### Prerequisites

- Python 3.12 or newer
- A virtual environment is recommended

### Install Dependencies

```text
python -m venv .venv
```

Windows PowerShell:

```text
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS or Linux:

```text
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Dashboard

```text
python run.py
```

Open `http://127.0.0.1:5000` in a browser.

The optional local notebook launcher is available with:

```text
python run.py --jupyter
```

### Run Tests

```text
python -m pytest -q
```

### Run the Cleveland Training Scripts

Train one bundled model:

```text
python scripts/training/train_model.py resources/datasets/heart_disease_cleveland_cleaned.csv --method knn --missing-strategy impute
```

Train the three deployment models:

```text
python scripts/training/train_all.py
```

Run the bundled analysis script:

```text
python scripts/analysis/run_analysis.py
```

Compare KNN, Naive Bayes, and SVM with stratified cross-validation:

```text
python scripts/analysis/model_comparison.py
```

The comparison script evaluates the models with fixed 5-fold stratified validation. It saves comparison plots and data under `storage/graphics/`.

## Cleveland Data Dictionary

| Field | Units | Domain | Description |
| --- | --- | --- | --- |
| `age` | years | 0-120 | Age of the patient |
| `sex` | binary | 0 = Female, 1 = Male | Biological sex of the patient |
| `cp` | coded | 1 = Typical, 2 = Atypical, 3 = Non-anginal, 4 = Asymptomatic | Chest pain type |
| `trestbps` | mmHg | 0-300 | Resting blood pressure on admission |
| `chol` | mg/dL | 0-1000 | Serum cholesterol |
| `fbs` | binary | 0 = False, 1 = True | Whether fasting blood sugar is greater than 120 mg/dL |
| `restecg` | coded | 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy | Resting electrocardiographic result |
| `thalach` | bpm | 0-250 | Maximum heart rate achieved |
| `exang` | binary | 0 = No, 1 = Yes | Exercise-induced angina |
| `oldpeak` | mm | 0-10 | ST depression induced by exercise relative to rest |
| `slope` | coded | 1 = Upsloping, 2 = Flat, 3 = Downsloping | Slope of the peak exercise ST segment |
| `ca` | count | 0-3 | Number of major vessels colored by fluoroscopy |
| `thal` | coded | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect | Thalassemia status |
| `target` | binary | 0 = No disease, 1 = Disease | Diagnosis label used for supervised learning |

The model uses the 13 feature fields from `age` through `thal`; `target` is the classifier field and is not used as a prediction input.

## Training Configuration

### Field Selection

Add Data accepts an explicit mapping from source columns to canonical schema fields. Map one `target` classifier field and at least one feature field; any schema field left unmapped is excluded from training. The dataset stores the ordered `selected_columns`, `feature_fields`, and `target_field` used by preparation and training. Later preparation and training requests use those stored fields rather than accepting a different selection.

### Bundled Training Contract

The application training workflow is intentionally fixed to the bundled Cleveland dataset. It keeps all 13 canonical feature fields, applies median imputation inside each estimator pipeline, uses five stratified validation folds, and fits the saved artifact on all 303 usable rows. This keeps the benchmark and the deployed prediction models on the same contract.

### Supported Methods

The authoritative method names, labels, parameter defaults, and constraints are returned by `GET /api/metadata/methods`.

- `naive_bayes` - Gaussian Naive Bayes, shown as Bayes in the benchmark display.
- `knn` - K-Nearest Neighbors with `n_neighbors=5`.
- `svm` - Support Vector Machine.

Training reports mean accuracy, precision, recall, and F1 values plus their standard deviations across the requested stratified folds.

## API Reference

All API endpoints use the `/api` prefix and return JSON. The examples below use the bundled Cleveland dataset key:

```text
heart_disease_cleveland_cleaned
```

The request lines show the HTTP method and parameters directly. They are not shell commands.

### Health and Dataset Information

#### Check API health

**GET** `/api/health`

Parameters: none.

Expected response:

```json
{
  "status": "ok",
  "model_exists": true
}
```

#### List available datasets

**GET** `/api/datasets`

Parameters: none.

Expected response:

```json
[
  {
    "dataset_key": "heart_disease_cleveland_cleaned",
    "label": "Cleveland Heart Disease Cleaned Dataset",
    "source": "Cleveland Clinic Foundation (Processed)",
    "schema_id": "cleveland_v1",
    "schema_version": "1",
    "intake_status": "legacy",
    "mapping_status": "bundled_schema",
    "training_available": true,
    "training_ready": false,
    "deletable": false
  }
]
```

#### Get one dataset

**GET** `/api/datasets/{dataset_key}`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| path | `dataset_key` | `heart_disease_cleveland_cleaned` |

Expected response:

```json
{
  "dataset_key": "heart_disease_cleveland_cleaned",
  "schema_id": "cleveland_v1",
  "schema_version": "1",
  "intake_status": "legacy",
  "mapping_status": "bundled_schema",
  "validation_mode": "NORMAL",
  "training_available": true,
  "training_ready": false,
  "deletable": false,
  "canonical_path": "resources/datasets/heart_disease_cleveland_cleaned.csv",
  "preparation": {}
}
```

#### Analyze Cleveland source fields

**GET** `/api/datasets/{dataset_key}/field-analysis`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| path | `dataset_key` | `heart_disease_cleveland_cleaned` |

Expected response:

```json
{
  "dataset_key": "heart_disease_cleveland_cleaned",
  "schema": {"schema_id": "cleveland_v1", "version": "1"},
  "total_rows": 303,
  "source_columns": [
    {
      "name": "age",
      "dtype": "int64",
      "rows": 303,
      "missing_count": 0,
      "unique_count": 41,
      "candidates": [
        {
          "schema_field": "age",
          "score": 1.0,
          "reason": "exact"
        }
      ]
    }
  ],
  "classifier_candidates": ["target"]
}
```

Provenance columns such as `source` and `source_row` are retained as metadata but are not offered as schema mapping choices.

#### Inspect training readiness

**GET** `/api/datasets/{dataset_key}/inspect?cv_folds=5`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| path | `dataset_key` | `heart_disease_cleveland_cleaned` |
| query | `cv_folds` | `5`; integer from `2` through `10` |

Expected response:

```json
{
  "dataset_key": "heart_disease_cleveland_cleaned",
  "filename": "heart_disease_cleveland_cleaned.csv",
  "total_rows": 303,
  "target": {
    "column": "target",
    "missing_count": 0,
    "class_counts": {"0": 160, "1": 143}
  },
  "cv": {
    "requested_folds": 5,
    "cv_strategy": "stratified",
    "folds_supported": true
  },
  "intake_status": "legacy",
  "mapping_status": "bundled_schema"
}
```

The legacy alias accepts the dataset as a query parameter:

**GET** `/api/inspect?dataset=heart_disease_cleveland_cleaned&cv_folds=5`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| query | `dataset` | `heart_disease_cleveland_cleaned` |
| query | `cv_folds` | `5` |

It returns the same inspection response as `/api/datasets/{dataset_key}/inspect`.

### Schema and Intake Metadata

#### Get the canonical schema

**GET** `/api/schemas`

The compatibility path `/api/metadata/schemas` returns the same response.

Parameters: none.

Expected response:

```json
[
  {
    "schema_id": "cleveland_v1",
    "version": "1",
    "label": "Cleveland heart disease schema",
    "fields": [
      {
        "field": "age",
        "label": "Age",
        "role": "feature",
        "units": "years",
        "aliases": ["age", "age_years", "patient_age"],
        "minimum": 0,
        "maximum": 120,
        "unit_options": ["years"]
      }
    ]
  }
]
```

The response contains all 14 fields in the data dictionary; the example shows the response shape and one field definition.

#### Get intake controls

**GET** `/api/intake`

The compatibility path `/api/metadata/intake` returns the same response.

Parameters: none.

Expected response:

```json
{
  "validation_modes": [
    {"value": "NORMAL", "label": "Test"},
    {"value": "NO_TEST", "label": "No Test"}
  ],
  "field_issue_actions": [
    {"value": "replace_null", "label": "Replace with NULL (NaN)"},
    {"value": "impute", "label": "Impute"},
    {"value": "drop_rows", "label": "Drop affected rows"},
    {"value": "drop_column", "label": "Drop column"}
  ]
}
```

#### Get the data dictionary

**GET** `/api/metadata/data-dictionary`

Parameters: none.

Expected response:

```json
[
  {
    "field": "age",
    "label": "Age",
    "units": "years",
    "domain": "Numeric (0-120)",
    "role": "feature",
    "description": "Age of the patient."
  }
]
```

The response includes all 14 Cleveland schema fields.

#### Get prediction field definitions

**GET** `/api/metadata/field-definitions`

Parameters: none.

Expected response:

```json
{
  "age": {
    "label": "Age",
    "type": "number",
    "minimum": 0,
    "maximum": 120,
    "step": 1,
    "required": true,
    "options": null
  },
  "sex": {
    "label": "Sex",
    "type": "select",
    "required": true,
    "options": [
      {"value": 0, "label": "0 = Female"},
      {"value": 1, "label": "1 = Male"}
    ]
  }
}
```

The response contains the 13 feature fields and excludes `target` because `target` is the prediction label.

#### Get training methods

**GET** `/api/metadata/methods`

Parameters: none.

Expected response:

```json
[
  {
    "method": "knn",
    "value": "knn",
    "label": "K-Nearest Neighbors",
    "needs_scaling": true,
    "supports_nan": false,
    "params": [
      {
        "name": "n_neighbors",
        "type": "int",
        "default": 5,
        "minimum": 1,
        "maximum": 50,
        "step": 1
      }
    ]
  }
]
```

The response contains the parameter catalog for every supported method.

#### Get missing-value strategies

**GET** `/api/metadata/missing-strategies`

Parameters: none.

Expected response:

```json
[
  {"value": "drop", "label": "Drop incomplete rows"},
  {"value": "impute", "label": "Median imputation"},
  {"value": "native", "label": "Native missing-value support"}
]
```

### Models

#### List trained models

**GET** `/api/models`

Parameters: none.

Expected response:

```json
[
  {
    "model_id": "model_123abc",
    "display_name": "Cleveland Heart Disease Cleaned Dataset - K-Nearest Neighbors",
    "dataset_key": "heart_disease_cleveland_cleaned",
    "method": "knn",
    "feature_fields": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    "missing_strategy": "drop",
    "accuracy": 0.85,
    "precision": 0.84,
    "recall": 0.83,
    "f1": 0.83,
    "rows": 303
  }
]
```

The current response contains all registered model records. The example shows the Cleveland dataset shape.

### Write Endpoints

Write endpoints accept JSON unless a request is explicitly described as form data. The request and response shapes below use the bundled Cleveland dataset key.

#### Upload a dataset

**POST** `/api/datasets/upload`

This endpoint accepts multipart form data. The required field is `data_file`; optional fields include `label`, `source`, `description`, `schema_id`, and `validation_mode`. Normal uploads continue to field mapping and review. `NO_TEST` uploads are trusted immediately and use the canonical fields present in the file. Upload metadata includes `source_columns` and 1-based `source_row_ids`; approved datasets additionally expose `selected_columns`, `feature_fields`, `accepted_row_ids`, and `dropped_row_ids`.

#### Apply Cleveland field mapping

**POST** `/api/datasets/{dataset_key}/field-mapping`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| path | `dataset_key` | `heart_disease_cleveland_cleaned` |
| JSON | `schema_id` | `cleveland_v1` |
| JSON | `mapping` | One entry for each selected source-to-schema field mapping |

Request parameters:

```json
{
  "schema_id": "cleveland_v1",
  "mapping": [
    {
      "source_column": "age",
      "schema_field": "age",
      "source_unit": "years"
    },
    {
      "source_column": "target",
      "schema_field": "target",
      "source_unit": "binary"
    }
  ]
}
```

Expected response:

```json
{
  "dataset_key": "heart_disease_cleveland_cleaned",
  "intake_status": "review",
  "mapping_status": "mapped",
  "mapping": {
    "mapped_schema_fields": ["age", "target"],
    "unmapped_schema_fields": ["sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    "selected_columns": ["age", "target"],
    "feature_fields": ["age"],
    "target_field": "target",
    "classifier_mapped": true
  },
  "report": {
    "total_rows": 303,
    "missing_schema_fields": [],
    "unmapped_source_columns": [],
    "fields": []
  }
}
```

The complete response includes `total_rows_before_review` plus per-field missing, conversion, range, and imputation details in `report.fields`. Each report field includes the raw `source_column`, a short human-readable `alias` such as `Blood Pressure` or `Fasting Blood Sugar`, and the schema field's machine-readable `aliases`. A partial mapping is valid: unmapped schema fields are intentionally excluded, but review requires exactly one target and at least one feature.

#### Review mapped field issues

**POST** `/api/datasets/{dataset_key}/review`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| path | `dataset_key` | `heart_disease_cleveland_cleaned` |
| JSON | `field_decisions` | One action for every field with a reported issue |

Request parameters:

```json
{
  "field_decisions": {
    "age": "replace_null",
    "chol": "impute",
    "target": "drop_rows"
  }
}
```

Available field actions are `replace_null`, `impute`, `drop_rows`, and `drop_column`. The `impute` action is accepted only when the validation report marks that field as imputable. `drop_rows` removes every row with a missing, conversion-invalid, or out-of-range value for that specific field; it is not a manual row-selection control. `drop_column` removes that mapped canonical column from the reviewed dataset while preserving the rows. Review must still leave exactly one target field and at least one feature field. The review response records the resulting `selected_row_ids` and `dropped_row_ids`, using the source file's 1-based `source_row` values.

Expected response:

```json
{
  "dataset_key": "heart_disease_cleveland_cleaned",
  "intake_status": "ready",
  "mapping_status": "reviewed",
  "review": {
    "decisions": {
      "age": "replace_null",
      "chol": "impute",
      "target": "drop_rows"
    },
    "rows_before": 303,
    "rows_after": 303,
    "final_row_count": 303,
    "dropped_rows": 0,
    "selected_row_ids": [1, 2, 3],
    "dropped_row_ids": []
  }
}
```

#### Prepare the Cleveland dataset

**POST** `/api/datasets/{dataset_key}/prepare?missing_strategy=drop`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| path | `dataset_key` | `heart_disease_cleveland_cleaned` |
| query | `missing_strategy` | `drop`, `impute`, or `native`; default `drop` |

Expected response:

```json
{
  "dataset_key": "heart_disease_cleveland_cleaned",
  "preparation": {
    "path": "storage/datasets/prepared/heart_disease_cleveland_cleaned__prepared__drop.csv",
    "missing_strategy": "drop",
    "feature_fields": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    "rows": 303,
    "training_row_ids": [1, 2, 3]
  },
  "message": "Dataset prepared with its selected fields and drop handling."
}
```

#### Train a model

**POST** `/api/models/train`

Request parameters:

```json
{
  "dataset": "heart_disease_cleveland_cleaned",
  "method": "knn",
  "missing_strategy": "drop",
  "cv_folds": 5,
  "random_state": 42,
  "n_neighbors": 5
}
```

The method-specific fields come from `GET /api/metadata/methods`.

The returned `feature_fields` and `training_row_ids` come from the dataset's approved intake and preparation metadata. A late feature-selection value is not used to override the approved dataset fields.

Expected response:

```json
{
  "model_id": "model_123abc",
  "dataset_key": "heart_disease_cleveland_cleaned",
  "method": "knn",
  "missing_strategy": "drop",
  "feature_fields": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
  "accuracy": 0.85,
  "precision": 0.84,
  "recall": 0.83,
  "f1": 0.83,
  "rows": 303,
  "cv_folds": 5,
  "random_state": 42,
  "metrics_plot": null,
  "feature_importance_plot": null
}
```

#### Make a prediction

**POST** `/api/predict`

Request parameters:

```json
{
  "model_id": "model_123abc",
  "age": 50,
  "sex": 1,
  "cp": 3,
  "trestbps": 120,
  "chol": 200,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.5,
  "slope": 1,
  "ca": 0,
  "thal": 3
}
```

`model_id` is optional. When it is omitted, the active registered Cleveland model is used. The feature fields and their constraints come from `GET /api/metadata/field-definitions` and the selected model record.

Expected response:

```json
{
  "prediction": 0,
  "probability": 0.72
}
```

#### Delete a trained model

**DELETE** `/api/models/{model_id}`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| path | `model_id` | `model_123abc` |

Expected response:

```json
{
  "message": "Model 'model_123abc' deleted successfully.",
  "model_id": "model_123abc"
}
```

#### Delete a dataset

**DELETE** `/api/datasets/{dataset_key}`

Parameters:

| Location | Name | Value |
| --- | --- | --- |
| path | `dataset_key` | `heart_disease_cleveland_cleaned` |

The bundled Cleveland dataset is read-only and cannot be deleted.

Expected response:

```json
{
  "error": "Bundled datasets cannot be deleted."
}
```

The response status is `403`.

## Error Responses

Errors are JSON objects. Common response shapes include:

```json
{
  "error": "Dataset not found.",
  "code": "DATASET_NOT_FOUND"
}
```

Common codes include:

| Code | Status | Meaning |
| --- | --- | --- |
| `DATASET_NOT_FOUND` | 404 | The requested dataset key is not available |
| `DATASET_INVALID` | 400 | The Cleveland CSV failed validation |
| `INTAKE_REVIEW_REQUIRED` | 409 | Mapping or review must finish before preparation or inspection |
| `MODEL_NOT_FOUND` | 404 | The requested model is not registered |
| `MAPPED_DATASET_NOT_FOUND` | 409 | The mapped canonical artifact is unavailable |
| `INSPECT_FAILED` | 500 | Dataset inspection failed unexpectedly |
| `FIELD_MAPPING_FAILED` | 400 | A field mapping entry is invalid |
| `DATASET_REVIEW_FAILED` | 400 | A field-level review decision is invalid |

## Command-Line Scripts

### `run.py`

Launches the Flask dashboard. `--jupyter` optionally starts JupyterLab alongside it for local Cleveland dataset analysis.

### `scripts/training/train_model.py`

Trains one model on the bundled Cleveland CSV. The positional dataset path defaults to `resources/datasets/heart_disease_cleveland_cleaned.csv`. Supported arguments include `--method`, `--cv-folds`, `--random-state`, and method-specific parameters.

### `scripts/training/train_all.py`

Trains every supported method on the bundled Cleveland dataset.

### `scripts/analysis/run_analysis.py`

Runs the bundled Cleveland exploratory analysis workflow.

---

Created for the University of Oklahoma ACS-5513 Machine Learning course.