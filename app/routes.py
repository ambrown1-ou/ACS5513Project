import json
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from model import (
    FEATURE_FIELDS,
    list_registered_models,
    load_model,
    load_training_registry,
    model_exists,
    predict,
    train_model,
)
from model.visualization import (
    create_correlation_matrix_plot,
    create_correlation_plot,
    create_distribution_plot,
    create_training_metrics_plot,
    create_strongest_correlations_plot,
)


web = Blueprint("web", __name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "inputs" / "heart_disease_cleveland_cleaned.csv"
DEFAULT_DATASET_KEY = DATA_PATH.stem
INFO_PATH = PROJECT_ROOT / "datasets_info.json"
INPUTS_DIR = PROJECT_ROOT / "inputs"

DATA_DICTIONARY = [
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

PREDICTION_FIELD_OPTIONS = {
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

FIELD_DEFINITIONS = {
    item["field"]: {
        "label": item["label"],
        "desc": item["description"],
        "domain": item["domain"],
        "options": PREDICTION_FIELD_OPTIONS.get(item["field"]),
    }
    for item in DATA_DICTIONARY
    if item["field"] != "target"
}


# Validates an upload name and returns a collision-safe path inside the inputs directory.
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


# Loads dataset metadata and builds the list of available CSV inputs.
def get_datasets():
    # Load metadata
    metadata = {}
    if INFO_PATH.exists():
        try:
            with open(INFO_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass

    # Scan inputs directory
    datasets = {}
    if INPUTS_DIR.exists():
        for csv_file in INPUTS_DIR.glob("*.csv"):
            key = csv_file.stem
            info = metadata.get(key, {
                "label": key.replace("_", " ").title(),
                "source": "Local Upload",
                "description": "User-supplied dataset.",
                "link": "#"
            })
            datasets[key] = {
                "label": info.get("label", key),
                "path": csv_file,
                "source": info.get("source", "Unknown"),
                "description": info.get("description", ""),
                "link": info.get("link", "#")
            }
    return datasets

# Renders the landing page and highlights the current model availability.
@web.get("/")
def index():
    return render_template("index.html", model_exists=model_exists())

# Loads the selected dataset view and builds the requested exploratory plot.
@web.route("/data", methods=["GET", "POST"])
def view_data():
    datasets = get_datasets()
    dataset_key = request.args.get("dataset") or request.form.get("dataset") or DEFAULT_DATASET_KEY
    if dataset_key not in datasets:
        # Fallback to first available if cleveland isn't there
        dataset_key = next(iter(datasets.keys())) if datasets else None
    
    if not dataset_key:
        flash("No datasets available. Please upload one.", "warning")
        return redirect(url_for("web.train"))
        
    dataset = datasets[dataset_key]
    view = request.args.get("view", "distributions")
    if view not in {"distributions", "matrix", "strongest", "custom"}:
        view = "distributions"
    
    plot_image = None
    plot_url = None
    strongest_pairs = []
    dimensions = request.form.get("dimensions", "2")
    selected_fields = {
        "x_field": "",
        "y_field": "",
        "z_field": "",
        "color_field": "",
    }
    
    try:
        if view == "custom":
            # Collect the requested axes and color field for the custom plot.
            selected_fields = {
                "x_field": request.form.get("x_field", ""),
                "y_field": request.form.get("y_field", ""),
                "z_field": request.form.get("z_field", ""),
                "color_field": request.form.get("color_field", ""),
            }
            plot_fields = [
                selected_fields["x_field"],
                selected_fields["y_field"],
                selected_fields["color_field"],
            ]
            if dimensions == "3":
                plot_fields.insert(2, selected_fields["z_field"])
            plot_image = create_correlation_plot(dataset["path"], plot_fields, int(dimensions))
        elif view == "distributions":
            plot_image = create_distribution_plot(dataset["path"])
        elif view == "matrix":
            plot_image = create_correlation_matrix_plot(dataset["path"])
        else:
            plot_image, strongest_pairs = create_strongest_correlations_plot(dataset["path"])
    except (ValueError, TypeError, FileNotFoundError) as error:
        flash(str(error), "error")
    
    return render_template(
        "data.html",
        view=view,
        dataset_key=dataset_key,
        dataset=dataset,
        datasets=datasets,
        dimensions=dimensions,
        fields=FEATURE_FIELDS,
        selected_fields=selected_fields,
        plot_image=plot_image,
        plot_url=plot_url,
        strongest_pairs=strongest_pairs,
        data_dictionary=DATA_DICTIONARY,
    )


# Handles dataset uploads, model training submissions, and the training results page.
@web.route("/train", methods=["GET", "POST"])
def train():
    datasets = get_datasets()
    if request.method == "GET":
        return render_template("train.html", datasets=datasets)

    form_id = request.form.get("form_id")

    if form_id == "upload":
        uploaded_file = request.files.get("data_file")
        if not uploaded_file or not uploaded_file.filename:
            flash("Choose a CSV file to upload.", "error")
            return render_template("train.html", datasets=datasets)

        try:
            file_path = _safe_upload_path(uploaded_file.filename)
            uploaded_file.save(file_path)
        except (OSError, ValueError) as error:
            flash(str(error), "error")
            return render_template("train.html", datasets=datasets)

        filename = file_path.name

        # Save metadata
        metadata = {}
        if INFO_PATH.exists():
            with open(INFO_PATH, "r") as f:
                metadata = json.load(f)
        
        key = Path(filename).stem
        metadata[key] = {
            "label": request.form.get("label", key),
            "source": request.form.get("source", "User Upload"),
            "description": request.form.get("description", ""),
            "link": "#"
        }
        
        with open(INFO_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        
        flash(f"Dataset '{filename}' uploaded successfully.", "success")
        return redirect(url_for("web.train"))

    elif form_id == "train":
        dataset_key = request.form.get("dataset")
        if not dataset_key or dataset_key not in datasets:
            flash("Please select a valid dataset for training.", "error")
            return render_template("train.html", datasets=datasets)

        dataset_path = datasets[dataset_key]["path"]
        
        try:
            method = request.form.get("method", "knn")
            cv_folds = int(request.form.get("cv_folds", 5))
            random_state = int(request.form.get("random_state", 42))
            
            kwargs = {}
            if method == "knn":
                kwargs["n_neighbors"] = int(request.form.get("n_neighbors", 5))
            
            result = train_model(
                dataset_path,
                method=method,
                random_state=random_state,
                cv_folds=cv_folds,
                dataset_key=dataset_key,
                **kwargs
            )
            metrics_plot = create_training_metrics_plot(
                result,
                title=f"{result['method'].replace('_', ' ').title()} Training Metrics",
            )
            return render_template("train_result.html", result=result, metrics_plot=metrics_plot)
        except (OSError, ValueError, TypeError) as error:
            flash(str(error), "error")
            return render_template("train.html", datasets=datasets)

    return redirect(url_for("web.train"))


# Renders the patient prediction form and returns the model output for a single case.
@web.route("/predict", methods=["GET", "POST"])
def prediction():
    available_models = list_registered_models()
    
    if request.method == "GET":
        return render_template("predict.html", fields=FIELD_DEFINITIONS, models=available_models)

    try:
        model_id = request.form.get("model_id")
        if not model_id:
            raise ValueError("Please select a model to test.")
        
        values = {field: request.form[field] for field in FEATURE_FIELDS}
        model = load_model(model_id=model_id)
        prediction_value, probability = predict(model, values)
    except (KeyError, OSError, ValueError, TypeError) as error:
        flash(str(error), "error")
        return redirect(url_for("web.prediction"))

    return render_template(
        "prediction_result.html",
        prediction=prediction_value,
        probability=probability,
        model_used=model_id
    )


# Loads saved training results and prepares the comparison table.
@web.route("/results")
def view_results():
    datasets = get_datasets()
    try:
        all_results = load_training_registry().get("runs", [])
    except ValueError as error:
        flash(str(error), "error")
        return redirect(url_for("web.index"))
    
    # Flatten results for easier display
    summary = []
    available_dataset_keys = set(datasets.keys())
    for result in all_results:
        ds_name = result.get("dataset_key")
        if ds_name not in available_dataset_keys:
            continue

        ds_label = datasets.get(ds_name, {}).get("label", ds_name.replace("_", " ").title())
        if "error" in result:
            summary.append({
                "model_id": result.get("model_id", "unknown"),
                "dataset": ds_label,
                "method": result.get("method", "Unknown").replace("_", " ").title(),
                "accuracy": "N/A",
                "f1": "N/A",
                "accuracy_std": "",
                "f1_std": "",
                "rows": None,
                "status": "Failed",
                "error": result["error"],
            })
            continue

        summary.append({
            "model_id": result.get("model_id", "unknown"),
            "dataset": ds_label,
            "method": result["method"].replace("_", " ").title(),
            "accuracy": f"{result['accuracy']:.2%}",
            "f1": f"{result['f1']:.2%}",
            "accuracy_std": f"+/- {result.get('accuracy_std', 0):.2%}",
            "f1_std": f"+/- {result.get('f1_std', 0):.2%}",
            "rows": result.get("rows"),
            "status": "Success",
            "error": None,
        })
    
    return render_template("results.html", summary=summary)
