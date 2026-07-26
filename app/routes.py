import json
from pathlib import Path

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from model import FEATURE_FIELDS, load_model, predict, train_model
from model.visualization import (
    create_correlation_matrix_plot,
    create_correlation_plot,
    create_distribution_plot,
    create_strongest_correlations_plot,
)


web = Blueprint("web", __name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "inputs" / "Heart_disease_cleveland_new.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "heart_disease_model.joblib"
INFO_PATH = PROJECT_ROOT / "datasets_info.json"
INPUTS_DIR = PROJECT_ROOT / "inputs"

DATA_DICTIONARY = [
    {"field": "age", "label": "Age", "units": "years", "domain": "Numeric (29-77)", "description": "Age of the patient."},
    {"field": "sex", "label": "Sex", "units": "binary", "domain": "0=Female, 1=Male", "description": "Biological sex of the patient."},
    {"field": "cp", "label": "Chest Pain Type", "units": "coded", "domain": "1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic", "description": "Type of chest pain experienced."},
    {"field": "trestbps", "label": "Resting BP", "units": "mm Hg", "domain": "Numeric (94-200)", "description": "Resting blood pressure on admission."},
    {"field": "chol", "label": "Serum Cholestrol", "units": "mg/dl", "domain": "Numeric (126-564)", "description": "Serum cholesterol measurements."},
    {"field": "fbs", "label": "Fasting Blood Sugar", "units": "binary", "domain": ">120 mg/dl (1=True, 0=False)", "description": "Fasting blood sugar level."},
    {"field": "restecg", "label": "Resting ECG", "units": "coded", "domain": "0=Normal, 1=ST-T wave, 2=LV hypertrophy", "description": "Resting electrocardiographic results."},
    {"field": "thalach", "label": "Max Heart Rate", "units": "bpm", "domain": "Numeric (71-202)", "description": "Maximum heart rate achieved."},
    {"field": "exang", "label": "Exercise Angina", "units": "binary", "domain": "1=Yes, 0=No", "description": "Exercise induced angina."},
    {"field": "oldpeak", "label": "ST Depression", "units": "mm", "domain": "Numeric (0-6.2)", "description": "ST depression induced by exercise relative to rest."},
    {"field": "slope", "label": "ST Slope", "units": "coded", "domain": "1=Upsloping, 2=Flat, 3=Downsloping", "description": "The slope of the peak exercise ST segment."},
    {"field": "ca", "label": "Colored Vessels", "units": "count", "domain": "0-3 vessels", "description": "Number of major vessels colored by fluoroscopy."},
    {"field": "thal", "label": "Thalassemia", "units": "coded", "domain": "3=Normal, 6=Fixed, 7=Reversable", "description": "A blood disorder called thalassemia."},
    {"field": "target", "label": "Diagnosis", "units": "binary", "domain": "0=<50% diameter, 1=>50% diameter", "description": "Heart disease diagnosis status."}
]

FIELD_DEFINITIONS = {item["field"]: {"label": item["label"], "desc": item["description"]} for item in DATA_DICTIONARY if item["field"] != "target"}

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

@web.get("/")
def index():
    return render_template("index.html", model_exists=MODEL_PATH.exists())

@web.route("/data", methods=["GET", "POST"])
def view_data():
    datasets = get_datasets()
    dataset_key = request.args.get("dataset", "Heart_disease_cleveland_new")
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

    df = pd.read_csv(dataset["path"])
    
    plot_image = None
    plot_url = None
    strongest_pairs = []
    dimensions = request.form.get("dimensions", "2")
    selected_fields = []
    
    try:
        if view == "custom":
            selected_fields = [request.form.get("x_field", ""), request.form.get("y_field", "")]
            if dimensions == "3":
                selected_fields.append(request.form.get("z_field", ""))
            plot_image = create_correlation_plot(dataset["path"], selected_fields, int(dimensions))
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

        filename = uploaded_file.filename
        file_path = INPUTS_DIR / filename
        uploaded_file.save(file_path)

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
            test_size = float(request.form.get("test_size", 0.2))
            random_state = int(request.form.get("random_state", 42))
            
            kwargs = {}
            if method == "knn":
                kwargs["n_neighbors"] = int(request.form.get("n_neighbors", 5))
            
            result = train_model(
                dataset_path,
                method=method,
                test_size=test_size,
                random_state=random_state,
                **kwargs
            )
            return render_template("train_result.html", result=result)
        except (ValueError, TypeError) as error:
            flash(str(error), "error")
            return render_template("train.html", datasets=datasets)

    return redirect(url_for("web.train"))


@web.route("/predict", methods=["GET", "POST"])
def prediction():
    output_dir = PROJECT_ROOT / "outputs"
    available_models = sorted([f.name for f in output_dir.glob("heart_disease_*.joblib")])
    
    if request.method == "GET":
        return render_template("predict.html", fields=FIELD_DEFINITIONS, models=available_models)

    try:
        model_name = request.form.get("model_name")
        if not model_name:
            raise ValueError("Please select a model to test.")
        
        values = {field: float(request.form[field]) for field in FEATURE_FIELDS}
        model = load_model(output_dir / model_name)
        prediction_value, probability = predict(model, values)
    except (KeyError, ValueError, TypeError, FileNotFoundError) as error:
        flash(str(error), "error")
        return redirect(url_for("web.prediction"))

    return render_template(
        "prediction_result.html",
        prediction=prediction_value,
        probability=probability,
        model_used=model_name
    )


@web.route("/results")
def view_results():
    results_path = PROJECT_ROOT / "outputs" / "training_results.json"
    if not results_path.exists():
        flash("Training results not found. Please run the training process first.", "warning")
        return redirect(url_for("web.index"))
    
    datasets = get_datasets()
    with open(results_path, "r") as f:
        all_results = json.load(f)
    
    # Flatten results for easier display
    summary = []
    for ds_name, methods in all_results.items():
        ds_label = datasets.get(ds_name, {}).get("label", ds_name.replace("_", " ").title())
        for method_name, metrics in methods.items():
            if "error" in metrics:
                summary.append({
                    "dataset": ds_label,
                    "method": method_name.replace("_", " ").title(),
                    "accuracy": "N/A",
                    "f1": "N/A",
                    "status": "Failed",
                    "error": metrics["error"]
                })
            else:
                summary.append({
                    "dataset": ds_label,
                    "method": method_name.replace("_", " ").title(),
                    "accuracy": f"{metrics['accuracy']:.2%}",
                    "f1": f"{metrics['f1']:.2%}",
                    "rows": metrics["rows"],
                    "status": "Success",
                    "error": None
                })
    
    return render_template("results.html", summary=summary)
