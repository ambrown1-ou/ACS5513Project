import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from model import (
    FEATURE_FIELDS,
    METHOD_LABELS,
    extract_feature_importances,
    list_registered_models,
    load_model,
    load_training_registry,
    missing_strategy_catalog,
    method_catalog,
    model_exists,
    predict,
)
from model.visualization import (
    create_correlation_matrix_plot,
    create_correlation_plot,
    create_distribution_plot,
    create_feature_importance_plot,
    create_training_metrics_plot,
    create_strongest_correlations_plot,
)
from config import paths as project_paths

# Import API helper functions
from .api import get_datasets


web = Blueprint("web", __name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPELINE_STAGE = (os.getenv("PIPELINE_STAGE") or "PROD").upper()
PIPELINE_STAGE_LABEL = {
    "DEV": "Development",
    "QA": "Quality Assurance",
    "PROD": "Production",
}.get(PIPELINE_STAGE, "Production")


def _format_model_timestamp(value):
    """Format a timestamp to a readable string."""
    if not value:
        return ""

    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return ""

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.strftime("%b %d, %Y %H:%M UTC").replace(" 0", " ")


def _model_display_name(model, datasets):
    """Generate a display name for a model."""
    dataset_key = model.get("dataset_key", "")
    dataset = datasets.get(dataset_key, {})
    dataset_label = dataset.get("label") or str(dataset_key).replace("_", " ").title()
    method = str(model.get("method", "")).strip().lower()
    method_label = METHOD_LABELS.get(method, method.replace("_", " ").title())
    display_name = f"{dataset_label} — {method_label}"
    timestamp = _format_model_timestamp(model.get("created_at"))
    return f"{display_name} ({timestamp})" if timestamp else display_name


def _models_for_display(models, datasets):
    """Format models for display with display names."""
    return [{**model, "display_name": _model_display_name(model, datasets)} for model in models]


def _api_response_status(response):
    """Read the status code from a Flask response or an endpoint response tuple."""
    if isinstance(response, tuple):
        return response[1]
    return response.status_code


def _training_form_state(methods, form_data=None):
    """Build training form state based on selected method."""
    form_data = form_data or {}
    default_method = methods[0]["value"] if methods else "knn"
    requested_method = form_data.get("method", default_method)
    selected_method = next((item for item in methods if item["value"] == requested_method), None)
    if selected_method is None:
        selected_method = methods[0] if methods else {
            "value": default_method,
            "params": [],
            "supports_nan": False,
        }

    state = {
        "dataset": form_data.get("dataset", ""),
        "method": selected_method["value"],
        "cv_folds": form_data.get("cv_folds", 5),
        "random_state": form_data.get("random_state", 42),
        "missing_strategy": form_data.get("missing_strategy", "drop"),
        "method_has_params": bool(selected_method.get("params")),
        "method_supports_nan": bool(selected_method.get("supports_nan")),
    }

    for parameter in selected_method.get("params", []):
        state[parameter["name"]] = form_data.get(parameter["name"], parameter.get("default"))

    if state["missing_strategy"] == "native" and not state["method_supports_nan"]:
        state["missing_strategy"] = "drop"

    return state


# ============================================================================
# Web Routes - UI Only
# ============================================================================

@web.get("/")
def index():
    """Render the landing page."""
    return render_template(
        "index.html",
        model_exists=model_exists(),
        pipeline_stage=PIPELINE_STAGE,
        pipeline_stage_label=PIPELINE_STAGE_LABEL,
    )


@web.route("/data", methods=["GET", "POST"])
def view_data():
    """Render data exploration page with plots."""
    datasets = get_datasets()

    # Get default dataset - prefer bundled Cleveland dataset
    _default_bundled = project_paths.BUNDLED_DATASETS_DIR / "heart_disease_cleveland_cleaned.csv"
    if _default_bundled.exists():
        DEFAULT_DATASET_KEY = _default_bundled.stem
    else:
        DEFAULT_DATASET_KEY = next(iter(datasets.keys())) if datasets else None

    dataset_key = request.args.get("dataset") or request.form.get("dataset") or DEFAULT_DATASET_KEY
    if dataset_key not in datasets:
        dataset_key = next(iter(datasets.keys())) if datasets else None

    if not dataset_key:
        flash("No datasets available. Please upload one.", "warning")
        return redirect(url_for("web.train"))

    dataset = datasets[dataset_key]
    view = request.args.get("view", "distributions")
    if view not in {"distributions", "matrix", "strongest", "custom"}:
        view = "distributions"

    plot_image = None
    strongest_pairs = []
    dimensions = request.form.get("dimensions", "2")
    selected_fields = {
        "x_field": "",
        "y_field": "",
        "z_field": "",
        "color_field": "",
    }
    missing_strategy = request.args.get("missing_strategy") or request.form.get("missing_strategy") or "impute"

    try:
        if view == "custom":
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
            if request.method == "POST":
                if missing_strategy == "impute":
                    plot_image = create_correlation_plot(dataset["path"], plot_fields, int(dimensions))
                else:
                    plot_image = create_correlation_plot(dataset["path"], plot_fields, int(dimensions), missing_strategy)
        elif view == "distributions":
            if missing_strategy == "impute":
                plot_image = create_distribution_plot(dataset["path"])
            else:
                plot_image = create_distribution_plot(dataset["path"], missing_strategy)
        elif view == "matrix":
            if missing_strategy == "impute":
                plot_image = create_correlation_matrix_plot(dataset["path"])
            else:
                plot_image = create_correlation_matrix_plot(dataset["path"], missing_strategy)
        else:
            if missing_strategy == "impute":
                plot_image, strongest_pairs = create_strongest_correlations_plot(dataset["path"])
            else:
                plot_image, strongest_pairs = create_strongest_correlations_plot(dataset["path"], pair_count=4, missing_strategy=missing_strategy)
    except (ValueError, TypeError, FileNotFoundError) as error:
        flash(str(error), "error")

    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest" and view == "custom":
        return render_template(
            "data_plot_result.html",
            view=view,
            dimensions=dimensions,
            plot_image=plot_image,
        )

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
        strongest_pairs=strongest_pairs,
        missing_strategy=missing_strategy,
    )


@web.route("/add-data", methods=["GET", "POST"])
def add_data():
    """Render the dataset intake page."""
    return train()


@web.route("/train", methods=["GET", "POST"])
def train():
    """Render training page - handles form submissions by calling API."""
    datasets = get_datasets()
    methods = method_catalog()
    missing_strategies = missing_strategy_catalog()

    def render_train_page(current_view, form_data=None):
        form_data = form_data or {}
        selected_dataset = datasets.get(form_data.get("dataset", ""), {})
        preparation = selected_dataset.get("preparation", {})
        strategy_labels = {item["value"]: item["label"] for item in missing_strategies}
        preparation_form = {
            "dataset": form_data.get("dataset", ""),
            "missing_strategy": form_data.get("missing_strategy", preparation.get("missing_strategy", "drop")),
        }
        preparation_status = {
            **preparation,
            "missing_strategy_label": strategy_labels.get(preparation.get("missing_strategy"), ""),
        }
        return render_template(
            "train.html",
            datasets=datasets,
            view=current_view,
            method_catalog=methods,
            missing_strategy_catalog=missing_strategies,
            training_form=_training_form_state(methods, form_data),
            preparation_form=preparation_form,
            selected_dataset=selected_dataset,
            preparation_status=preparation_status,
            embed_flashes=True,
        )

    default_view = "manage-data" if request.path == "/add-data" else "train-model"
    view = request.args.get("view", default_view)
    if view not in {"manage-data", "train-model"}:
        view = default_view

    def manage_page_url(dataset=None):
        if request.path == "/add-data":
            return url_for("web.add_data", **({"dataset": dataset} if dataset else {}))
        values = {"view": "manage-data"}
        if dataset:
            values["dataset"] = dataset
        return url_for("web.train", **values)

    if request.method == "GET":
        form_data = {"dataset": request.args.get("dataset", "")}
        return render_train_page(view, form_data)

    form_id = request.form.get("form_id")

    # All form submissions call the API - web layer just handles responses and redirects
    if form_id == "upload":
        uploaded_file = request.files.get("data_file")
        if not uploaded_file or not uploaded_file.filename:
            flash("Choose a CSV file to upload.", "error")
            return render_train_page("manage-data")

        # Upload via API - could be done via JavaScript fetch, but for simplicity,
        # the form submission still goes through here since we need file handling
        try:
            from .api import upload_dataset as api_upload
            response = api_upload()
            if _api_response_status(response) == 201:
                data = response[0].get_json()
                flash(f"Dataset '{data['filename']}' uploaded successfully. Complete intake review before training.", "success")
                return redirect(manage_page_url(data["dataset_key"]))
            else:
                flash("Upload failed.", "error")
                return render_train_page("manage-data")
        except Exception as error:
            flash(f"Upload error: {error}", "error")
            return render_train_page("manage-data")

    elif form_id == "delete":
        dataset_key = request.form.get("dataset", "")
        dataset = datasets.get(dataset_key)
        if not dataset:
            flash("Please select a valid dataset to delete.", "error")
            return redirect(manage_page_url())

        if not dataset.get("deletable"):
            flash("Bundled datasets cannot be deleted.", "error")
            return redirect(manage_page_url())

        try:
            from .api import delete_dataset as api_delete
            response = api_delete(dataset_key)
            if _api_response_status(response) == 200:
                flash(f"Dataset '{dataset['label']}' deleted.", "success")
            else:
                flash("Could not delete the dataset.", "error")
        except Exception as error:
            flash(f"Delete error: {error}", "error")

        return redirect(manage_page_url())

    elif form_id == "prepare":
        dataset_key = request.form.get("dataset", "")
        if not dataset_key or dataset_key not in datasets:
            flash("Please select a valid dataset to prepare.", "error")
            return render_train_page("manage-data", request.form)

        missing_strategy = request.form.get("missing_strategy", "drop")

        try:
            from .api import prepare_dataset as api_prepare
            response = api_prepare(dataset_key)
            if _api_response_status(response) == 200:
                response_body = response[0] if isinstance(response, tuple) else response
                preparation = response_body.get_json().get("preparation", {})
                selected_fields = ", ".join(preparation.get("feature_fields", []))
                flash(f"Dataset prepared with selected fields ({selected_fields}) and {missing_strategy} handling.", "success")
                return redirect(manage_page_url(dataset_key))
            else:
                flash("Could not prepare the dataset.", "error")
                return render_train_page("manage-data", request.form)
        except Exception as error:
            flash(f"Preparation error: {error}", "error")
            return render_train_page("manage-data", request.form)

    elif form_id == "train":
        dataset_key = request.form.get("dataset")
        if not dataset_key or dataset_key not in datasets:
            flash("Please select a valid dataset for training.", "error")
            return render_train_page("train-model", request.form)

        dataset = datasets[dataset_key]
        if not dataset.get("training_available", dataset.get("training_ready")):
            flash("Complete field mapping and review in Add Data before selecting training parameters.", "error")
            return redirect(manage_page_url(dataset_key))

        try:
            from .api import train_model_api
            method = request.form.get("method", "knn")
            cv_folds = int(request.form.get("cv_folds", 5))
            random_state = int(request.form.get("random_state", 42))

            # Build JSON payload for API call
            payload = {
                "dataset": dataset_key,
                "method": method,
                "cv_folds": cv_folds,
                "random_state": random_state,
            }

            # Add method parameters
            selected_method = next((m for m in methods if m["value"] == method), None)
            if selected_method:
                for parameter in selected_method.get("params", []):
                    if parameter["name"] in request.form:
                        payload[parameter["name"]] = request.form.get(parameter["name"])

            # Call API with JSON payload
            with current_app.test_request_context(
                method='POST',
                json=payload,
                content_type='application/json'
            ):
                response = train_model_api()
                if isinstance(response, tuple) and response[1] == 201:
                    result = response[0].get_json()

                    # Render training result page
                    from model.visualization import create_training_metrics_plot, create_feature_importance_plot

                    metrics_plot = None
                    feature_importance_plot = None

                    try:
                        trained_model = load_model(model_path=result.get("model_path"))
                        feature_fields = result.get("feature_fields", FEATURE_FIELDS)
                        feature_importances = extract_feature_importances(trained_model, feature_fields)
                        if feature_importances:
                            feature_importance_plot = create_feature_importance_plot(
                                feature_importances,
                                title=f"{result['method'].replace('_', ' ').title()} Feature Importance",
                            )
                    except Exception:
                        pass

                    return render_template(
                        "train_result.html",
                        result=result,
                        metrics_plot=metrics_plot,
                        feature_importance_plot=feature_importance_plot,
                        model_name=_model_display_name(result, datasets),
                    )
                else:
                    error_data = response[0].get_json() if isinstance(response, tuple) else {}
                    flash(f"Training failed: {error_data.get('error', 'Unknown error')}", "error")
                    return render_train_page("train-model", request.form)
        except (ValueError, TypeError) as error:
            flash(str(error), "error")
            return render_train_page("train-model", request.form)
        except Exception as error:
            flash(f"Training error: {error}", "error")
            return render_train_page("train-model", request.form)

    return redirect(url_for("web.train", view=view))


@web.route("/predict", methods=["GET", "POST"])
def prediction():
    """Render the prediction shell; the browser fills it from API metadata."""
    if request.method == "GET":
        return render_template("predict.html")

    datasets = get_datasets()
    available_models = _models_for_display(list_registered_models(), datasets)

    try:
        model_id = request.form.get("model_id")
        if not model_id:
            raise ValueError("Please select a model to test.")

        # Find the selected model record
        from model import predict as model_predict
        selected_model = next(
            (m for m in available_models if m.get("model_id") == model_id),
            None,
        )
        selected_fields = selected_model.get("feature_fields", FEATURE_FIELDS) if selected_model else FEATURE_FIELDS
        values = {field: request.form[field] for field in selected_fields}
        model = load_model(model_id=model_id)
        prediction_value, probability = model_predict(model, values, feature_fields=selected_fields)
    except (KeyError, OSError, ValueError, TypeError) as error:
        flash(str(error), "error")
        return redirect(url_for("web.prediction"))

    return render_template(
        "prediction_result.html",
        prediction=prediction_value,
        probability=probability,
        model_used=selected_model["display_name"] if selected_model else "the selected trained model",
    )


@web.route("/results", methods=["GET", "POST"])
def view_results():
    """Render results/models page."""
    datasets = get_datasets()
    if request.method == "POST":
        if request.form.get("form_id") != "delete":
            flash("Invalid model action.", "error")
            return redirect(url_for("web.view_results"))

        model_id = request.form.get("model_id", "")
        try:
            from .api import delete_model_api
            response = delete_model_api(model_id)
            if isinstance(response, tuple) and response[1] == 200:
                flash(f"Model '{model_id}' deleted.", "success")
            else:
                flash("Could not delete the model.", "error")
        except Exception as error:
            flash(f"Delete error: {error}", "error")
        return redirect(url_for("web.view_results"))

    try:
        all_results = load_training_registry().get("runs", [])
    except ValueError as error:
        flash(str(error), "error")
        return redirect(url_for("web.index"))

    # Flatten results for display
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
                "model_name": _model_display_name(result, datasets),
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
            "model_name": _model_display_name(result, datasets),
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
