import json
import importlib
import re
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from flask import jsonify

from app import app
api = importlib.import_module("app.api")
import app.routes as routes
from model.pipeline import FEATURE_FIELDS


class RouteTests(unittest.TestCase):
    def test_landing_page_combines_project_context_and_method_sections(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Heart Disease Classification Analysis", html)
        self.assertIn("ACS 5513: Applied Machine Learning | University of Oklahoma | Applied Computing M.S.", html)
        self.assertIn("Data Intake &amp; Approval", html)
        self.assertIn("Training &amp; Evaluation", html)
        self.assertIn("Naive Bayes, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)", html)
        self.assertIn("Additional models are planned", html)
        self.assertNotIn("Project Overview", html)
        self.assertNotIn(">Introduction<", html)
        self.assertNotIn("Appendices", html)
        self.assertIn("Project Repository", html)
        self.assertIn('href="https://github.com/ambrown1-ou/ACS5513Project"', html)
        self.assertNotIn("Interactive Laboratory (Jupyter)", html)
        self.assertEqual(len(re.findall(r'class="choice(?:\s|\")', html)), 6)
        self.assertLess(html.find("Heart Disease Classification Analysis"), html.find("ACS 5513: Applied Machine Learning | University of Oklahoma | Applied Computing M.S."))
        self.assertEqual(html.count('class="method-paragraph"'), 2)
        self.assertNotIn('class="method-section"', html)

    def test_site_navigation_exposes_five_top_level_areas(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        topnav = html[html.index('<nav class="topnav">'):html.index('</nav>', html.index('<nav class="topnav">'))]
        for label, path in (
            ("Add Data", "/add-data"),
            ("Explore Data", "/data"),
            ("Train", "/train"),
            ("Results", "/results"),
            ("Predict", "/predict"),
        ):
            self.assertIn(f'href="{path}"', topnav)
            self.assertIn(f">{label}</a>", topnav)

    def test_add_data_and_train_have_distinct_default_pages(self):
        client = app.test_client()

        add_data = client.get("/add-data")
        train = client.get("/train")

        self.assertEqual(add_data.status_code, 200)
        self.assertIn("<h1>Add Data</h1>", add_data.get_data(as_text=True))
        self.assertEqual(train.status_code, 200)
        self.assertIn("<h1>Model Training</h1>", train.get_data(as_text=True))
        self.assertNotIn("<h1>Add Data</h1>", train.get_data(as_text=True))

    def test_data_page_uses_foldable_api_schema_dictionary(self):
        response = app.test_client().get("/data")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-schema-dictionary', html)
        self.assertIn('data-dictionary-rows', html)
        self.assertIn('/static/data_dictionary.js', html)
        self.assertNotIn('{% for entry in data_dictionary %}', html)

    # Rejects traversal filenames while keeping sanitized uploads inside inputs.
    def test_upload_filename_stays_inside_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs_dir = root / "inputs"
            inputs_dir.mkdir()
            info_path = root / "datasets_info.json"
            info_path.write_text("{}", encoding="utf-8")

            with patch.object(api, "INPUTS_DIR", inputs_dir), patch.object(api, "INFO_PATH", info_path):
                response = app.test_client().post(
                    "/train",
                    data={
                        "form_id": "upload",
                        "label": "Probe",
                        "data_file": (BytesIO(b"age,target\n1,0\n"), "..\\outside.csv"),
                    },
                    content_type="multipart/form-data",
                )

            self.assertEqual(response.status_code, 302)
            self.assertFalse((root / "outside.csv").exists())
            self.assertTrue((inputs_dir / "outside.csv").exists())
            self.assertIn("view=manage-data", response.headers["Location"])
            self.assertIn("dataset=outside", response.headers["Location"])

    def test_prepare_dataset_persists_configuration_and_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs_dir = root / "inputs"
            bundled_dir = root / "bundled"
            prepared_dir = root / "prepared"
            inputs_dir.mkdir()
            bundled_dir.mkdir()
            dataset_path = inputs_dir / "uploaded.csv"
            dataset_path.write_text(
                "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,heart_disease_binary\n"
                "63,1,1,145,,1,2,150,0,2.3,3,0,6,0\n"
                "67,1,4,160,286,0,2,108,1,1.5,2,3,3,1\n",
                encoding="utf-8",
            )
            info_path = root / "datasets_info.json"
            info_path.write_text(json.dumps({"uploaded": {"label": "Uploaded Probe"}}), encoding="utf-8")

            with patch.object(api, "INPUTS_DIR", inputs_dir), patch.object(api, "BUNDLED_DATASETS_DIR", bundled_dir), patch.object(
                api, "PREPARED_DATASETS_DIR", prepared_dir
            ), patch.object(api, "INFO_PATH", info_path):
                response = app.test_client().post(
                    "/train",
                    data={
                        "form_id": "prepare",
                        "dataset": "uploaded",
                        "missing_strategy": "impute",
                    },
                )

            metadata = json.loads(info_path.read_text(encoding="utf-8"))
            preparation = metadata["uploaded"]["preparation"]
            self.assertEqual(response.status_code, 302)
            self.assertTrue((prepared_dir / "uploaded__prepared__impute.csv").exists())
            self.assertEqual(preparation["feature_fields"], FEATURE_FIELDS)
            self.assertEqual(preparation["missing_strategy"], "impute")

    def test_add_data_omits_preparation_controls(self):
        response = app.test_client().get("/add-data")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('data-intake-app', html)
        self.assertIn('id="field-mapping-stage"', html)
        self.assertIn('id="review-stage"', html)
        self.assertIn('id="ready-stage"', html)
        self.assertIn('Leave a schema field unmapped to exclude it from training.', html)
        self.assertIn('Drop affected rows to remove rows missing that field.', html)
        self.assertNotIn('data-analysis-summary', html)
        self.assertIn('data-upload-modes', html)
        self.assertNotIn('data-upload-feature-sets', html)
        self.assertNotIn('name="feature_set"', html)
        self.assertIn('data-validation-mode-warning', html)
        self.assertIn('data-validation-mode-warning hidden', html)
        self.assertIn('<select name="validation_mode" data-upload-modes required>', html)
        self.assertIn('value="NORMAL" selected', html)
        self.assertIn('value="NO_TEST"', html)
        self.assertNotIn('type="radio"', html)
        self.assertNotIn('Warning: If the data is not properly prepared', html)
        self.assertNotIn('Loading validation modes...', html)
        self.assertNotIn('data-preparation-form', html)
        self.assertNotIn('data-preparation-dataset', html)
        self.assertNotIn('id="dataset-inspect"', html)
        self.assertNotIn('Prepare a dataset', html)

    def test_training_view_preselects_dataset_from_upload_redirect(self):
        response = app.test_client().get("/train?view=train-model&dataset=heart_disease_cleveland_cleaned")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-training-dataset', html)
        self.assertIn('id="training-form-state"', html)
        self.assertIn("heart_disease_cleveland_cleaned", html)

    def test_delete_removes_uploaded_dataset_and_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs_dir = root / "inputs"
            bundled_dir = root / "bundled"
            inputs_dir.mkdir()
            bundled_dir.mkdir()
            dataset_path = inputs_dir / "uploaded.csv"
            dataset_path.write_text("age,target\n1,0\n", encoding="utf-8")
            info_path = root / "datasets_info.json"
            info_path.write_text(json.dumps({"uploaded": {"label": "Uploaded Probe"}}), encoding="utf-8")

            with patch.object(api, "INPUTS_DIR", inputs_dir), patch.object(api, "BUNDLED_DATASETS_DIR", bundled_dir), patch.object(api, "INFO_PATH", info_path):
                response = app.test_client().post(
                    "/train",
                    data={"form_id": "delete", "dataset": "uploaded"},
                )

            metadata = json.loads(info_path.read_text(encoding="utf-8"))
            self.assertEqual(response.status_code, 302)
            self.assertFalse(dataset_path.exists())
            self.assertNotIn("uploaded", metadata)

    def test_delete_rejects_bundled_dataset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs_dir = root / "inputs"
            bundled_dir = root / "bundled"
            inputs_dir.mkdir()
            bundled_dir.mkdir()
            dataset_path = bundled_dir / "bundled.csv"
            dataset_path.write_text("age,target\n1,0\n", encoding="utf-8")
            info_path = root / "datasets_info.json"
            info_path.write_text(json.dumps({"bundled": {"label": "Bundled Probe"}}), encoding="utf-8")

            with patch.object(api, "INPUTS_DIR", inputs_dir), patch.object(api, "BUNDLED_DATASETS_DIR", bundled_dir), patch.object(api, "INFO_PATH", info_path):
                response = app.test_client().post(
                    "/train",
                    data={"form_id": "delete", "dataset": "bundled"},
                )

            self.assertEqual(response.status_code, 302)
            self.assertTrue(dataset_path.exists())

    # Carries the selected dataset from the custom plot form into the plot builder.
    def test_custom_plot_preserves_selected_dataset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs_dir = root / "inputs"
            inputs_dir.mkdir()
            first = inputs_dir / "first.csv"
            second = inputs_dir / "second.csv"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            info_path = root / "datasets_info.json"
            info_path.write_text(json.dumps({}), encoding="utf-8")

            with patch.object(api, "INPUTS_DIR", inputs_dir), patch.object(api, "INFO_PATH", info_path), patch.object(
                routes, "create_correlation_plot", return_value="plot"
            ) as create_plot:
                response = app.test_client().post(
                    "/data?view=custom&dataset=second",
                    data={
                        "dataset": "second",
                        "dimensions": "2",
                        "x_field": "age",
                        "y_field": "chol",
                        "color_field": "sex",
                    },
                )

            self.assertEqual(response.status_code, 200)
            create_plot.assert_called_once_with(second, ["age", "chol", "sex"], 2)

    def test_custom_plot_xhr_returns_only_plot_fragment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs_dir = root / "inputs"
            inputs_dir.mkdir()
            dataset_path = inputs_dir / "dataset.csv"
            dataset_path.write_text("", encoding="utf-8")
            info_path = root / "datasets_info.json"
            info_path.write_text(json.dumps({}), encoding="utf-8")

            with patch.object(api, "INPUTS_DIR", inputs_dir), patch.object(api, "INFO_PATH", info_path), patch.object(
                routes, "create_correlation_plot", return_value="plot"
            ):
                response = app.test_client().post(
                    "/data?view=custom&dataset=dataset",
                    data={
                        "dataset": "dataset",
                        "dimensions": "2",
                        "x_field": "age",
                        "y_field": "chol",
                        "color_field": "sex",
                    },
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )

            html = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn('id="plot-result"', html)
            self.assertNotIn("<html", html)
            self.assertNotIn('data-plot-form="custom-plot"', html)

    # Exposes the dynamic 2D/3D plot-builder hooks and hides z-axis controls by default.
    def test_custom_plot_form_exposes_dynamic_controls(self):
        response = app.test_client().get("/data?view=custom")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-plot-form="custom-plot"', html)
        self.assertIn('data-plot-axis="x_field"', html)
        self.assertIn('data-plot-axis="color_field"', html)
        self.assertIn('class="dimension-chip"', html)
        self.assertRegex(html, r'id="z-axis-field"[^>]*hidden')
        self.assertIn('syncFieldOptions', html)

    # Returns a client error before model execution for incomplete API input.
    def test_api_rejects_incomplete_payload(self):
        response = app.test_client().post("/api/predict", json={"unexpected": 1})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing fields", response.get_json()["error"])


    def test_training_form_exposes_model_specific_parameters(self):
        response = app.test_client().get("/train?view=train-model")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('id="training-parameter-fields"', html)
        self.assertIn('data-api-form="training"', html)
        self.assertIn('form_builder.js', html)
        self.assertNotIn('data-parameter-for-method=', html)
        self.assertNotIn('name="n_neighbors"', html)

        methods = app.test_client().get("/api/metadata/methods").get_json()
        knn = next(method for method in methods if method["value"] == "knn")
        self.assertEqual(knn["params"][0]["name"], "n_neighbors")

    def test_training_form_includes_added_method_options(self):
        response = app.test_client().get("/train?view=train-model")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        methods = app.test_client().get("/api/metadata/methods").get_json()
        random_forest = next(method for method in methods if method["value"] == "random_forest")
        self.assertIn("n_estimators", [parameter["name"] for parameter in random_forest["params"]])
        self.assertNotIn('value="random_forest"', html)
        self.assertNotIn('name="n_estimators"', html)
        self.assertNotIn('name="feature_set"', html)
        self.assertNotIn('name="missing_strategy"', html)
        self.assertIn('data-training-guide', html)

    def test_prediction_form_uses_api_metadata_shell(self):
        response = app.test_client().get("/predict")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-api-form="prediction"', html)
        self.assertIn('data-prediction-model', html)
        self.assertIn('data-prediction-fields', html)
        self.assertIn('form_builder.js', html)
        self.assertNotIn('name="age"', html)

        field_definitions = app.test_client().get("/api/metadata/field-definitions").get_json()
        self.assertIn("age", field_definitions)
        self.assertIn("minimum", field_definitions["age"])

    def test_training_results_page_renders_full_parameter_table(self):
        result = {
            "dataset_key": "heart",
            "method": "knn",
            "rows": 100,
            "cv_folds": 5,
            "random_state": 42,
            "accuracy": 0.85,
            "accuracy_std": 0.03,
            "precision": 0.82,
            "precision_std": 0.04,
            "recall": 0.8,
            "recall_std": 0.05,
            "f1": 0.81,
            "f1_std": 0.04,
            "model_id": "heart_knn_001",
            "model_path": "storage/models/heart.joblib",
            "created_at": "2026-08-01T00:00:00+00:00",
        }

        prepared_dataset = {
            "label": "Heart",
            "path": Path("storage/datasets/uploads/heart.csv"),
            "prepared_path": Path("storage/datasets/prepared/heart__prepared__drop.csv"),
            "training_ready": True,
            "preparation": {"feature_fields": ["age", "chol"], "missing_strategy": "drop"},
        }
        with patch.object(routes, "get_datasets", return_value={"heart": prepared_dataset}), patch.object(
            api, "train_model_api", side_effect=lambda: (jsonify(result), 201)
        ), patch.object(routes, "load_model", return_value=object()), patch.object(
            routes, "extract_feature_importances", return_value={"age": 0.5, "chol": 0.5}
        ):
            response = app.test_client().post(
                "/train",
                data={
                    "form_id": "train",
                    "dataset": "heart",
                    "method": "knn",
                    "cv_folds": "5",
                    "random_state": "42",
                    "n_neighbors": "5",
                },
            )

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("model_id", html)
        self.assertIn("heart_knn_001", html)
        self.assertIn("model_path", html)
        self.assertIn("storage/models/heart.joblib", html)

    def test_results_delete_delegates_to_model_registry(self):
        with patch.object(api, "delete_registered_model", return_value={"model_id": "heart_knn_001"}) as delete_model:
            response = app.test_client().post(
                "/results",
                data={"form_id": "delete", "model_id": "heart_knn_001"},
            )

        self.assertEqual(response.status_code, 302)
        delete_model.assert_called_once_with("heart_knn_001")


if __name__ == "__main__":
    unittest.main()