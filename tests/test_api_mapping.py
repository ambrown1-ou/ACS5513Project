import json
import os
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from app import app
import importlib

api = importlib.import_module("app.api")


class ApiMappingTests(unittest.TestCase):
    def setUp(self):
        self.pipeline_stage = patch.dict(os.environ, {"PIPELINE_STAGE": "QA"})
        self.pipeline_stage.start()
        self.client = app.test_client()
        self.root = TemporaryDirectory()
        root = Path(self.root.name)
        self.inputs = root / "inputs"
        self.bundled = root / "bundled"
        self.mapped = root / "mapped"
        self.prepared = root / "prepared"
        for directory in (self.inputs, self.bundled, self.mapped, self.prepared):
            directory.mkdir()
        self.info_path = root / "datasets_info.json"
        self.info_path.write_text("{}", encoding="utf-8")
        self.patches = patch.multiple(
            api,
            INPUTS_DIR=self.inputs,
            BUNDLED_DATASETS_DIR=self.bundled,
            MAPPED_DATASETS_DIR=self.mapped,
            PREPARED_DATASETS_DIR=self.prepared,
            INFO_PATH=self.info_path,
        )
        self.patches.start()

    def tearDown(self):
        self.patches.stop()
        self.pipeline_stage.stop()
        self.root.cleanup()

    def upload(self, validation_mode="NORMAL"):
        csv_text = (
            "patient_age,gender,chest_pain,resting_bp,cholesterol,fasting_blood_sugar,ecg,max_hr,"
            "exercise_angina,st_depression,st_slope,major_vessels,thalassemia,diagnosis,source\n"
            "63,1,1,19.33,6.03,1,2,150,0,0.23,3,0,6,0,A\n"
            "67,1,4,21.33,7.39,0,2,108,1,0.15,2,3,3,1,B\n"
            "59,1,3,20.00,,1,0,120,0,0.10,2,0,3,0,A\n"
            "121,1,4,21.33,7.39,0,2,129,1,0.26,2,2,7,1,B\n"
        )
        return self.client.post(
            "/api/datasets/upload",
            data={
                "label": "Mapped Probe",
                "source": "Test source",
                "schema_id": "cleveland_v1",
                "validation_mode": validation_mode,
                "data_file": (BytesIO(csv_text.encode("utf-8")), "mapped_probe.csv"),
            },
            content_type="multipart/form-data",
        )

    def test_runtime_metadata_overlays_bundled_defaults_without_mutating_them(self):
        bundled_metadata = {
            "bundled": {
                "label": "Bundled Default",
                "source": "Test bundle",
            }
        }
        bundled_info_path = self.bundled / "datasets_info.json"
        bundled_info_path.write_text(json.dumps(bundled_metadata), encoding="utf-8")
        (self.bundled / "bundled.csv").write_text("age,target\n63,1\n", encoding="utf-8")

        response = self.upload("NO_TEST")

        self.assertEqual(response.status_code, 201, response.get_json())
        self.assertEqual(json.loads(bundled_info_path.read_text(encoding="utf-8")), bundled_metadata)
        runtime_metadata = json.loads(self.info_path.read_text(encoding="utf-8"))
        self.assertIn(response.get_json()["dataset_key"], runtime_metadata)
        self.assertEqual(self.client.get("/api/datasets/bundled").get_json()["label"], "Bundled Default")

    def mapping_entries(self):
        return [
            {"source_column": "patient_age", "schema_field": "age"},
            {"source_column": "gender", "schema_field": "sex"},
            {"source_column": "chest_pain", "schema_field": "cp"},
            {"source_column": "resting_bp", "schema_field": "trestbps", "source_unit": "kPa"},
            {"source_column": "cholesterol", "schema_field": "chol", "source_unit": "mmol/L"},
            {"source_column": "fasting_blood_sugar", "schema_field": "fbs"},
            {"source_column": "ecg", "schema_field": "restecg"},
            {"source_column": "max_hr", "schema_field": "thalach"},
            {"source_column": "exercise_angina", "schema_field": "exang"},
            {"source_column": "st_depression", "schema_field": "oldpeak", "source_unit": "cm"},
            {"source_column": "st_slope", "schema_field": "slope"},
            {"source_column": "major_vessels", "schema_field": "ca"},
            {"source_column": "thalassemia", "schema_field": "thal"},
            {"source_column": "diagnosis", "schema_field": "target"},
        ]

    def test_intake_metadata_exposes_test_and_no_test_modes(self):
        response = self.client.get("/api/intake")

        self.assertEqual(response.status_code, 200)
        modes = response.get_json()["validation_modes"]
        self.assertEqual([(mode["value"], mode["label"]) for mode in modes], [
            ("NORMAL", "Test"),
            ("NO_TEST", "No Test"),
        ])
        self.assertNotIn("feature_sets", response.get_json())
        self.assertEqual(
            [action["value"] for action in response.get_json()["field_issue_actions"]],
            ["replace_null", "impute", "drop_rows", "drop_column"],
        )

        legacy_response = self.client.get("/api/metadata/intake")
        self.assertEqual(legacy_response.status_code, 200)

    def test_schema_short_and_legacy_routes_return_the_same_catalog(self):
        short_response = self.client.get("/api/schemas")
        legacy_response = self.client.get("/api/metadata/schemas")

        self.assertEqual(short_response.status_code, 200)
        self.assertEqual(legacy_response.status_code, 200)
        self.assertEqual(short_response.get_json(), legacy_response.get_json())

    def test_schema_and_data_dictionary_share_the_same_fields(self):
        schema_response = self.client.get("/api/schemas")
        dictionary_response = self.client.get("/api/metadata/data-dictionary")

        self.assertEqual(schema_response.status_code, 200)
        self.assertEqual(dictionary_response.status_code, 200)
        schema = schema_response.get_json()[0]
        dictionary = dictionary_response.get_json()
        self.assertEqual(schema["schema_id"], "cleveland_v1")
        self.assertEqual(
            {field["field"] for field in schema["fields"]},
            {field["field"] for field in dictionary},
        )
        self.assertIn("role", dictionary[0])
        self.assertIn("allowed", next(field for field in dictionary if field["field"] == "sex"))

    def test_normal_upload_requires_mapping_then_review_before_training(self):
        upload = self.upload()
        self.assertEqual(upload.status_code, 201)
        uploaded = upload.get_json()
        dataset_key = uploaded["dataset_key"]
        self.assertEqual(uploaded["intake_status"], "mapping")
        self.assertEqual(uploaded["source_row_ids"], [1, 2, 3, 4])
        self.assertEqual(uploaded["selected_columns"], [])
        self.assertFalse(uploaded["training_available"])

        schemas = self.client.get("/api/schemas")
        self.assertEqual(schemas.status_code, 200)
        self.assertEqual(schemas.get_json()[0]["schema_id"], "cleveland_v1")

        analysis = self.client.get(f"/api/datasets/{dataset_key}/field-analysis")
        self.assertEqual(analysis.status_code, 200)
        source_columns = analysis.get_json()["source_columns"]
        self.assertNotIn("source", {column["name"] for column in source_columns})
        self.assertNotIn("source_row", {column["name"] for column in source_columns})
        age_column = next(column for column in source_columns if column["name"] == "patient_age")
        self.assertEqual(age_column["candidates"][0]["schema_field"], "age")

        blocked = self.client.get(f"/api/datasets/{dataset_key}/inspect")
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(blocked.get_json()["code"], "INTAKE_REVIEW_REQUIRED")

        mapped = self.client.post(
            f"/api/datasets/{dataset_key}/field-mapping",
            json={"schema_id": "cleveland_v1", "mapping": self.mapping_entries()},
        )
        self.assertEqual(mapped.status_code, 200)
        mapping_payload = mapped.get_json()
        self.assertEqual(mapping_payload["intake_status"], "review")
        self.assertEqual(mapping_payload["mapping"]["feature_fields"], [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ])
        self.assertEqual(mapping_payload["report"]["source_row_ids"], [1, 2, 3, 4])
        self.assertEqual(mapping_payload["report"]["total_rows_before_review"], 4)
        self.assertAlmostEqual(
            mapping_payload["report"]["fields"][3]["canonical_unit"] == "mmHg",
            True,
        )
        self.assertGreater(mapping_payload["report"]["rows_with_missing_data"], 0)
        self.assertGreater(mapping_payload["report"]["rows_with_out_of_range_values"], 0)

        reviewed = self.client.post(
            f"/api/datasets/{dataset_key}/review",
            json={"field_decisions": {"age": "drop_rows", "chol": "drop_rows"}},
        )
        self.assertEqual(reviewed.status_code, 200, reviewed.get_json())
        reviewed_payload = reviewed.get_json()
        self.assertEqual(reviewed_payload["intake_status"], "ready")
        self.assertTrue(reviewed_payload["dataset"]["training_available"])
        self.assertEqual(reviewed_payload["review"]["dropped_rows"], 2)
        self.assertEqual(reviewed_payload["review"]["final_row_count"], 2)
        self.assertEqual(reviewed_payload["dataset"]["final_row_count"], 2)
        self.assertEqual(reviewed_payload["review"]["dropped_row_ids"], [3, 4])
        self.assertEqual(reviewed_payload["review"]["total_rows_before_review"], 4)
        self.assertEqual(reviewed_payload["review"]["selected_row_ids"], [1, 2])
        self.assertEqual(reviewed_payload["dataset"]["accepted_row_ids"], [1, 2])
        self.assertEqual(reviewed_payload["dataset"]["feature_fields"], [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ])

        datasets = self.client.get("/api/datasets").get_json()
        dataset = next(item for item in datasets if item["dataset_key"] == dataset_key)
        self.assertEqual(dataset["intake_status"], "ready")
        self.assertTrue(dataset["training_available"])
        self.assertEqual(dataset["final_row_count"], 2)
        self.assertEqual(dataset["intake"]["review_decisions"], {"age": "drop_rows", "chol": "drop_rows"})

        inspected = self.client.get(f"/api/datasets/{dataset_key}/inspect?cv_folds=2")
        self.assertEqual(inspected.status_code, 200)
        self.assertEqual(inspected.get_json()["dataset_key"], dataset_key)

        prepared = self.client.post(
            f"/api/datasets/{dataset_key}/prepare",
            json={"missing_strategy": "impute"},
        )
        self.assertEqual(prepared.status_code, 200)
        self.assertNotIn("feature_set", prepared.get_json()["preparation"])
        self.assertEqual(prepared.get_json()["preparation"]["feature_fields"], [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ])

        metadata = json.loads(self.info_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata[dataset_key]["intake_status"], "ready")
        self.assertEqual(metadata[dataset_key]["final_row_count"], 2)
        self.assertEqual(metadata[dataset_key]["dropped_row_ids"], [3, 4])
        self.assertTrue((self.mapped / f"{dataset_key}__mapped.csv").exists())
        self.assertTrue((self.mapped / f"{dataset_key}__reviewed.csv").exists())

    def test_no_test_upload_skips_mapping_and_analysis(self):
        upload = self.upload("NO_TEST")
        self.assertEqual(upload.status_code, 201)
        payload = upload.get_json()
        self.assertEqual(payload["intake_status"], "trusted")
        self.assertEqual(payload["final_row_count"], 4)
        self.assertTrue(payload["training_available"])

        analysis = self.client.get(f"/api/datasets/{payload['dataset_key']}/field-analysis")
        self.assertEqual(analysis.status_code, 409)
        self.assertEqual(analysis.get_json()["code"], "NO_TEST_MAPPING_SKIPPED")

    def test_review_accepts_field_level_issue_decisions(self):
        upload = self.upload()
        self.assertEqual(upload.status_code, 201)
        dataset_key = upload.get_json()["dataset_key"]
        mapped = self.client.post(
            f"/api/datasets/{dataset_key}/field-mapping",
            json={"schema_id": "cleveland_v1", "mapping": self.mapping_entries()},
        )
        self.assertEqual(mapped.status_code, 200)
        report = mapped.get_json()["report"]
        field_decisions = {
            field["schema_field"]: "replace_null"
            for field in report["fields"]
            if field["missing_count"] or field["invalid_count"]
        }

        reviewed = self.client.post(
            f"/api/datasets/{dataset_key}/review",
            json={"field_decisions": field_decisions},
        )
        self.assertEqual(reviewed.status_code, 200, reviewed.get_json())
        self.assertEqual(reviewed.get_json()["review"]["decisions"], field_decisions)

    def test_review_can_drop_a_column_and_keep_remaining_rows(self):
        upload = self.upload()
        self.assertEqual(upload.status_code, 201)
        dataset_key = upload.get_json()["dataset_key"]
        mapped = self.client.post(
            f"/api/datasets/{dataset_key}/field-mapping",
            json={"schema_id": "cleveland_v1", "mapping": self.mapping_entries()},
        )
        self.assertEqual(mapped.status_code, 200)

        reviewed = self.client.post(
            f"/api/datasets/{dataset_key}/review",
            json={"field_decisions": {"age": "drop_column", "chol": "drop_rows"}},
        )

        self.assertEqual(reviewed.status_code, 200, reviewed.get_json())
        payload = reviewed.get_json()
        self.assertEqual(payload["review"]["selected_row_ids"], [1, 2, 4])
        self.assertEqual(payload["review"]["dropped_row_ids"], [3])
        self.assertNotIn("age", payload["dataset"]["selected_columns"])
        self.assertNotIn("age", payload["dataset"]["feature_fields"])
        reviewed_path = self.mapped / f"{dataset_key}__reviewed.csv"
        self.assertNotIn("age", pd.read_csv(reviewed_path).columns)

    def test_approved_dataset_trains_without_preparation_artifact(self):
        upload = self.upload("NO_TEST")
        self.assertEqual(upload.status_code, 201)
        dataset_key = upload.get_json()["dataset_key"]
        training_result = {
            "model_id": "probe-model",
            "dataset_key": dataset_key,
            "method": "knn",
            "missing_strategy": "impute",
            "feature_fields": ["age", "sex"],
            "model_path": "storage/models/probe-model.joblib",
        }

        with patch.object(api, "train_model", return_value=training_result) as train:
            response = self.client.post(
                "/api/models/train",
                json={
                    "dataset": dataset_key,
                    "method": "knn",
                    "cv_folds": 2,
                    "random_state": 42,
                },
            )

        self.assertEqual(response.status_code, 201, response.get_json())
        train.assert_called_once()
        self.assertEqual(Path(train.call_args.args[0]), self.inputs / "mapped_probe.csv")
        self.assertEqual(train.call_args.kwargs["feature_fields"], [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ])
        self.assertEqual(train.call_args.kwargs["missing_strategy"], "impute")

    def test_production_blocks_model_and_dataset_mutations(self):
        with patch.dict(os.environ, {"PIPELINE_STAGE": "PROD"}), patch.object(api, "train_model") as train, patch.object(
            api, "delete_registered_model"
        ) as delete_model:
            responses = [
                self.client.post("/api/models/train", json={}),
                self.client.post("/api/models/train-all", json={}),
                self.client.delete("/api/datasets/uploaded"),
                self.client.delete("/api/models/model-id"),
            ]

        for response in responses:
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.get_json()["error"], api.PRODUCTION_WRITE_ERROR)
            self.assertEqual(response.get_json()["code"], "PRODUCTION_WRITE_FORBIDDEN")
        train.assert_not_called()
        delete_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
