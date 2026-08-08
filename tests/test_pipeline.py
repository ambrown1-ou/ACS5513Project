import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from model.pipeline import (
    delete_registered_model,
    FEATURE_FIELDS,
    extract_feature_importances,
    load_model,
    load_training_data,
    predict,
    prepare_training_data,
    inspect_training_data,
    train_model,
    validate_feature_values,
)
from config import paths as project_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = project_paths.BUNDLED_DATASETS_DIR / "heart_disease_cleveland_cleaned.csv"


class PipelineTests(unittest.TestCase):
    def selected_feature_fields(self):
        return ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]

    def write_strategy_collapse_dataset(self, directory):
        csv_text = "\n".join(
            [
                "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,heart_disease_binary",
                "63,1,1,145,,1,2,150,0,2.3,3,0,6,0",
                "60,1,2,130,,0,0,160,0,0.0,2,0,3,0",
                "67,1,4,160,286,0,2,108,1,1.5,2,3,3,1",
                "59,1,3,150,260,1,0,120,0,1.0,2,0,3,1",
            ]
        )
        data_path = Path(directory) / "strategy_collapse.csv"
        data_path.write_text(csv_text, encoding="utf-8")
        return data_path

    # Returns a representative valid prediction payload.
    def valid_values(self):
        return {
            "age": 63,
            "sex": 1,
            "cp": 1,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 2,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0,
            "thal": 6,
        }

    # Rejects feature values outside the documented clinical domains.
    def test_validate_feature_values_rejects_out_of_range_age(self):
        values = self.valid_values()
        values["age"] = 121

        with self.assertRaisesRegex(ValueError, "age"):
            validate_feature_values(values)

    # Keeps the bundled dataset normalized to the shared feature contract.
    def test_load_training_data_normalizes_target_column(self):
        data = load_training_data(DATA_PATH)

        self.assertEqual(list(data.columns), FEATURE_FIELDS + ["target"])
        self.assertEqual(set(data["target"].unique()), {0, 1})

    # Produces cross-validated metrics and a unique artifact without registry mutation.
    def test_train_model_returns_cross_validation_metrics(self):
        result = train_model(
            DATA_PATH,
            method="knn",
            feature_fields=self.selected_feature_fields(),
            cv_folds=3,
            register=False,
        )
        artifact = PROJECT_ROOT / result["model_path"]
        try:
            self.assertEqual(result["cv_folds"], 3)
            self.assertEqual(result["cv_strategy"], "stratified")
            self.assertEqual(result["group_count"], 1)
            self.assertIn("accuracy_std", result)
            self.assertIn("f1_std", result)
            self.assertTrue(artifact.exists())
        finally:
            artifact.unlink(missing_ok=True)

    def test_delete_registered_model_removes_artifact_and_reassigns_active_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / "models"
            artifact_dir.mkdir()
            registry_path = root / "training_results.json"
            deleted_artifact = artifact_dir / "deleted.joblib"
            retained_artifact = artifact_dir / "retained.joblib"
            deleted_artifact.write_bytes(b"deleted")
            retained_artifact.write_bytes(b"retained")
            registry_path.write_text(
                json.dumps(
                    {
                        "active_model_id": "deleted",
                        "runs": [
                            {"model_id": "deleted", "model_path": str(deleted_artifact)},
                            {"model_id": "retained", "model_path": str(retained_artifact)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("model.pipeline.REGISTRY_PATH", registry_path), patch("model.pipeline.MODEL_ARTIFACTS_DIR", artifact_dir):
                deleted = delete_registered_model("deleted")

            updated_registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(deleted["model_id"], "deleted")
            self.assertFalse(deleted_artifact.exists())
            self.assertTrue(retained_artifact.exists())
            self.assertEqual(updated_registry["active_model_id"], "retained")
            self.assertEqual([run["model_id"] for run in updated_registry["runs"]], ["retained"])

    # Supports one of the added tree-based ensemble methods end to end.
    def test_train_model_supports_random_forest(self):
        result = train_model(
            DATA_PATH,
            method="random_forest",
            cv_folds=3,
            register=False,
            n_estimators=10,
            max_depth=5,
            min_samples_leaf=1,
        )
        artifact = PROJECT_ROOT / result["model_path"]
        try:
            self.assertEqual(result["method"], "random_forest")
            self.assertEqual(result["cv_folds"], 3)
            self.assertIn("accuracy_std", result)
            model = load_model(model_path=result["model_path"])
            importances = extract_feature_importances(model, result["feature_fields"])
            self.assertIsNotNone(importances)
            self.assertEqual(len(importances), len(result["feature_fields"]))
            self.assertAlmostEqual(sum(importances.values()), 1.0, places=5)
            self.assertTrue(artifact.exists())
        finally:
            artifact.unlink(missing_ok=True)

    # Trains on an arbitrary explicit field selection and keeps it in the result metadata.
    def test_train_model_supports_explicit_feature_fields(self):
        feature_fields = ["age", "chol", "thalach"]
        result = train_model(
            DATA_PATH,
            method="random_forest",
            feature_fields=feature_fields,
            cv_folds=3,
            register=False,
            n_estimators=10,
            max_depth=5,
            min_samples_leaf=1,
        )
        artifact = PROJECT_ROOT / result["model_path"]
        try:
            self.assertNotIn("feature_set", result)
            self.assertEqual(result["feature_fields"], feature_fields)
            self.assertEqual(result["training_row_ids"], list(range(1, 304)))
            self.assertTrue(artifact.exists())
        finally:
            artifact.unlink(missing_ok=True)

    # Keeps missing feature rows when the model is configured to impute inside the pipeline.
    def test_train_model_imputes_missing_values(self):
        csv_text = "\n".join(
            [
                "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,heart_disease_binary",
                "63,1,1,145,,1,2,150,0,2.3,3,0,6,0",
                "67,1,4,160,286,0,2,108,1,1.5,2,3,3,1",
                "67,1,4,120,229,0,2,129,1,2.6,2,2,7,1",
                "59,1,3,150,260,1,0,120,0,1.0,2,0,3,0",
            ]
        )

        with TemporaryDirectory() as directory:
            data_path = Path(directory) / "missing_values.csv"
            data_path.write_text(csv_text, encoding="utf-8")

            result = train_model(
                data_path,
                method="random_forest",
                cv_folds=2,
                missing_strategy="impute",
                register=False,
                n_estimators=10,
                max_depth=5,
                min_samples_leaf=1,
            )
            artifact = PROJECT_ROOT / result["model_path"]
            try:
                self.assertEqual(result["missing_strategy"], "impute")
                self.assertEqual(result["rows"], 4)
                self.assertTrue(artifact.exists())
            finally:
                artifact.unlink(missing_ok=True)

    def test_train_model_imputes_missing_values_for_knn(self):
        with TemporaryDirectory() as directory:
            data_path = self.write_strategy_collapse_dataset(directory)

            result = train_model(
                data_path,
                method="knn",
                cv_folds=2,
                missing_strategy="impute",
                n_neighbors=1,
                register=False,
            )
            artifact = PROJECT_ROOT / result["model_path"]
            try:
                self.assertEqual(result["missing_strategy"], "impute")
                self.assertEqual(result["rows"], 4)
                self.assertTrue(artifact.exists())
            finally:
                artifact.unlink(missing_ok=True)

    def test_strategy_diagnostics_prevent_single_class_drop_training(self):
        with TemporaryDirectory() as directory:
            data_path = self.write_strategy_collapse_dataset(directory)
            summary = inspect_training_data(data_path, cv_folds=2)
            selection = summary["selection"]

            self.assertEqual(selection["class_counts_by_strategy"]["drop"], {1: 2})
            self.assertEqual(selection["class_counts_by_strategy"]["impute"], {0: 2, 1: 2})
            self.assertFalse(selection["folds_supported_by_strategy"]["drop"])
            self.assertTrue(selection["folds_supported_by_strategy"]["impute"])

            with self.assertRaisesRegex(ValueError, "selected fields.*drop.*one target class"):
                train_model(
                    data_path,
                    method="random_forest",
                    feature_fields=self.selected_feature_fields(),
                    cv_folds=2,
                    register=False,
                )

    def test_prepare_training_data_writes_selected_configuration(self):
        with TemporaryDirectory() as directory:
            data_path = self.write_strategy_collapse_dataset(directory)
            prepared_path = Path(directory) / "prepared.csv"

            result = prepare_training_data(
                data_path,
                prepared_path,
                feature_fields=self.selected_feature_fields(),
                missing_strategy="impute",
            )

            self.assertNotIn("feature_set", result)
            self.assertEqual(result["feature_fields"], self.selected_feature_fields())
            self.assertEqual(result["missing_strategy"], "impute")
            self.assertEqual(result["rows"], 4)
            self.assertEqual(result["training_row_ids"], [1, 2, 3, 4])
            self.assertTrue(prepared_path.exists())
            prepared = load_training_data(
                prepared_path,
                feature_fields=self.selected_feature_fields(),
                missing_strategy="impute",
            )
            self.assertEqual(set(prepared["target"].unique()), {0, 1})

    # Accepts the full prediction payload for a model trained on an explicit subset.
    def test_predict_uses_model_feature_subset(self):
        feature_fields = ["age", "sex", "cp", "chol"]
        result = train_model(
            DATA_PATH,
            method="random_forest",
            feature_fields=feature_fields,
            cv_folds=3,
            register=False,
            n_estimators=10,
            max_depth=5,
            min_samples_leaf=1,
        )
        artifact = PROJECT_ROOT / result["model_path"]
        try:
            model = load_model(model_path=result["model_path"])
            prediction, probability = predict(model, self.valid_values(), feature_fields=result["feature_fields"])
            self.assertIn(prediction, {0, 1})
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
        finally:
            artifact.unlink(missing_ok=True)

    # Uses grouped cross-validation when the dataset provides multiple source values.
    def test_train_model_uses_grouped_cross_validation_when_source_varies(self):
        csv_text = "\n".join(
            [
                "source,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,heart_disease_binary",
                "A,63,1,1,145,233,1,2,150,0,2.3,3,0,6,0",
                "B,67,1,4,160,286,0,2,108,1,1.5,2,3,3,1",
                "A,67,1,4,120,229,0,2,129,1,2.6,2,2,7,1",
                "B,59,1,3,150,260,1,0,120,0,1.0,2,0,3,0",
            ]
        )

        with TemporaryDirectory() as directory:
            data_path = Path(directory) / "grouped_sources.csv"
            data_path.write_text(csv_text, encoding="utf-8")

            result = train_model(
                data_path,
                method="random_forest",
                feature_fields=self.selected_feature_fields(),
                cv_folds=2,
                register=False,
                n_estimators=10,
                max_depth=5,
                min_samples_leaf=1,
            )
            artifact = PROJECT_ROOT / result["model_path"]
            try:
                self.assertEqual(result["cv_strategy"], "stratified_group")
                self.assertEqual(result["group_count"], 2)
                self.assertEqual(result["rows"], 4)
                self.assertTrue(artifact.exists())
            finally:
                artifact.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()