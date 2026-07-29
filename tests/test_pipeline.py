import unittest
from pathlib import Path

from model.pipeline import FEATURE_FIELDS, load_training_data, train_model, validate_feature_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "inputs" / "heart_disease_cleveland_cleaned.csv"


class PipelineTests(unittest.TestCase):
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
        result = train_model(DATA_PATH, method="knn", cv_folds=3, register=False)
        artifact = PROJECT_ROOT / result["model_path"]
        try:
            self.assertEqual(result["cv_folds"], 3)
            self.assertIn("accuracy_std", result)
            self.assertIn("f1_std", result)
            self.assertTrue(artifact.exists())
        finally:
            artifact.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()