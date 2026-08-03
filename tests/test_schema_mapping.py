import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from model.dataset_mapping import apply_mapping, review_mapping, validate_mapping
from model.schema import SCHEMA_ID, get_schema


class SchemaMappingTests(unittest.TestCase):
    def test_schema_exposes_roles_aliases_units_and_ranges(self):
        schema = get_schema()
        self.assertEqual(schema["schema_id"], SCHEMA_ID)
        age = next(field for field in schema["fields"] if field["field"] == "age")
        blood_pressure = next(field for field in schema["fields"] if field["field"] == "trestbps")
        target = next(field for field in schema["fields"] if field["field"] == "target")
        self.assertEqual(age["role"], "feature")
        self.assertIn("patient_age", age["aliases"])
        self.assertEqual(blood_pressure["unit_options"], ["mmHg", "kPa"])
        self.assertEqual(blood_pressure["conversions"]["kPa"]["id"], "kPa_to_mmHg")
        self.assertEqual(target["role"], "classifier")
        self.assertEqual(
            {field["field"]: field["alias_label"] for field in schema["fields"]},
            {
                "age": "Age",
                "sex": "Sex",
                "cp": "Chest Pain",
                "trestbps": "Blood Pressure",
                "chol": "Cholesterol",
                "fbs": "Fasting Blood Sugar",
                "restecg": "Resting ECG",
                "thalach": "Maximum Heart Rate",
                "exang": "Exercise Angina",
                "oldpeak": "ST Depression",
                "slope": "ST Slope",
                "ca": "Major Vessels",
                "thal": "Thalassemia",
                "target": "Diagnosis",
            },
        )

    def test_mapping_converts_values_and_review_applies_decisions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.csv"
            mapped_path = root / "mapped.csv"
            reviewed_path = root / "reviewed.csv"
            source_path.write_text(
                "patient_age,resting_bp,cholesterol,sex,diagnosis,source\n"
                "63,16,5,1,0,A\n"
                "200,not-a-number,5,2,1,B\n"
                ",16,5,1,0,C\n",
                encoding="utf-8",
            )
            mapping = [
                {"source_column": "patient_age", "schema_field": "age"},
                {"source_column": "resting_bp", "schema_field": "trestbps", "source_unit": "kPa"},
                {"source_column": "cholesterol", "schema_field": "chol", "source_unit": "mmol/L"},
                {"source_column": "sex", "schema_field": "sex"},
                {"source_column": "diagnosis", "schema_field": "target"},
            ]

            result = apply_mapping(source_path, mapped_path, mapping)
            mapped = pd.read_csv(mapped_path)
            report = result["report"]

            self.assertEqual(mapped["source_row"].tolist(), [1, 2, 3])
            self.assertEqual(result["source_row_ids"], [1, 2, 3])
            self.assertAlmostEqual(mapped.loc[0, "trestbps"], 16 * 7.50061683, places=5)
            self.assertAlmostEqual(mapped.loc[0, "chol"], 5 * 38.66976, places=5)
            self.assertEqual(report["rows_with_missing_data"], 1)
            self.assertEqual(report["rows_with_conversion_errors"], 1)
            self.assertEqual(report["rows_with_out_of_range_values"], 1)
            self.assertEqual(report["total_rows_before_review"], 3)
            age_report = next(field for field in report["fields"] if field["schema_field"] == "age")
            self.assertEqual(age_report["alias"], "Age")
            blood_pressure_report = next(field for field in report["fields"] if field["schema_field"] == "trestbps")
            self.assertEqual(blood_pressure_report["alias"], "Blood Pressure")
            self.assertIn("patient_age", age_report["aliases"])
            self.assertIn(2, [sample["row"] for sample in report["sample_rows"]])
            self.assertEqual(set(report["missing_schema_fields"]), {
                "cp", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            })

            reviewed = review_mapping(
                mapped_path,
                reviewed_path,
                result,
                {
                    "missing_rows": "drop",
                    "missing_columns": "allow",
                    "out_of_range": "drop",
                    "conversion_errors": "drop",
                    "unmapped_columns": "keep",
                },
            )
            self.assertEqual(reviewed["rows_before"], 3)
            self.assertEqual(reviewed["rows_after"], 1)
            self.assertEqual(len(pd.read_csv(reviewed_path)), 1)

    def test_mapping_rejects_duplicate_schema_assignments(self):
        with self.assertRaisesRegex(ValueError, "mapped more than once"):
            validate_mapping(
                [
                    {"source_column": "one", "schema_field": "age"},
                    {"source_column": "two", "schema_field": "age"},
                ],
                ["one", "two"],
            )

    def test_field_review_decisions_replace_and_impute_only_issue_values(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.csv"
            mapped_path = root / "mapped.csv"
            reviewed_path = root / "reviewed.csv"
            source_path.write_text(
                "age,sex,diagnosis\n"
                "63,1,0\n"
                "200,2,1\n"
                ",1,0\n",
                encoding="utf-8",
            )
            result = apply_mapping(
                source_path,
                mapped_path,
                [
                    {"source_column": "age", "schema_field": "age"},
                    {"source_column": "sex", "schema_field": "sex"},
                    {"source_column": "diagnosis", "schema_field": "target"},
                ],
            )

            self.assertEqual(result["report"]["fields"][0]["schema_field"], "age")
            self.assertEqual(result["report"]["fields"][0]["missing_count"], 1)
            self.assertEqual(result["report"]["fields"][0]["invalid_count"], 1)
            field_decisions = {"age": "impute", "sex": "replace_null"}
            reviewed = review_mapping(
                mapped_path,
                reviewed_path,
                result,
                field_decisions=field_decisions,
            )
            reviewed_data = pd.read_csv(reviewed_path)
            self.assertEqual(reviewed["rows_after"], 3)
            self.assertEqual(reviewed_data.loc[1, "age"], 63)
            self.assertTrue(pd.isna(reviewed_data.loc[1, "sex"]))

    def test_review_tracks_selected_columns_and_explicit_source_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.csv"
            mapped_path = root / "mapped.csv"
            reviewed_path = root / "reviewed.csv"
            source_path.write_text(
                "age,sex,diagnosis,notes\n"
                "63,1,0,keep-a\n"
                "67,,1,drop-b\n"
                "59,0,0,keep-c\n",
                encoding="utf-8",
            )
            result = apply_mapping(
                source_path,
                mapped_path,
                [
                    {"source_column": "age", "schema_field": "age"},
                    {"source_column": "sex", "schema_field": "sex"},
                    {"source_column": "diagnosis", "schema_field": "target"},
                ],
            )

            reviewed = review_mapping(
                mapped_path,
                reviewed_path,
                result,
                field_decisions={"sex": "drop_rows"},
            )

            self.assertEqual(reviewed["source_row_ids"], [1, 2, 3])
            self.assertEqual(reviewed["selected_row_ids"], [1, 3])
            self.assertEqual(reviewed["dropped_row_ids"], [2])
            self.assertEqual(reviewed["selected_columns"], ["age", "sex", "target"])
            self.assertEqual(reviewed["feature_fields"], ["age", "sex"])
            self.assertEqual(pd.read_csv(reviewed_path)["source_row"].tolist(), [1, 3])

    def test_field_review_can_drop_an_invalid_column_without_dropping_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.csv"
            mapped_path = root / "mapped.csv"
            reviewed_path = root / "reviewed.csv"
            source_path.write_text(
                "age,sex,diagnosis\n"
                "63,1,0\n"
                "200,0,1\n"
                "59,1,0\n",
                encoding="utf-8",
            )
            result = apply_mapping(
                source_path,
                mapped_path,
                [
                    {"source_column": "age", "schema_field": "age"},
                    {"source_column": "sex", "schema_field": "sex"},
                    {"source_column": "diagnosis", "schema_field": "target"},
                ],
            )

            reviewed = review_mapping(
                mapped_path,
                reviewed_path,
                result,
                field_decisions={"age": "drop_column"},
            )
            reviewed_data = pd.read_csv(reviewed_path)

            self.assertEqual(reviewed["rows_before"], 3)
            self.assertEqual(reviewed["rows_after"], 3)
            self.assertNotIn("age", reviewed_data.columns)
            self.assertEqual(reviewed["selected_columns"], ["sex", "target"])
            self.assertEqual(reviewed["feature_fields"], ["sex"])
            self.assertEqual(reviewed["target_field"], "target")


if __name__ == "__main__":
    unittest.main()
