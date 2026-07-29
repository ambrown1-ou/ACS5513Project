import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app import app
import app.routes as routes


class RouteTests(unittest.TestCase):
    # Rejects traversal filenames while keeping sanitized uploads inside inputs.
    def test_upload_filename_stays_inside_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs_dir = root / "inputs"
            inputs_dir.mkdir()
            info_path = root / "datasets_info.json"
            info_path.write_text("{}", encoding="utf-8")

            with patch.object(routes, "INPUTS_DIR", inputs_dir), patch.object(routes, "INFO_PATH", info_path):
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

            with patch.object(routes, "INPUTS_DIR", inputs_dir), patch.object(routes, "INFO_PATH", info_path), patch.object(
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

    # Returns a client error before model execution for incomplete API input.
    def test_api_rejects_incomplete_payload(self):
        response = app.test_client().post("/api/predict", json={"unexpected": 1})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing fields", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()