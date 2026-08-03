import unittest
from app import app


class ApiInspectTests(unittest.TestCase):
    def test_inspect_returns_ok_for_existing_dataset(self):
        # Uses the shipped Cleveland cleaned CSV in resources/datasets/.
        response = app.test_client().get("/api/inspect?dataset=heart_disease_cleveland_cleaned")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("dataset_key"), "heart_disease_cleveland_cleaned")

    def test_inspect_returns_404_for_missing_dataset(self):
        response = app.test_client().get("/api/inspect?dataset=this_dataset_does_not_exist")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("code", data)
        self.assertEqual(data.get("code"), "DATASET_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
