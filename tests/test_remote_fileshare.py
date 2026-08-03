import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.remote_fileshare import RemoteFileshare


class RemoteFileshareTests(unittest.TestCase):
    def test_add_sync_delete_and_get_all_work_with_local_backend(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_dir = temp_path / "cache"
            fileshare = RemoteFileshare(backend="local", local_cache_dir=str(cache_dir))

            source_file = temp_path / "dataset.csv"
            source_file.write_text("age,target\n63,1\n", encoding="utf-8")

            result = fileshare.add(source_file)
            self.assertEqual(result["name"], "dataset.csv")
            self.assertTrue((cache_dir / "dataset.csv").exists())

            synced_files = fileshare.sync()
            self.assertEqual(len(synced_files), 1)
            self.assertEqual(synced_files[0]["name"], "dataset.csv")

            all_files = fileshare.get_all()
            self.assertEqual(len(all_files), 1)

            deleted = fileshare.delete("dataset.csv")
            self.assertTrue(deleted)
            self.assertEqual(fileshare.get_all(), [])

    def test_onedrive_and_sharepoint_backends_require_graph_configuration(self):
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"

            for backend in ("onedrive", "sharepoint"):
                fileshare = RemoteFileshare(backend=backend, local_cache_dir=str(cache_dir))
                with self.assertRaisesRegex(RuntimeError, "Microsoft Graph"):
                    fileshare.sync()


if __name__ == "__main__":
    unittest.main()
