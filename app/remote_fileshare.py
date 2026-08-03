import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv


load_dotenv()


class RemoteFileshare:
    """Storage-agnostic remote file helper for local, S3, Google Drive, OneDrive, or SharePoint backends."""

    def __init__(self, backend: Optional[str] = None, local_cache_dir: Optional[str] = None):
        self.backend = (backend or os.getenv("REMOTE_FILESHARE_BACKEND", "local")).lower()
        self.local_cache_dir = Path(local_cache_dir or os.getenv("REMOTE_FILESHARE_LOCAL_CACHE_DIR", "./.remote-cache"))
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    def sync(self) -> List[Dict[str, object]]:
        """Sync the local cache to the configured backend and return metadata."""
        if self.backend == "local":
            return self._sync_local()
        if self.backend == "s3":
            return self._sync_s3()
        if self.backend == "gdrive":
            return self._sync_gdrive()
        if self.backend in {"onedrive", "sharepoint"}:
            return self._sync_microsoft_graph()
        raise ValueError(f"Unsupported backend: {self.backend}")

    def add(self, source_path, remote_path: Optional[str] = None, overwrite: bool = False) -> Dict[str, object]:
        """Upload a local file to the configured backend."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file does not exist: {source}")

        if self.backend == "local":
            return self._add_local(source, remote_path or source.name, overwrite)
        if self.backend == "s3":
            return self._add_s3(source, remote_path or source.name, overwrite)
        if self.backend == "gdrive":
            return self._add_gdrive(source, remote_path or source.name, overwrite)
        if self.backend in {"onedrive", "sharepoint"}:
            return self._add_microsoft_graph(source, remote_path or source.name, overwrite)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def delete(self, remote_path: str) -> bool:
        """Delete a file from the configured backend."""
        if self.backend == "local":
            return self._delete_local(remote_path)
        if self.backend == "s3":
            return self._delete_s3(remote_path)
        if self.backend == "gdrive":
            return self._delete_gdrive(remote_path)
        if self.backend in {"onedrive", "sharepoint"}:
            return self._delete_microsoft_graph(remote_path)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def get_all(self) -> List[Dict[str, object]]:
        """Return metadata for all known files."""
        if self.backend == "local":
            return self._get_all_local()
        if self.backend == "s3":
            return self._get_all_s3()
        if self.backend == "gdrive":
            return self._get_all_gdrive()
        if self.backend in {"onedrive", "sharepoint"}:
            return self._get_all_microsoft_graph()
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _sync_local(self) -> List[Dict[str, object]]:
        files: List[Dict[str, object]] = []
        for path in sorted(self.local_cache_dir.iterdir()):
            if path.is_file():
                files.append({
                    "name": path.name,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                })
        return files

    def _add_local(self, source: Path, remote_path: str, overwrite: bool) -> Dict[str, object]:
        destination = self.local_cache_dir / remote_path
        if destination.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {remote_path}")
        destination.write_bytes(source.read_bytes())
        return {
            "name": destination.name,
            "path": str(destination),
            "size": destination.stat().st_size,
            "modified": destination.stat().st_mtime,
        }

    def _delete_local(self, remote_path: str) -> bool:
        target = self.local_cache_dir / remote_path
        if target.exists() and target.is_file():
            target.unlink()
            return True
        return False

    def _get_all_local(self) -> List[Dict[str, object]]:
        return self._sync_local()

    def _sync_s3(self) -> List[Dict[str, object]]:
        raise NotImplementedError("S3 backend is not implemented yet")

    def _add_s3(self, source: Path, remote_path: str, overwrite: bool) -> Dict[str, object]:
        raise NotImplementedError("S3 backend is not implemented yet")

    def _delete_s3(self, remote_path: str) -> bool:
        raise NotImplementedError("S3 backend is not implemented yet")

    def _get_all_s3(self) -> List[Dict[str, object]]:
        raise NotImplementedError("S3 backend is not implemented yet")

    def _sync_gdrive(self) -> List[Dict[str, object]]:
        raise NotImplementedError("Google Drive backend is not implemented yet")

    def _add_gdrive(self, source: Path, remote_path: str, overwrite: bool) -> Dict[str, object]:
        raise NotImplementedError("Google Drive backend is not implemented yet")

    def _delete_gdrive(self, remote_path: str) -> bool:
        raise NotImplementedError("Google Drive backend is not implemented yet")

    def _get_all_gdrive(self) -> List[Dict[str, object]]:
        raise NotImplementedError("Google Drive backend is not implemented yet")

    def _sync_microsoft_graph(self) -> List[Dict[str, object]]:
        raise RuntimeError(
            "Microsoft Graph backend is not configured yet. Set MICROSOFT_TENANT_ID, MICROSOFT_CLIENT_ID, "
            "MICROSOFT_CLIENT_SECRET, and MICROSOFT_DRIVE_ID to enable OneDrive or SharePoint support."
        )

    def _add_microsoft_graph(self, source: Path, remote_path: str, overwrite: bool) -> Dict[str, object]:
        self._sync_microsoft_graph()
        raise AssertionError("Unreachable")

    def _delete_microsoft_graph(self, remote_path: str) -> bool:
        self._sync_microsoft_graph()
        raise AssertionError("Unreachable")

    def _get_all_microsoft_graph(self) -> List[Dict[str, object]]:
        self._sync_microsoft_graph()
        raise AssertionError("Unreachable")
