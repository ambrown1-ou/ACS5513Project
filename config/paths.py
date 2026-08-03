import os
from pathlib import Path
from typing import Iterable

# Project-rooted paths (repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Bundled, read-only resources included with the repository
RESOURCES_ROOT = PROJECT_ROOT / "resources"
BUNDLED_DATASETS_DIR = RESOURCES_ROOT / "datasets"

# Runtime storage root (configurable via APP_STORAGE_ROOT)
_configured = os.environ.get("APP_STORAGE_ROOT")
if _configured:
    # If the operator configured a storage root, use it as provided (do not force a resolve
    # which may touch network shares). Expand ~ if present.
    STORAGE_ROOT = Path(_configured).expanduser()
else:
    # Default to a `storage/` folder inside the project root for local development.
    STORAGE_ROOT = (PROJECT_ROOT / "storage").resolve()

# Specific runtime subfolders
CONFIG_DIR = STORAGE_ROOT / "config"
DATASETS_DIR = STORAGE_ROOT / "datasets"
UPLOADED_DATASETS_DIR = DATASETS_DIR / "uploads"
MAPPED_DATASETS_DIR = DATASETS_DIR / "mapped"
PREPARED_DATASETS_DIR = DATASETS_DIR / "prepared"
GRAPHICS_DIR = STORAGE_ROOT / "graphics"
MODEL_ARTIFACTS_DIR = STORAGE_ROOT / "models"
REGISTRY_DIR = STORAGE_ROOT / "registry"
REPORTS_DIR = STORAGE_ROOT / "reports"

# Common file locations
DATASET_METADATA_PATH = CONFIG_DIR / "datasets_info.json"
REGISTRY_PATH = REGISTRY_DIR / "training_results.json"
DEFAULT_MODEL_PATH = MODEL_ARTIFACTS_DIR / "heart_disease_model.joblib"


def ensure_runtime_directories(dirs: Iterable[Path] = None):
    """Create the runtime storage directories if they do not exist.

    This does not create bundled resource directories.
    """
    targets = list(dirs) if dirs is not None else [
        CONFIG_DIR,
        DATASETS_DIR,
        UPLOADED_DATASETS_DIR,
        MAPPED_DATASETS_DIR,
        PREPARED_DATASETS_DIR,
        GRAPHICS_DIR,
        MODEL_ARTIFACTS_DIR,
        REGISTRY_DIR,
        REPORTS_DIR,
    ]
    for p in targets:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort creation; do not raise to avoid startup hard-failure for remote shares.
            pass


def relative_storage_path(path):
    """Return a storage-relative path (POSIX) for a given path.

    Attempts a safe conversion to a path relative to `STORAGE_ROOT`. If that
    fails (for example when the storage root is a remote UNC path), a
    best-effort relpath is returned.
    """
    p = Path(path)
    try:
        # Prefer resolving the candidate but do not force resolution of STORAGE_ROOT.
        resolved = p.resolve()
        try:
            return resolved.relative_to(STORAGE_ROOT).as_posix()
        except Exception:
            # Fallback to os.path.relpath which works for UNC and other roots.
            return Path(os.path.relpath(str(resolved), str(STORAGE_ROOT))).as_posix()
    except Exception:
        # If resolution fails, fall back to a textual relpath attempt.
        try:
            return Path(os.path.relpath(str(p), str(STORAGE_ROOT))).as_posix()
        except Exception:
            # As a final fallback, return the stringified path.
            return str(p)
