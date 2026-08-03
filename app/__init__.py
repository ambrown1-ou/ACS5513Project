import sys

from pathlib import Path

from flask import Flask


from .api import api
from .routes import web
from config import paths as project_paths

PACKAGE_ROOT = Path(__file__).resolve().parent


def create_app():
    # Use package-local templates and static assets inside the `app/` folder.
    app = Flask(
        __name__,
        template_folder=str(PACKAGE_ROOT / "templates"),
        static_folder=str(PACKAGE_ROOT / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.secret_key = "heart-disease-app"

    # Ensure runtime storage directories exist (best-effort).
    try:
        project_paths.ensure_runtime_directories()
        # Expose the configured storage root for diagnostics (not secrets).
        app.config["STORAGE_ROOT"] = str(project_paths.STORAGE_ROOT)
    except Exception:
        # Avoid crashing the app on startup; errors will surface when storage is used.
        app.config["STORAGE_ROOT"] = None

    app.register_blueprint(web)
    app.register_blueprint(api)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)