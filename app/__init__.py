import sys

from pathlib import Path

from flask import Flask

from .api import api
from .routes import web

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app():
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.secret_key = "heart-disease-app"
    app.register_blueprint(web)
    app.register_blueprint(api)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)