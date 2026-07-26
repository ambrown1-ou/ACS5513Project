import subprocess
import sys
from pathlib import Path

from app import app


PROJECT_ROOT = Path(__file__).resolve().parent


def start_jupyter_lab():
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jupyter",
            "lab",
            f"--notebook-dir={PROJECT_ROOT}",
        ],
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    jupyter_process = start_jupyter_lab()
    try:
        app.run(debug=True, use_reloader=False)
    finally:
        jupyter_process.terminate()
        jupyter_process.wait()