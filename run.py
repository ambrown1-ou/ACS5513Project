import argparse
import subprocess
import sys
from pathlib import Path

from app import app


PROJECT_ROOT = Path(__file__).resolve().parent


# Starts JupyterLab in the project root when the launcher flag is enabled.
def start_jupyter_lab():
    try:
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
    except OSError as error:
        raise RuntimeError(
            "Failed to start JupyterLab from run.py: the 'jupyter' command could not be launched."
        ) from error


# Parses launcher flags and runs the Flask app with optional JupyterLab support.
def main():
    parser = argparse.ArgumentParser(description="Launch the heart disease dashboard.")
    parser.add_argument(
        "--jupyter",
        action="store_true",
        help="Start JupyterLab alongside the Flask app.",
    )
    args = parser.parse_args()

    jupyter_process = start_jupyter_lab() if args.jupyter else None

    try:
        app.run(debug=True, use_reloader=False)
    finally:
        if jupyter_process is not None:
            jupyter_process.terminate()
            jupyter_process.wait()


if __name__ == "__main__":
    main()