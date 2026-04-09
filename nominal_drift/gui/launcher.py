"""Launch the Nominal Drift GUI.

Usage:
    python -m nominal_drift.gui.launcher
"""
import shutil
import subprocess
import sys
from pathlib import Path


def _check_streamlit() -> None:
    """Ensure streamlit is importable; give a clear message if not."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "ERROR: Streamlit is not installed.\n"
            "Run:  pip install streamlit\n"
            "  or: pip install -e .  (from the nominal-drift root directory)",
            file=sys.stderr,
        )
        sys.exit(1)


def launch() -> None:
    """Launch the Nominal Drift Streamlit app."""
    _check_streamlit()

    gui_path = Path(__file__).resolve().parent / "app.py"
    if not gui_path.exists():
        print(f"ERROR: GUI entry point not found: {gui_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run", str(gui_path),
        "--server.headless", "true",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nNominal Drift GUI stopped.")
    except FileNotFoundError:
        print(f"ERROR: Could not execute: {sys.executable}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    launch()
