"""Launch the Nominal Drift GUI."""
import subprocess
import sys
from pathlib import Path


def launch():
    gui_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(gui_path)])


if __name__ == "__main__":
    launch()
