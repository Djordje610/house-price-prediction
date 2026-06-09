# -*- coding: utf-8 -*-
"""Jedna komanda za frontend: python run.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "ridge_model.joblib"


def main() -> None:
    if not MODEL.exists():
        print("Prvi put - cuvam modele...")
        subprocess.run([sys.executable, str(ROOT / "shared" / "save_models.py")], check=True)

    print("Frontend: http://localhost:8501  (Ctrl+C za gasenje)\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "app.py")],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
