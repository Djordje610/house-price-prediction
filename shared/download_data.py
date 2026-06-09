# -*- coding: utf-8 -*-
"""Preuzima Ames Housing train.csv i test.csv (Kaggle format)."""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

FILES = {
    "train.csv": "https://raw.githubusercontent.com/caiks/AMES/master/train.csv",
    "test.csv": "https://raw.githubusercontent.com/caiks/AMES/master/test.csv",
}


def download_one(filename: str, url: str) -> None:
    out = DATA_DIR / filename
    if out.exists():
        print(f"Vec postoji: {out}")
        return
    print(f"Preuzimam {filename} ...")
    urllib.request.urlretrieve(url, out)
    print(f"Sacuvano: {out}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        download_one(filename, url)
    print("Gotovo.")


if __name__ == "__main__":
    main()
