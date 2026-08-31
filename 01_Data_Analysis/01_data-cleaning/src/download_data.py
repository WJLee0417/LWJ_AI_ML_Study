"""Download the public UCI Online Retail source file without modifying it."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve


SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/"
    "Online%20Retail.xlsx"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION.exists():
        print(f"Downloading source data to: {DESTINATION}")
        urlretrieve(SOURCE_URL, DESTINATION)
    else:
        print(f"Using existing source data: {DESTINATION}")

    print(f"SHA-256: {sha256(DESTINATION)}")


if __name__ == "__main__":
    main()
