"""Download the public Olist dataset archive without storing it in Git."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve


SOURCE_URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw" / "olist-brazilian-ecommerce.zip"


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
    print(f"SHA-256: {sha256(DESTINATION)}")


if __name__ == "__main__":
    main()
