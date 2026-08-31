"""Download and extract the public UCI Seoul Bike Sharing Demand dataset."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


SOURCE_URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "seoul-bike-sharing-demand.zip"
CSV_PATH = RAW_DIR / "SeoulBikeData.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading source data to: {ZIP_PATH}")
        urlretrieve(SOURCE_URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        csv_member = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_member) as source, CSV_PATH.open("wb") as target:
            shutil.copyfileobj(source, target)

    print(f"Source archive SHA-256: {sha256(ZIP_PATH)}")
    print(f"Extracted CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
