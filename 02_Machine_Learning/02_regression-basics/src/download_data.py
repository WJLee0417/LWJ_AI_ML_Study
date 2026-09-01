from pathlib import Path
from urllib.request import urlretrieve

OUTPUT = Path(__file__).resolve().parents[1] / "data/raw/housing.csv"
urlretrieve("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv", OUTPUT)
print(f"Downloaded {OUTPUT}")
