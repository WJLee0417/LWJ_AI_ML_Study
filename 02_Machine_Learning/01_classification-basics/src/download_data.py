from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(URL, OUTPUT)
print(f"Downloaded {OUTPUT}")
