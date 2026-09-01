from copy import deepcopy

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
payload = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 30,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 10000,
    "BILL_AMT2": 10000,
    "BILL_AMT3": 10000,
    "BILL_AMT4": 10000,
    "BILL_AMT5": 10000,
    "BILL_AMT6": 10000,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 1000,
    "PAY_AMT4": 1000,
    "PAY_AMT5": 1000,
    "PAY_AMT6": 1000,
}


def assert_invalid(field: str, value: int | float) -> None:
    invalid_payload = deepcopy(payload)
    invalid_payload[field] = value
    assert client.post("/predict", json=invalid_payload).status_code == 422


response = client.post("/predict", json=payload)
body = response.json()
assert response.status_code == 200
assert 0 <= body["default_probability"] <= 1
assert body["risk_grade"] in {"low", "medium", "high"}
assert body["decision_support"] in {"승인 보조 의견", "추가 심사", "고위험 추가 심사"}
assert isinstance(body["top_risk_signals"], list)

assert client.post("/predict", json={}).status_code == 422
assert_invalid("LIMIT_BAL", 0)
assert_invalid("AGE", 17)
assert_invalid("AGE", 101)
assert_invalid("PAY_0", -3)
assert_invalid("PAY_0", 10)
assert_invalid("EDUCATION", 7)
assert_invalid("MARRIAGE", 4)
