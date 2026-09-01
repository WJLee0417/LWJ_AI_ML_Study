from copy import deepcopy

import numpy as np
from fastapi.testclient import TestClient

import app.main as api


client = TestClient(api.app)
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
assert isinstance(body["policy_reason"], str)
assert isinstance(body["top_risk_signals"], list)

assert client.post("/predict", json={}).status_code == 422
assert_invalid("LIMIT_BAL", 0)
assert_invalid("AGE", 17)
assert_invalid("AGE", 101)
assert_invalid("PAY_0", -3)
assert_invalid("PAY_0", 10)
assert_invalid("EDUCATION", 7)
assert_invalid("MARRIAGE", 4)


class StubModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, _):
        return np.array([[1 - self.probability, self.probability]])


policy_cases = [
    (0.29, "low", "승인 보조 의견", 0.3, "0.30 미만"),
    (0.30, "medium", "추가 심사", 0.3, "0.30 이상 0.70 미만"),
    (0.69, "medium", "추가 심사", 0.3, "0.30 이상 0.70 미만"),
    (0.70, "high", "고위험 추가 심사", 0.7, "0.70 이상"),
]
original_model = api.model
try:
    for probability, grade, decision, threshold, reason in policy_cases:
        api.model = StubModel(probability)
        policy_response = client.post("/predict", json=payload)
        policy_body = policy_response.json()
        assert policy_response.status_code == 200
        assert policy_body["default_probability"] == probability
        assert policy_body["risk_grade"] == grade
        assert policy_body["decision_support"] == decision
        assert policy_body["policy_threshold"] == threshold
        assert reason in policy_body["policy_reason"]
finally:
    api.model = original_model
