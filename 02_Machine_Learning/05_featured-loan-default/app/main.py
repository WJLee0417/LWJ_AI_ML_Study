from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models" / "loan_default_model.joblib")
app = FastAPI(title="Loan Default Risk API")


class Application(BaseModel):
    LIMIT_BAL: float = Field(gt=0, description="신용한도는 0보다 커야 합니다.")
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)

    PAY_0: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


def build_risk_signals(item: Application) -> list[str]:
    signals: list[str] = []
    payment_statuses = [item.PAY_0, item.PAY_2, item.PAY_3, item.PAY_4, item.PAY_5, item.PAY_6]
    total_bill = sum(
        [item.BILL_AMT1, item.BILL_AMT2, item.BILL_AMT3, item.BILL_AMT4, item.BILL_AMT5, item.BILL_AMT6]
    )
    total_payment = sum(
        [item.PAY_AMT1, item.PAY_AMT2, item.PAY_AMT3, item.PAY_AMT4, item.PAY_AMT5, item.PAY_AMT6]
    )

    if max(payment_statuses) > 0:
        signals.append("최근 상환 상태 지연")
    if total_bill > 0 and total_payment / total_bill < 0.1:
        signals.append("상환액 대비 청구액 비율 낮음")
    if item.BILL_AMT1 / item.LIMIT_BAL > 0.7:
        signals.append("신용한도 대비 최근 청구액 높음")

    return signals or ["입력된 최근 청구·상환 정보 기준 추가 위험 신호 없음"]


def build_decision(probability: float) -> dict[str, str | float]:
    if probability >= 0.7:
        return {
            "risk_grade": "high",
            "decision_support": "고위험 추가 심사",
            "policy_threshold": 0.7,
            "policy_reason": "연체 확률이 0.70 이상이므로 고위험 추가 심사 대상입니다.",
        }
    if probability >= 0.3:
        return {
            "risk_grade": "medium",
            "decision_support": "추가 심사",
            "policy_threshold": 0.3,
            "policy_reason": "연체 확률이 0.30 이상 0.70 미만이므로 추가 심사 대상입니다.",
        }
    return {
        "risk_grade": "low",
        "decision_support": "승인 보조 의견",
        "policy_threshold": 0.3,
        "policy_reason": "연체 확률이 추가 심사 기준인 0.30 미만입니다.",
    }


@app.post("/predict")
def predict(item: Application):
    probability = float(model.predict_proba(pd.DataFrame([item.model_dump()]))[0, 1])
    decision = build_decision(probability)

    return {
        "default_probability": probability,
        **decision,
        "top_risk_signals": build_risk_signals(item),
        "warning": "심사 보조 결과이며 자동 승인·거절에 사용할 수 없습니다.",
    }
