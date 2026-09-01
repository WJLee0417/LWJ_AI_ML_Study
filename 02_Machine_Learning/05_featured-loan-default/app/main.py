from pathlib import Path
import joblib,pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
ROOT=Path(__file__).resolve().parents[1]; model=joblib.load(ROOT/"models/loan_default_model.joblib"); app=FastAPI()
class Application(BaseModel):
 LIMIT_BAL:float; SEX:int; EDUCATION:int; MARRIAGE:int; AGE:int; PAY_0:int; PAY_2:int; PAY_3:int; PAY_4:int; PAY_5:int; PAY_6:int; BILL_AMT1:float; BILL_AMT2:float; BILL_AMT3:float; BILL_AMT4:float; BILL_AMT5:float; BILL_AMT6:float; PAY_AMT1:float; PAY_AMT2:float; PAY_AMT3:float; PAY_AMT4:float; PAY_AMT5:float; PAY_AMT6:float
@app.post("/predict")
def predict(item:Application):
 p=float(model.predict_proba(pd.DataFrame([item.model_dump()]))[0,1]); return {"default_probability":p,"risk_grade":"high" if p>=.7 else "medium" if p>=.3 else "low","warning":"심사 보조 결과이며 자동 승인·거절에 사용할 수 없습니다."}
