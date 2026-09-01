from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
payload={"LIMIT_BAL":50000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":30,"PAY_0":0,"PAY_2":0,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,"BILL_AMT1":10000,"BILL_AMT2":10000,"BILL_AMT3":10000,"BILL_AMT4":10000,"BILL_AMT5":10000,"BILL_AMT6":10000,"PAY_AMT1":1000,"PAY_AMT2":1000,"PAY_AMT3":1000,"PAY_AMT4":1000,"PAY_AMT5":1000,"PAY_AMT6":1000}
r=client.post("/predict",json=payload)
assert r.status_code==200 and 0<=r.json()["default_probability"]<=1
assert client.post("/predict",json={}).status_code==422
