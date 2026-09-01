from pathlib import Path
import joblib,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,precision_score,recall_score
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/"data/processed/loan_default_cleaned.csv"); y=d.pop("default_next_month")
xtr,xte,ytr,yte=train_test_split(d,y,test_size=.2,stratify=y,random_state=42)
rows=[]; fitted={}
for name,w in [("baseline",None),("balanced","balanced")]:
 m=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,class_weight=w))]).fit(xtr,ytr)
 p=m.predict_proba(xte)[:,1]; rows.append({"model":name,"pr_auc":average_precision_score(yte,p),"recall":recall_score(yte,p>=.5),"precision":precision_score(yte,p>=.5,zero_division=0)}); fitted[name]=m
out=pd.DataFrame(rows).sort_values("pr_auc",ascending=False); out.to_csv(ROOT/"results/model-comparison.csv",index=False)
m=fitted[out.iloc[0].model]; joblib.dump(m,ROOT/"models/loan_default_model.joblib"); p=m.predict_proba(xte)[:,1]
pd.DataFrame([{"threshold":t,"flagged":int((p>=t).sum()),"recall":recall_score(yte,p>=t),"precision":precision_score(yte,p>=t,zero_division=0)} for t in (.3,.5,.7)]).to_csv(ROOT/"results/threshold-policy.csv",index=False)
