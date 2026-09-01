"""Select loan-default model and policy on validation; report final test once."""
from pathlib import Path
import json, joblib, pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; d=pd.read_csv(ROOT/"data/processed/loan_default_cleaned.csv"); y=d.pop("default_next_month")
xtr,xtmp,ytr,ytmp=train_test_split(d,y,test_size=.4,stratify=y,random_state=42)
xv,xte,yv,yte=train_test_split(xtmp,ytmp,test_size=.5,stratify=ytmp,random_state=42)
models={"baseline":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000))]),"balanced":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,class_weight="balanced"))]),"smote":ImbPipeline([("scale",StandardScaler()),("smote",SMOTE(random_state=42)),("model",LogisticRegression(max_iter=2000))]),"random_forest":RandomForestClassifier(n_estimators=300,min_samples_leaf=10,class_weight="balanced",random_state=42,n_jobs=-1)}
rows=[]; fitted={}
for name,m in models.items():
 m.fit(xtr,ytr); p=m.predict_proba(xv)[:,1]; rows.append({"model":name,"validation_pr_auc":average_precision_score(yv,p),"validation_recall":recall_score(yv,p>=.5),"validation_precision":precision_score(yv,p>=.5,zero_division=0)}); fitted[name]=m
comparison=pd.DataFrame(rows).sort_values("validation_pr_auc",ascending=False); selected=comparison.iloc[0].model; model=fitted[selected]; pv=model.predict_proba(xv)[:,1]
policy=pd.DataFrame([{"threshold":t,"validation_flagged":int((pv>=t).sum()),"validation_recall":recall_score(yv,pv>=t),"validation_precision":precision_score(yv,pv>=t,zero_division=0)} for t in (.3,.5,.7)])
threshold=.3; pt=model.predict_proba(xte)[:,1]; final={"model":selected,"threshold":threshold,"test_pr_auc":average_precision_score(yte,pt),"test_recall":recall_score(yte,pt>=threshold),"test_precision":precision_score(yte,pt>=threshold,zero_division=0)}
comparison.to_csv(ROOT/"results/model-comparison.csv",index=False); policy.to_csv(ROOT/"results/threshold-policy.csv",index=False); (ROOT/"results/final-test.json").write_text(json.dumps(final,indent=2),encoding="utf-8"); joblib.dump(model,ROOT/"models/loan_default_model.joblib")
